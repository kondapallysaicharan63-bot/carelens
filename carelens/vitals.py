"""Contactless heart rate (rPPG, green channel) and breathing rate (chest motion).

Both estimators only report when the person is still and the region is well lit,
and every estimate carries a quality score in [0, 1]. These are wellness trends,
not measurements for diagnosis.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np
from scipy import signal

from . import pose as P


# ---------------------------------------------------------------- signal core

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x)


def dominant_frequency(x: np.ndarray, fs: float, lo: float, hi: float):
    """Return (peak_hz, quality) from a Welch spectrum restricted to [lo, hi].

    quality = share of in-band power within +/- 10% of the peak (and its first
    harmonic), mapped to 0..1. A clean sinusoid scores close to 1; noise ~0.1-0.2.
    """
    n = len(x)
    nperseg = min(n, int(fs * max(8.0, 2.5 / lo)))
    nfft = max(4096, 1 << int(np.ceil(np.log2(n * 4))))
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft,
                          detrend="linear")
    band = (f >= lo) & (f <= hi)
    if not band.any() or pxx[band].sum() <= 0:
        return None, 0.0
    fb, pb = f[band], pxx[band]
    peak = float(fb[np.argmax(pb)])
    win = max(0.1 * peak, 2 * (f[1] - f[0]))
    near = (np.abs(fb - peak) <= win) | (np.abs(fb - 2 * peak) <= win)
    ratio = float(pb[near].sum() / pb.sum())
    # Expected share for white noise is roughly the fraction of the band covered.
    chance = float(near.mean())
    quality = float(np.clip((ratio - chance) / (1 - chance + 1e-9), 0, 1))
    return peak, quality


def _clip_box(x0, y0, x1, y1, w, h):
    x0, x1 = int(max(0, min(x0, x1))), int(min(w, max(x0, x1)))
    y0, y1 = int(max(0, min(y0, y1))), int(min(h, max(y0, y1)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def face_roi(lm: np.ndarray, frame_h: int, frame_w: int):
    """Cheek/nose skin patch from pose face landmarks. lm is in frame-height units."""
    if lm[[P.NOSE, P.L_EYE, P.R_EYE], 2].mean() < 0.5:
        return None
    px = lm[:, :2] * frame_h
    eyes = (px[P.L_EYE] + px[P.R_EYE]) / 2
    nose = px[P.NOSE]
    eye_d = np.linalg.norm(px[P.L_EYE] - px[P.R_EYE])
    if eye_d < 6:
        return None
    cx = (eyes[0] + nose[0]) / 2
    top = eyes[1] + 0.25 * (nose[1] - eyes[1])
    return _clip_box(cx - 0.9 * eye_d, top, cx + 0.9 * eye_d, nose[1] + 0.35 * eye_d,
                     frame_w, frame_h)


def chest_roi(lm: np.ndarray, frame_h: int, frame_w: int):
    """Upper torso box between shoulders and hips."""
    if lm[[P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP], 2].mean() < 0.5:
        return None
    px = lm[:, :2] * frame_h
    sh = (px[P.L_SHOULDER] + px[P.R_SHOULDER]) / 2
    hip = (px[P.L_HIP] + px[P.R_HIP]) / 2
    sw = np.linalg.norm(px[P.L_SHOULDER] - px[P.R_SHOULDER])
    torso = np.linalg.norm(hip - sh)
    half_w = max(sw * 0.45, torso * 0.3)
    c = sh + 0.35 * (hip - sh)
    return _clip_box(c[0] - half_w, c[1] - 0.3 * torso, c[0] + half_w, c[1] + 0.3 * torso,
                     frame_w, frame_h)


# ---------------------------------------------------------------- estimators

class _WindowEstimator:
    def __init__(self, window_s, step_s, lo, hi, min_quality, still_motion, per_min):
        self.window_s, self.step_s = window_s, step_s
        self.lo, self.hi = lo, hi
        self.min_quality = min_quality
        self.still_motion = still_motion
        self.per_min = per_min
        self._buf: deque = deque()
        self._last_emit = -1e9
        self.estimates: list = []

    def _reset(self):
        self._buf.clear()

    def _push(self, t, value):
        self._buf.append((t, value))
        while self._buf and self._buf[0][0] < t - self.window_s:
            self._buf.popleft()

    def _maybe_estimate(self, t, extra=None):
        if t - self._last_emit < self.step_s or len(self._buf) < 10:
            return None
        ts = np.array([b[0] for b in self._buf])
        if ts[-1] - ts[0] < self.window_s * 0.9:
            return None
        self._last_emit = t
        fs = (len(ts) - 1) / (ts[-1] - ts[0])
        grid = np.linspace(ts[0], ts[-1], len(ts))
        x = np.interp(grid, ts, np.array([b[1] for b in self._buf]))
        x = signal.detrend(x)
        if np.allclose(x, 0):
            return None
        xf = bandpass(x, fs, self.lo, self.hi)
        f, q = dominant_frequency(xf, fs, self.lo, self.hi)
        if f is None:
            return None
        est = {"t": round(t, 2), "value": round(f * 60.0, 1), "quality": round(q, 3),
               "reported": q >= self.min_quality, **(extra or {})}
        self.estimates.append(est)
        return est


class HeartRateEstimator(_WindowEstimator):
    """rPPG: mean green value of a face skin patch, band-passed 0.7-3 Hz (42-180 bpm)."""

    def __init__(self, window_s=15.0, step_s=1.0, min_quality=0.35, still_motion=0.08,
                 min_brightness=40, max_brightness=230):
        super().__init__(window_s, step_s, 0.7, 3.0, min_quality, still_motion, True)
        self.min_b, self.max_b = min_brightness, max_brightness

    def update(self, t, image, state):
        if not state.present or state.motion > self.still_motion:
            self._reset(); return None
        roi = face_roi(state.landmarks, image.shape[0], image.shape[1])
        if roi is None:
            self._reset(); return None
        x0, y0, x1, y1 = roi
        patch = image[y0:y1, x0:x1]
        brightness = float(patch.mean())
        if not (self.min_b <= brightness <= self.max_b):
            self._reset(); return None
        self._push(t, float(patch[:, :, 1].mean()))   # green channel (BGR index 1)
        return self._maybe_estimate(t, {"brightness": round(brightness, 1)})


class BreathingRateEstimator(_WindowEstimator):
    """Vertical chest motion (phase correlation between frames), band-passed
    0.1-0.5 Hz (6-30 breaths/min)."""

    def __init__(self, window_s=30.0, step_s=2.0, min_quality=0.35, still_motion=0.08):
        super().__init__(window_s, step_s, 0.1, 0.5, min_quality, still_motion, True)
        self._prev_gray = None
        self._pos = 0.0

    def _reset(self):
        super()._reset()
        self._prev_gray, self._pos = None, 0.0

    def update(self, t, image, state):
        if not state.present or state.motion > self.still_motion:
            self._reset(); return None
        roi = chest_roi(state.landmarks, image.shape[0], image.shape[1])
        if roi is None:
            self._reset(); return None
        x0, y0, x1, y1 = roi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev_gray is not None:
            a = self._prev_gray[y0:y1, x0:x1]
            b = gray[y0:y1, x0:x1]
            win = cv2.createHanningWindow((b.shape[1], b.shape[0]), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
            self._pos += dy
        self._prev_gray = gray
        self._push(t, self._pos)
        return self._maybe_estimate(t)
