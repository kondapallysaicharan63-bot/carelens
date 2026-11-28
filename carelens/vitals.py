"""Contactless heart rate (rPPG: POS on the face ROI's mean RGB, or the green channel) and breathing rate.

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
from .params import PARAMS
from .rppg import FaceRPPG

_V = PARAMS.vitals
_SIG = _V.signal
_ROI = _V.roi
_COMMON = PARAMS.common


# ---------------------------------------------------------------- signal core

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = _SIG.bandpass_order) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, nyq * _SIG.bandpass_nyquist_frac)
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x)


def dominant_frequency(x: np.ndarray, fs: float, lo: float, hi: float):
    """Return (peak_hz, quality) from a Welch spectrum restricted to [lo, hi].

    quality = share of in-band power within +/- 10% of the peak (and its first
    harmonic), mapped to 0..1. A clean sinusoid scores close to 1; noise ~0.1-0.2.
    """
    n = len(x)
    nperseg = min(n, int(fs * max(_SIG.welch_min_segment_s, _SIG.welch_segment_cycles / lo)))
    nfft = max(_SIG.welch_nfft_min, 1 << int(np.ceil(np.log2(n * _SIG.welch_nfft_oversample))))
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft,
                          detrend="linear")
    band = (f >= lo) & (f <= hi)
    if not band.any() or pxx[band].sum() <= 0:
        return None, 0.0
    fb, pb = f[band], pxx[band]
    peak = float(fb[np.argmax(pb)])
    win = max(_SIG.peak_window_frac * peak, 2 * (f[1] - f[0]))
    near = (np.abs(fb - peak) <= win) | (np.abs(fb - 2 * peak) <= win)
    ratio = float(pb[near].sum() / pb.sum())
    # Expected share for white noise is roughly the fraction of the band covered.
    chance = float(near.mean())
    quality = float(np.clip((ratio - chance) / (1 - chance + 1e-9), 0, 1))
    return peak, quality


def _clip_box(x0, y0, x1, y1, w, h):
    x0, x1 = int(max(0, min(x0, x1))), int(min(w, max(x0, x1)))
    y0, y1 = int(max(0, min(y0, y1))), int(min(h, max(y0, y1)))
    if x1 - x0 < _ROI.min_box_px or y1 - y0 < _ROI.min_box_px:
        return None
    return x0, y0, x1, y1


def face_roi(lm: np.ndarray, frame_h: int, frame_w: int):
    """Cheek/nose skin patch from pose face landmarks. lm is in frame-height units."""
    if lm[[P.NOSE, P.L_EYE, P.R_EYE], 2].mean() < _ROI.min_landmark_visibility:
        return None
    px = lm[:, :2] * frame_h
    eyes = (px[P.L_EYE] + px[P.R_EYE]) / 2
    nose = px[P.NOSE]
    eye_d = np.linalg.norm(px[P.L_EYE] - px[P.R_EYE])
    if eye_d < _ROI.face_min_eye_distance_px:
        return None
    cx = (eyes[0] + nose[0]) / 2
    top = eyes[1] + _ROI.face_top_eye_to_nose_frac * (nose[1] - eyes[1])
    hw = _ROI.face_half_width_eye_dist * eye_d
    return _clip_box(cx - hw, top, cx + hw, nose[1] + _ROI.face_bottom_below_nose_eye_dist * eye_d,
                     frame_w, frame_h)


def chest_roi(lm: np.ndarray, frame_h: int, frame_w: int):
    """Upper torso box between shoulders and hips."""
    if lm[[P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP], 2].mean() < _ROI.min_landmark_visibility:
        return None
    px = lm[:, :2] * frame_h
    sh = (px[P.L_SHOULDER] + px[P.R_SHOULDER]) / 2
    hip = (px[P.L_HIP] + px[P.R_HIP]) / 2
    sw = np.linalg.norm(px[P.L_SHOULDER] - px[P.R_SHOULDER])
    torso = np.linalg.norm(hip - sh)
    half_w = max(sw * _ROI.chest_half_width_shoulder_frac, torso * _ROI.chest_half_width_torso_frac)
    c = sh + _ROI.chest_center_along_torso_frac * (hip - sh)
    hh = _ROI.chest_half_height_torso_frac * torso
    return _clip_box(c[0] - half_w, c[1] - hh, c[0] + half_w, c[1] + hh,
                     frame_w, frame_h)


def estimate_window(ts, values, lo: float, hi: float, min_quality: float) -> Optional[dict]:
    """Pure per-window estimate on an ROI-level trace: resample, detrend, band-pass, spectrum.

    Returns {"value" (per minute), "quality", "reported"} or None. This is the single code path
    used by the estimators and by the shared fixtures."""
    ts = np.asarray(ts, dtype=float)
    if len(ts) < max(2, _SIG.estimator_min_samples) or not ts[-1] > ts[0]:
        return None
    fs = (len(ts) - 1) / (ts[-1] - ts[0])
    grid = np.linspace(ts[0], ts[-1], len(ts))
    x = np.interp(grid, ts, np.asarray(values, dtype=float))
    x = signal.detrend(x)
    if np.allclose(x, 0):
        return None
    xf = bandpass(x, fs, lo, hi)
    f, q = dominant_frequency(xf, fs, lo, hi)
    if f is None:
        return None
    return {"value": round(f * 60.0, 1), "quality": round(q, 3), "reported": q >= min_quality}


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
        if t - self._last_emit < self.step_s or len(self._buf) < _SIG.estimator_min_samples:
            return None
        ts = np.array([b[0] for b in self._buf])
        if ts[-1] - ts[0] < self.window_s * _SIG.estimator_window_fill_frac:
            return None
        self._last_emit = t
        r = estimate_window(ts, [b[1] for b in self._buf], self.lo, self.hi, self.min_quality)
        if r is None:
            return None
        est = {"t": round(t, 2), "value": r["value"], "quality": r["quality"],
               "reported": r["reported"], **(extra or {})}
        self.estimates.append(est)
        return est


class HeartRateEstimator(_WindowEstimator):
    """rPPG heart rate from the pose-derived face patch.

    ``method="pos"`` (default): the patch's mean RGB goes through the W2 pipeline (``carelens.rppg``: POS with
    a CHROM cross-check, window/step/gates from ``params.rppg``), so it is robust to common-mode intensity
    changes. ``method="green"``: the original mean-green band-pass + Welch estimate (window/step from
    ``params.vitals.heart_rate``), kept as the baseline and as the trace-level path used by the ``hr_green_*``
    fixtures. Both gate on pose motion (``still_motion``) and on brightness [min_brightness, max_brightness]."""

    def __init__(self, window_s=None, step_s=None,
                 min_quality=_COMMON.min_quality, still_motion=_COMMON.still_motion_bodysizes_s,
                 min_brightness=_V.heart_rate.min_brightness,
                 max_brightness=_V.heart_rate.max_brightness, method: str = "pos"):
        if method not in ("pos", "green"):
            raise ValueError("method must be 'pos' or 'green'")
        self.method = method
        if window_s is None:
            window_s = PARAMS.rppg.window_s if method == "pos" else _V.heart_rate.window_s
        if step_s is None:
            step_s = PARAMS.rppg.step_s if method == "pos" else _V.heart_rate.step_s
        super().__init__(window_s, step_s, _V.heart_rate.band_lo_hz, _V.heart_rate.band_hi_hz,
                         min_quality, still_motion, True)
        self.min_b, self.max_b = min_brightness, max_brightness
        # the pipeline's face patch is in native pixels (not the web's <=160 px canvas): its own size gate
        self._face = FaceRPPG(PARAMS.rppg.model_copy(update={"min_roi_px": _V.pipeline_min_roi_px}))

    def _reset(self):
        super()._reset()
        self._face.reset()

    def update(self, t, image, state):
        if not state.present or state.motion > self.still_motion:
            self._reset(); return None
        roi = face_roi(state.landmarks, image.shape[0], image.shape[1])
        if roi is None:
            self._reset(); return None
        x0, y0, x1, y1 = roi
        patch = image[y0:y1, x0:x1]
        if self.method == "green":
            return self.update_roi(t, float(patch[:, :, 1].mean()),   # green channel (BGR index 1)
                                   float(patch.mean()), state.motion)
        return self.update_rgb(t, patch.reshape(-1, 3).mean(axis=0)[::-1].tolist(),   # BGR -> RGB
                               patch.shape[0] * patch.shape[1], float(patch.mean()), state.motion)

    def update_rgb(self, t, rgb, roi_px, brightness, motion):
        """ROI-level step for the POS method: pose gating, then ``FaceRPPG``. ``motion`` (body sizes/s) is passed
        on as face_motion rescaled so that the pose gate ``still_motion`` equals ``rppg.motion_max`` (it feeds the
        quality term). Appends an estimate whenever a window was analysed (with its ``span_s``)."""
        if motion > self.still_motion or not (self.min_b <= brightness <= self.max_b):
            self._reset(); return None
        fm = motion / self.still_motion * PARAMS.rppg.motion_max if self.still_motion > 0 else 0.0
        u = self._face.update(t, rgb, roi_px, fm, brightness)
        if u is None or u["bpm"] is None:
            return None
        est = {"t": round(t, 2), "value": round(u["bpm"], 1), "quality": round(u["quality"], 3),
               "reported": bool(u["reported"] and u["quality"] >= self.min_quality),
               "brightness": round(brightness, 1), "span_s": round(u["span_s"], 2)}
        self.estimates.append(est)
        return est

    def update_roi(self, t, green, brightness, motion):
        """Green-channel ROI-level step (after face ROI extraction): gating, then push + window estimate."""
        if motion > self.still_motion or not (self.min_b <= brightness <= self.max_b):
            self._reset(); return None
        self._push(t, green)
        return self._maybe_estimate(t, {"brightness": round(brightness, 1)})


class BreathingRateEstimator(_WindowEstimator):
    """Vertical chest motion (phase correlation between frames), band-passed
    0.1-0.5 Hz (6-30 breaths/min)."""

    def __init__(self, window_s=_V.breathing.window_s, step_s=_V.breathing.step_s,
                 min_quality=_COMMON.min_quality, still_motion=_COMMON.still_motion_bodysizes_s):
        super().__init__(window_s, step_s, _V.breathing.band_lo_hz, _V.breathing.band_hi_hz,
                         min_quality, still_motion, True)
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
        return self.update_displacement(t, self._pos)

    def update_displacement(self, t, pos):
        """Trace-level step: push cumulative chest displacement, window estimate."""
        self._push(t, pos)
        return self._maybe_estimate(t)
