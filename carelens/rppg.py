"""Face remote PPG (rPPG): heart rate from the colour of facial skin, via POS with a CHROM cross-check.

Reference implementation of the web "Heart rate (face)" mode (W2). The TypeScript port
``web/src/core/rppg.ts`` mirrors this file operation for operation; both read
``shared/params.json -> rppg`` and replay ``shared/fixtures/rppg_*.json``. The spectral path is the
one W1 uses (``carelens/dsp.py``: linear resampling, Hann, zero-padded radix-2 FFT, 0.7-3.0 Hz band,
harmonic-sum fundamental selection, parabolic refinement, band SNR).

Input per frame (the "parity level"; image processing is NOT part of the parity contract):
``t`` (s); ``roi_rgb`` = mean [R, G, B] (0-255) over the skin ROI, or ``None`` when no face was found;
``roi_px`` = number of pixels in the ROI polygons (before any skin filter); ``skin_frac`` (optional, default 1) =
fraction of them kept by the skin filter and averaged; ``face_motion`` = speed of the mean ROI landmark position in eye spans per second
(eye span = distance between the outer eye corners, landmarks 33 and 263; so it is frame-rate and distance
independent); ``brightness`` = mean ROI luminance
(0-255). The ROI is the union of three polygons (forehead, left cheek, right cheek) whose corners are
FaceLandmarker (478-point) landmark indices in ``rppg.roi_landmarks`` (chosen to stay clear of eyes,
brows, lips and hair); the capture layer may drop non-skin pixels with a YCrCb range (``skin_*`` params).

Pipeline per update (every ``step_s``):

1. Frame state, in priority order. *no_face* (``roi_rgb`` is None); *too_small* (``roi_px`` < min_roi_px);
   *skin_filter* (``skin_frac`` < min_skin_frac: the skin filter rejected most of the ROI, e.g. strongly
   coloured light); *lighting* (brightness outside [min_brightness, max_brightness], or the relative change of brightness
   over the last ``brightness_lag_s`` exceeds ``max_brightness_rate`` per second; the state is held for
   ``lighting_hold_s`` after the last bad frame). Any of these clears the sample buffer and restarts the
   valid-face timer (``track_epoch`` counts restarts; a UI measurement must restart when it changes).
2. *settling* until ``settle_s`` of continuous valid face.
3. Window: the last ``min(window_s, span)`` of samples, linearly interpolated per channel onto a uniform
   ``fs_hz`` grid (same grid rule as W1). Rejected as *gaps* if the largest real-sample gap > max_gap_s or
   the real rate < min_rate_hz.
4. *motion* if more than ``motion_frac_max`` of the window's frames have face_motion > ``motion_max``
   (the window is not analysed, so bpm is null).
5. POS (Wang et al. 2017) on the resampled RGB: sub-windows of l = round(pos_window_s * fs) samples, stride
   1; per sub-window Cn = C / mean(C) per channel, S1 = G - B, S2 = G + B - 2R (the projection
   [[0, 1, -1], [-2, 1, 1]]), h = S1 + (std(S1) / std(S2)) S2, h -= mean(h), overlap-add into H. The sum
   is linearly detrended and goes through the W1 spectral path -> ``bpm`` (POS) and ``snr``.
   CHROM (de Haan & Jeanne 2013), independent: Rn, Gn, Bn = channel / window mean, Xs = 3Rn - 2Gn,
   Ys = 1.5Rn + Gn - 1.5Bn, each linearly detrended and band-passed with the W1 raised-cosine FFT band-pass,
   alpha = std(Xf) / std(Yf), S = Xf - alpha Yf -> same spectral path -> ``bpm_chrom``.
6. Gates, in order: no_face / too_small / lighting / settling / gaps / motion / low_signal (POS SNR <
   min_snr) / disagree (|bpm - bpm_chrom| > agree_bpm: POS is withheld when the independent estimate
   does not confirm it) / ok.

Quality in [0, 1] = q_w_snr q_snr + q_w_agree q_agree + q_w_motion q_motion; every term is monotone:
q_snr = clip((SNR_dB - gate_dB) / (q_snr_full_db - gate_dB)) (rises with SNR), q_agree =
clip(1 - |bpm - bpm_chrom| / agree_bpm) (rises as POS and CHROM agree), q_motion =
clip(1 - mean(face_motion) / motion_max) (falls with motion). It is 0 when a window was not analysed.

Limits (do not over-claim): rPPG accuracy depends on skin tone, lighting and motion; the synthetic
generator below is a documented approximation, not evidence of performance on people. Validate per
condition on consented recordings (PRD section 14). Wellbeing demonstration only. Not a medical device.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from .dsp import (bandpass, detrend_linear, detrend_relative, fft_size, grid_times, power_spectrum,
                  spectral_peak, spectral_snr)
from .params import PARAMS, Rppg

REASONS = ("ok", "no_face", "too_small", "skin_filter", "motion", "lighting", "settling", "low_signal", "disagree", "gaps")
_EPS = 1e-30


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ------------------------------------------------------------------ pulse extraction

def pos_signal(y: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    """POS pulse signal of an (n, 3) RGB trace (columns R, G, B). Overlap-added sliding sub-windows."""
    n = y.shape[0]
    big_l = int(math.floor(win_s * fs + 0.5))
    h_sum = np.zeros(n)
    for m in range(big_l - 1, n):
        seg = y[m - big_l + 1:m + 1]
        mu = np.mean(seg, axis=0)
        if not bool(np.all(mu > 1e-6)):
            continue   # a channel that is ~0 over the whole sub-window carries no information: skip it
        cn = seg / mu
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
        sd1, sd2 = float(np.std(s1)), float(np.std(s2))
        a = sd1 / sd2 if sd2 > 1e-12 else 0.0
        h = s1 + a * s2
        h_sum[m - big_l + 1:m + 1] += h - float(np.mean(h))
    return h_sum


def chrom_signal(y: np.ndarray, n_fft: int, fs: float, p: Rppg) -> np.ndarray:
    """CHROM pulse signal of an (n, 3) RGB trace: chrominance pair, band-passed, alpha-tuned."""
    rn, gn, bn = (y[:, c] / float(np.mean(y[:, c])) for c in range(3))
    xs = detrend_linear(3.0 * rn - 2.0 * gn)
    ys = detrend_linear(1.5 * rn + gn - 1.5 * bn)
    xf, yf = bandpass(xs, n_fft, fs, p), bandpass(ys, n_fft, fs, p)
    sdy = float(np.std(yf))
    a = float(np.std(xf)) / sdy if sdy > 1e-12 else 0.0
    return xf - a * yf


def _spectrum(sig: np.ndarray, n_fft: int, p: Rppg) -> tuple[float, float]:
    """(fundamental Hz, SNR) of a pulse signal via the W1 spectral path."""
    power = power_spectrum(sig, n_fft)
    f = spectral_peak(power, n_fft, p.fs_hz, p)
    return f, spectral_snr(power, n_fft, p.fs_hz, f, p)


def quality_score(snr: float, delta_bpm: float, mean_motion: float, p: Rppg) -> float:
    gate_db = 10.0 * math.log10(p.min_snr)
    q_snr = _clip01((10.0 * math.log10(max(snr, _EPS)) - gate_db) / (p.q_snr_full_db - gate_db))
    q_agree = _clip01(1.0 - abs(delta_bpm) / p.agree_bpm)
    q_motion = _clip01(1.0 - mean_motion / p.motion_max)
    return p.q_w_snr * q_snr + p.q_w_agree * q_agree + p.q_w_motion * q_motion


# ------------------------------------------------------------------ window analysis

def analyze_window(ts, rgb, motion, p: Rppg = PARAMS.rppg) -> dict:
    """Analyse buffered real samples (``ts`` ascending, last = window end; ``rgb`` (m, 3); ``motion`` (m,)).
    Returns ``{"gaps": True}`` if unusable, ``{"gaps": False, "motion": True, ...}`` if too much motion,
    else the estimate dict ``{gaps, motion, bpm, bpm_chrom, snr, delta, motion_mean, quality}``."""
    ts, rgb, motion = (np.asarray(a, dtype=np.float64) for a in (ts, rgb, motion))
    t_end = float(ts[-1])
    w = min(p.window_s, t_end - float(ts[0]))
    n = int(math.floor(w * p.fs_hz + 1e-6))
    big_l = int(math.floor(p.pos_window_s * p.fs_hz + 0.5))
    if n < max(16, big_l + 1):
        return {"gaps": True}
    idx = int(np.searchsorted(ts, t_end - w, side="left"))
    lo = max(idx - 1, 0)
    mot = motion[idx:]
    ts, rgb = ts[lo:], rgb[lo:]
    if ts.size < 2:
        return {"gaps": True}
    rate = (ts.size - 1) / (ts[-1] - ts[0])
    if float(np.max(np.diff(ts))) > p.max_gap_s or rate < p.min_rate_hz:
        return {"gaps": True}
    mot_mean = float(np.mean(mot))
    if float(np.mean(mot > p.motion_max)) > p.motion_frac_max:
        return {"gaps": False, "motion": True, "motion_mean": mot_mean, "span_s": w}
    grid = grid_times(t_end, n, p.fs_hz)
    y = np.empty((n, 3))
    for c in range(3):
        y[:, c] = np.interp(grid, ts, rgb[:, c])
        if not float(np.mean(y[:, c])) > 1e-6:
            return {"gaps": True}
    n_fft = fft_size(n, p.zero_pad_factor)
    f_pos, snr = _spectrum(detrend_linear(pos_signal(y, p.fs_hz, p.pos_window_s)), n_fft, p)
    f_chr, _ = _spectrum(chrom_signal(y, n_fft, p.fs_hz, p), n_fft, p)
    bpm, bpm_chrom = 60.0 * f_pos, 60.0 * f_chr
    delta = bpm - bpm_chrom
    return {"gaps": False, "motion": False, "bpm": bpm, "bpm_chrom": bpm_chrom, "snr": snr, "delta": delta,
            "motion_mean": mot_mean, "span_s": w, "quality": quality_score(snr, delta, mot_mean, p)}


def green_estimate(ts, rgb, p: Rppg = PARAMS.rppg) -> Optional[dict]:
    """Baseline for comparison only (not part of the parity contract): green channel alone through the same
    resampling and spectral path. Returns ``{bpm, snr}`` or None if the window is unusable."""
    ts, rgb = np.asarray(ts, dtype=np.float64), np.asarray(rgb, dtype=np.float64)
    t_end = float(ts[-1])
    w = min(p.window_s, t_end - float(ts[0]))
    n = int(math.floor(w * p.fs_hz + 1e-6))
    if n < 16:
        return None
    grid = grid_times(t_end, n, p.fs_hz)
    yg = np.interp(grid, ts, rgb[:, 1])
    if not float(np.mean(yg)) > 1e-6:
        return None
    rel, _ = detrend_relative(yg)
    f, snr = _spectrum(rel, fft_size(n, p.zero_pad_factor), p)
    return {"bpm": 60.0 * f, "snr": snr}


# ------------------------------------------------------------------ streaming

class FaceRPPG:
    """Feed frames with ``update(t, roi_rgb, roi_px, face_motion, brightness)``; returns an output dict once per
    ``step_s`` (else None): ``{t, bpm, bpm_chrom, quality, reported, reason, snr, track_epoch}`` (full
    precision; round in the UI only). ``bpm``/``bpm_chrom`` are filled whenever a window was analysed, but only
    ``reported`` outputs may be shown. ``track_epoch`` counts valid-face restarts."""

    def __init__(self, params: Rppg = PARAMS.rppg):
        self.p = params
        self.reset()

    def reset(self) -> None:
        self._t: deque = deque()
        self._rgb: deque = deque()
        self._mot: deque = deque()
        self._bright: deque = deque()
        self._valid_since: Optional[float] = None
        self._light_until = -math.inf
        self._epoch = 0
        self._state = "ok"
        self._next_out: Optional[float] = None
        self._last_t: Optional[float] = None

    def _clear(self, state: str) -> None:
        self._t.clear(); self._rgb.clear(); self._mot.clear()
        self._valid_since = None
        self._state = state

    def _brightness_bad(self, t: float, b: float) -> bool:
        p = self.p
        dq = self._bright
        dq.append((t, b))
        while len(dq) >= 2 and dq[1][0] <= t - p.brightness_lag_s:
            dq.popleft()
        bad = not (p.min_brightness <= b <= p.max_brightness)
        t0, b0 = dq[0]
        if t - t0 >= 0.5 * p.brightness_lag_s:
            if abs(b - b0) / max(b0, 1.0) / (t - t0) > p.max_brightness_rate:
                bad = True
        return bad

    def update(self, t: float, roi_rgb, roi_px: float, face_motion: float, brightness: float,
               skin_frac: float = 1.0) -> Optional[dict]:
        p = self.p
        if self._last_t is not None and not t > self._last_t:
            return None   # non-monotonic timestamp: ignore the frame
        self._last_t = t
        if self._next_out is None:
            self._next_out = t + p.step_s
        if roi_rgb is None:
            self._bright.clear()
            self._clear("no_face")
        elif roi_px < p.min_roi_px:
            self._bright.clear()
            self._clear("too_small")
        elif skin_frac < p.min_skin_frac:
            self._bright.clear()
            self._clear("skin_filter")
        else:
            if self._brightness_bad(t, brightness):
                self._light_until = t + p.lighting_hold_s
            if t < self._light_until:
                self._clear("lighting")
            else:
                self._state = "ok"
                if self._valid_since is None:
                    self._valid_since = t
                    self._epoch += 1
                self._t.append(t); self._rgb.append(tuple(roi_rgb)); self._mot.append(face_motion)
                while len(self._t) >= 2 and self._t[1] < t - p.window_s:
                    self._t.popleft(); self._rgb.popleft(); self._mot.popleft()
        if t < self._next_out:
            return None
        while self._next_out <= t:
            self._next_out += p.step_s
        return self._emit(t)

    def valid_seconds(self, t: float) -> float:
        """Seconds of continuous valid face so far (0 if none)."""
        return 0.0 if self._valid_since is None else t - self._valid_since

    def _emit(self, t: float) -> dict:
        p = self.p
        out = {"t": t, "bpm": None, "bpm_chrom": None, "quality": 0.0, "reported": False,
               "reason": "ok", "snr": None, "span_s": None, "track_epoch": self._epoch}
        if self._state != "ok":
            out["reason"] = self._state
            return out
        if self._valid_since is None or t - self._valid_since < p.settle_s:
            out["reason"] = "settling"
            return out
        a = analyze_window(self._t, self._rgb, self._mot, p)
        if a["gaps"]:
            out["reason"] = "gaps"
        elif a["motion"]:
            out["reason"] = "motion"
            out["span_s"] = a["span_s"]
        elif not all(math.isfinite(a[k]) for k in ("bpm", "bpm_chrom", "snr", "quality")):
            out["reason"] = "low_signal"   # degenerate window (e.g. a dead channel): never report, never emit NaN
            out["span_s"] = a["span_s"]
        else:
            out.update(bpm=a["bpm"], bpm_chrom=a["bpm_chrom"], quality=a["quality"], snr=a["snr"], span_s=a["span_s"])
            if a["snr"] < p.min_snr:
                out["reason"] = "low_signal"
            elif abs(a["delta"]) > p.agree_bpm:
                out["reason"] = "disagree"
            else:
                out["reason"] = "ok"
                out["reported"] = True
        return out


def run_face_rppg(inp: dict) -> dict:
    """Fixture runner: replay ``{t, roi_rgb, roi_px, face_motion, brightness}`` arrays."""
    fr = FaceRPPG()
    ups = []
    for i, t in enumerate(inp["t"]):
        sf = inp.get("skin_frac")
        u = fr.update(t, inp["roi_rgb"][i], inp["roi_px"][i], inp["face_motion"][i], inp["brightness"][i],
                      1.0 if sf is None else sf[i])
        if u is not None:
            ups.append(u)
    return {"updates": ups}


# ------------------------------------------------------------------ synthetic traces (tests + fixtures)
#
# A documented SYNTHETIC approximation of a skin-colour trace; it is not evidence about real people.
#  * Pulse: relative modulation of each channel proportional to the POS-paper skin-model signature
#    [0.33, 0.77, 0.53] (R, G, B), negative-going with blood volume (blood absorbs).
#  * Skin tone: three DC colours and green peak-to-peak pulse strengths (fraction of DC). Darker skin gets
#    a darker DC and a weaker pulse (melanin absorbs light that would reach the blood); the sensor noise is
#    an absolute level, so the darker profiles also have a lower SNR. The numbers are round illustrative
#    values, not measured populations.
#  * Motion/specular noise: a band-limited random intensity modulation that overlaps the pulse band,
#    common-mode across channels (multiplicative), plus a small white additive specular part. POS is
#    designed to reject the common-mode part.
#  * Lighting: slow multiplicative drift and optional steps; sensor noise: white.

SKIN_TONES = {
    "light": {"dc": (224.0, 172.0, 150.0), "pulse": 0.0070},
    "medium": {"dc": (180.0, 125.0, 95.0), "pulse": 0.0050},
    "dark": {"dc": (105.0, 72.0, 56.0), "pulse": 0.0030},
}
PULSE_SIGNATURE = (0.33 / 0.77, 1.0, 0.53 / 0.77)   # relative modulation per channel, green = 1


def _pulse_wave(tau: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((tau - 0.18) / 0.09) ** 2) + 0.4 * np.exp(-0.5 * ((tau - 0.45) / 0.12) ** 2)


def _band_noise(rng, t: np.ndarray, lo: float = 0.6, hi: float = 3.2, k: int = 6) -> np.ndarray:
    """Unit-variance random signal made of k sinusoids in [lo, hi] Hz with slowly varying amplitudes."""
    out = np.zeros_like(t)
    for _ in range(k):
        f, ph, ph2 = rng.uniform(lo, hi), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        out += (1.0 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + ph2)) * np.sin(2 * np.pi * f * t + ph)
    return out / math.sqrt(k * 0.5 * (1.0 + 0.125))


def synth_face_trace(duration: float = 40.0, bpm: float = 72.0, *, skin: str = "medium", fs: float = 30.0,
                     seed: int = 0, noise: float = 0.06, motion_noise: float = 0.0, specular: float = 0.1,
                     drift: float = 0.03, roi_px: int = 2400, events: tuple = (), pulse_vec=None,
                     interference=None) -> dict:
    """Synthetic face-ROI trace at the parity level. ``noise`` = white sensor noise sd (0-255 levels);
    ``motion_noise`` = sd of the common-mode multiplicative intensity noise (fraction); ``specular`` = the
    additive white part as a fraction of ``motion_noise`` x mean DC; ``drift`` = amplitude of the slow lighting
    drift (fraction). ``events``: ``(t0, t1, kind[, arg])`` with kind in ``motion`` (face_motion ~ 1.0 and
    ``arg`` = extra intensity-noise sd, default 0.05), ``lightstep`` (multiply the light by ``arg``),
    ``noface`` (roi_rgb None), ``small`` (roi_px = 100), ``skinless`` (skin_frac = 0.05). ``interference`` = ``(bpm, amp, (r, g, b))`` adds a
    second periodic colour modulation with its own channel signature (a coloured flicker/artefact). Returns ``t, roi_rgb, roi_px, face_motion,
    brightness`` lists plus ``truth_bpm``, ``skin``."""
    rng = np.random.default_rng(seed)
    prof = SKIN_TONES[skin]
    n = int(duration * fs)
    t = np.arange(n) / fs
    period = 60.0 / bpm
    wave = np.zeros(n)
    for b0 in np.arange(rng.uniform(0, period) - period, duration + period, period):
        m = (t >= b0) & (t < b0 + 1.2)
        wave[m] += _pulse_wave(t[m] - b0)
    wave = wave / 1.4
    wave -= wave.mean()
    vec = np.array(PULSE_SIGNATURE if pulse_vec is None else pulse_vec)
    dc = np.array(prof["dc"])
    rgb = dc[None, :] * (1.0 - prof["pulse"] * vec[None, :] * wave[:, None])
    if interference is not None:
        ibpm, iamp, ivec = interference
        rgb = rgb * (1.0 + iamp * np.array(ivec)[None, :] * np.sin(2 * np.pi * ibpm / 60.0 * t + 0.7)[:, None])
    light = 1.0 + drift * np.sin(2 * np.pi * 0.04 * t + rng.uniform(0, 6.28))
    mn = np.full(n, motion_noise)
    motion = np.abs(rng.normal(0, 0.05, n))
    px = np.full(n, roi_px, dtype=np.int64)
    skin_frac = np.ones(n)
    noface = np.zeros(n, bool)
    for ev in events:
        t0, t1, kind = ev[0], ev[1], ev[2]
        arg = ev[3] if len(ev) > 3 else None
        m = (t >= t0) & (t < t1)
        if kind == "motion":
            motion[m] = 1.0 + np.abs(rng.normal(0, 0.2, m.sum()))
            mn[m] = max(motion_noise, 0.05 if arg is None else arg)
        elif kind == "lightstep":
            light[m] *= arg
        elif kind == "noface":
            noface |= m
        elif kind == "small":
            px[m] = 100
        elif kind == "skinless":
            skin_frac[m] = 0.05
    nm = _band_noise(rng, t)
    rgb = rgb * (1.0 + mn * nm)[:, None] + (specular * mn * nm * dc.mean())[:, None]
    rgb = rgb * light[:, None] + rng.normal(0, noise, (n, 3))
    rgb = np.clip(rgb, 0.0, 255.0)
    bright = rgb.mean(axis=1) + rng.normal(0, 0.1, n)
    roi = [None if noface[i] else rgb[i].tolist() for i in range(n)]
    return {"t": t.tolist(), "roi_rgb": roi, "roi_px": px.tolist(),
            "face_motion": np.where(noface, 0.0, motion).tolist(), "brightness": bright.tolist(),
            "skin_frac": skin_frac.tolist(),
            "truth_bpm": bpm, "skin": skin}
