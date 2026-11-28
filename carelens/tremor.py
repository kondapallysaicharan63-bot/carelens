"""Rhythmic hand movement (3-12 Hz) from MediaPipe HandLandmarker points (W8, Experimental).

Python reference; ``web/src/core/tremor.ts`` mirrors it operation for operation and both read
``shared/params.json -> tremor``. Not a medical device and not a test for any condition: it reports whether a clear
rhythmic movement of the held-out hand was seen, its frequency and its size as a percentage of hand size, as a wellbeing
trend to compare with your own earlier results. Many things change it (caffeine, tiredness, cold, holding a position).
Seeing no clear rhythm is the EXPECTED result for most people and is reported neutrally.

Input per frame: time ``t`` (s), 21 landmarks (x, y normalised by frame width / height) or ``None``, a hand score.

Chain
  1. Present frames: landmarks found and score >= ``min_hand_score``. The analysed run is the longest stretch of present
     frames without a gap above ``max_gap_s`` (a hand lost mid-test shortens the usable window), and at most the last
     ``test_s`` seconds of it.
  2. Signal: centroid of 11 points (wrist, 5 MCPs: thumb + 4 fingers, 5 fingertips) in aspect-corrected coordinates
     (x * w/h, y), divided by the hand size (wrist -> middle MCP distance, median over the window): a dimensionless
     displacement, so distance to the camera does not matter. Linear interpolation onto the shared grid (``fs_hz``
     points ending at the last sample), closed-form linear detrend per axis.
  3. Gross motion: the same signal low-passed below ``motion_band_hi_hz`` (raised-cosine, zero at ``taper_stop_lo_hz``);
     RMS of the 2-D vector in % of hand size above ``moving_max_pct`` -> ``moving``.
  4. Voluntary movement is removed by the shared FFT taper band-pass (stop < ``taper_stop_lo_hz``, pass
     ``band_lo_hz``..``band_hi_hz``, stop > ``taper_stop_hi_hz``), ``edge_trim_s`` is dropped at both ends (filter edge
     effects), then the 2-D signal is projected on its principal axis (closed-form 2x2 PCA) -> 1-D.
  5. Spectrum of the 1-D signal: Hann, zero-pad, radix-2 FFT; peak in the band with parabolic (log) refinement.
     ``prominence`` = power within +-``prom_halfwidth_hz`` of the peak / total in-band power. ``rms_pct`` = RMS of the
     1-D signal in % of hand size (``amp_pct``); ``p2p_pct`` = max - min.

Gating (first match wins): ``no_hand`` (no present frame), ``too_short`` (usable window < ``min_window_s``), ``low_fps``
(effective rate < ``min_rate_hz``: 12 Hz needs > 24 Hz, so a lower rate would alias; no value is ever reported then),
``moving``, ``no_rhythm`` (``prominence`` < ``min_prominence`` OR ``rms_pct`` < ``noise_floor_pct``), else ``ok``.
Only ``ok`` is ``reported``.

Noise floor: MediaPipe-like landmark jitter (0.3-0.5 % of frame height per point, independent) on a hand ~0.24 frame
heights long gives an in-band RMS of about 0.3-0.5 % of hand size after averaging 11 points; ``noise_floor_pct`` (1.0)
sits 2-3x above that. Measured on synthetic steady hands (see tests/test_tremor.py): no_rhythm in >= 95 % of runs.
Known limits: linear resampling onto the 30 Hz grid under-reads the amplitude of fast movements when frame times are not
aligned with the grid (up to ~10 % at 11 Hz with 2 ms timing jitter); a steady hand held very close to the camera
(landmarks jitter more) can approach the floor.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from .dsp import bandpass, band_bins, detrend_linear, fft_size, grid_times, parabolic_log, power_spectrum
from .params import PARAMS, Tremor

WRIST, MIDDLE_MCP = 0, 9
CENTROID_INDICES = [0, 2, 5, 9, 13, 17, 4, 8, 12, 16, 20]   # wrist, thumb MCP + 4 finger MCPs, 5 fingertips
REASONS = ("ok", "no_hand", "too_short", "low_fps", "moving", "no_rhythm")
_EPS = 1e-30


def _result(reason: str, **kw) -> dict:
    out = {"reported": reason == "ok", "reason": reason, "freq_hz": None, "amp_pct": None, "p2p_pct": None,
           "prominence": None, "rms_pct": None, "motion_pct": None, "duration_s": 0.0, "rate_hz": None, "n_frames": 0}
    out.update(kw)
    return out


def _median(v: list) -> float:
    s = sorted(v)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def analyse(ts: Sequence[float], frames: Sequence[Optional[np.ndarray]], scores: Sequence[float], aspect: float,
            tp: Tremor = PARAMS.tremor) -> dict:
    """One test's frames -> result dict (see module docstring). ``frames[i]`` is a (21, 2) array or None."""
    present = [i for i, f in enumerate(frames) if f is not None and scores[i] >= tp.min_hand_score]
    if not present:
        return _result("no_hand")
    runs, cur = [], [present[0]]
    for i in present[1:]:
        if ts[i] - ts[cur[-1]] > tp.max_gap_s:
            runs.append(cur)
            cur = []
        cur.append(i)
    runs.append(cur)
    run = max(runs, key=lambda r: ts[r[-1]] - ts[r[0]])   # longest; ties: the first
    t_end = float(ts[run[-1]])
    sel = [i for i in run if ts[i] >= t_end - tp.test_s - 1e-9]
    duration = t_end - float(ts[sel[0]])
    rate = (len(sel) - 1) / duration if duration > 0 else 0.0
    base = {"duration_s": duration, "rate_hz": rate, "n_frames": len(sel)}
    if duration < tp.min_window_s - 1e-9:
        return _result("too_short", **base)
    if rate < tp.min_rate_hz:
        return _result("low_fps", **base)

    pts = [np.asarray(frames[i], dtype=np.float64) * np.array([aspect, 1.0]) for i in sel]
    size = _median([math.hypot(*(p[MIDDLE_MCP] - p[WRIST])) for p in pts])
    if not size > 1e-6:
        return _result("no_hand", **base)
    cen = np.array([p[CENTROID_INDICES].mean(axis=0) for p in pts]) / size
    t_sel = np.asarray([ts[i] for i in sel], dtype=np.float64)
    fs = tp.fs_hz
    n = int(math.floor(duration * fs + 1e-6))
    grid = grid_times(t_end, n, fs)
    x = detrend_linear(np.interp(grid, t_sel, cen[:, 0]))
    y = detrend_linear(np.interp(grid, t_sel, cen[:, 1]))
    n_fft = fft_size(n, tp.zero_pad_factor)

    low = SimpleNamespace(taper_stop_lo_hz=0.0, band_lo_hz=0.0, band_hi_hz=tp.motion_band_hi_hz, taper_stop_hi_hz=tp.taper_stop_lo_hz)
    lx, ly = bandpass(x, n_fft, fs, low), bandpass(y, n_fft, fs, low)
    motion = 100.0 * math.sqrt(float(np.mean(lx * lx + ly * ly)))
    if motion > tp.moving_max_pct:
        return _result("moving", motion_pct=motion, **base)

    k = int(round(tp.edge_trim_s * fs))
    bx, by = bandpass(x, n_fft, fs, tp)[k:n - k], bandpass(y, n_fft, fs, tp)[k:n - k]
    bx, by = bx - np.mean(bx), by - np.mean(by)
    sxx, syy, sxy = float(np.mean(bx * bx)), float(np.mean(by * by)), float(np.mean(bx * by))
    th = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    u = math.cos(th) * bx + math.sin(th) * by
    rms = 100.0 * math.sqrt(float(np.mean(u * u)))
    p2p = 100.0 * float(np.max(u) - np.min(u))

    m = u.size
    n2 = fft_size(m, tp.zero_pad_factor)
    power = power_spectrum(u, n2)
    k_lo, k_hi = band_bins(n2, fs, tp.band_lo_hz, tp.band_hi_hz)
    k0 = k_lo + int(np.argmax(power[k_lo:k_hi + 1]))
    freq = (k0 + parabolic_log(power, k0)) * fs / n2
    kk = np.arange(k_lo, k_hi + 1)
    near = np.abs(kk * fs / n2 - freq) <= tp.prom_halfwidth_hz
    prom = float(np.sum(power[k_lo:k_hi + 1][near]) / (np.sum(power[k_lo:k_hi + 1]) + _EPS))

    common = dict(prominence=prom, rms_pct=rms, motion_pct=motion, **base)
    if prom < tp.min_prominence or rms < tp.noise_floor_pct:
        return _result("no_rhythm", **common)
    return _result("ok", freq_hz=freq, amp_pct=rms, p2p_pct=p2p, **common)


def run_tremor(inp: dict) -> dict:
    """Parity runner: input = {w, h, fps, [t], score[], landmarks[]} (landmarks: 21 x [x, y] or null per frame)."""
    n = len(inp["landmarks"])
    ts = inp["t"] if "t" in inp else [i / inp["fps"] for i in range(n)]
    frames = [None if f is None else np.asarray(f, dtype=np.float64) for f in inp["landmarks"]]
    return analyse(ts, frames, inp["score"], inp["w"] / inp["h"])


# ------------------------------------------------------------------ synthetic hands (tests, fixtures, noise-floor study)

# a flat hand, palm to the camera, fingers spread: positions in hand-size units from the wrist (x right, y down = up in the world)
HAND_TEMPLATE = np.array([
    (0.0, 0.0), (-0.45, -0.25), (-0.70, -0.55), (-0.95, -0.80), (-1.15, -1.00),      # wrist, thumb CMC, MCP, IP, tip
    (-0.32, -0.98), (-0.38, -1.40), (-0.40, -1.70), (-0.42, -1.95),                    # index
    (0.0, -1.00), (0.0, -1.50), (0.0, -1.85), (0.0, -2.15),                            # middle
    (0.30, -0.95), (0.34, -1.35), (0.36, -1.65), (0.38, -1.90),                        # ring
    (0.58, -0.85), (0.65, -1.15), (0.70, -1.35), (0.73, -1.55),                        # pinky
])


def synth_hand(duration: float = 15.0, fps: float = 30.0, hz: float = 0.0, amp_rms_pct: float = 0.0, jitter: float = 0.004,
               drift_pct: float = 1.5, seed: int = 0, w: int = 640, h: int = 480, hand_frac: float = 0.24,
               angle_deg: float = 30.0, lost: Sequence[tuple] = (), t_jitter_s: float = 0.0, arm_pct: float = 0.0,
               arm_hz: float = 0.5) -> dict:
    """A synthetic held-out hand as the parity input dict.

    ``hz``/``amp_rms_pct``: sinusoidal movement along ``angle_deg`` with that RMS in % of hand size. ``jitter``:
    per-point independent Gaussian landmark noise, fraction of frame height (MediaPipe-like: 0.003-0.005).
    ``drift_pct``: voluntary slow drift (three sub-1 Hz sinusoids in x and y), RMS in % of hand size.
    ``arm_pct``/``arm_hz``: a large arm movement (sinusoid, RMS in % of hand size). ``lost``: (t0, t1) spans without a hand.
    ``t_jitter_s``: uniform timing jitter of the frame times."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * fps))
    t = np.arange(n) / fps
    if t_jitter_s:
        t = np.maximum(0.0, t + rng.uniform(-t_jitter_s, t_jitter_s, n))
        t[0] = 0.0
        t = np.maximum.accumulate(t)
    aspect = w / h
    d = np.zeros((n, 2))
    if hz > 0 and amp_rms_pct > 0:
        a = math.radians(angle_deg)
        d += (math.sqrt(2.0) * amp_rms_pct / 100.0 * np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi)))[:, None] * np.array([math.cos(a), math.sin(a)])
    if drift_pct > 0:
        for axis in (0, 1):
            comp = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in (0.15, 0.4, 0.8))
            d[:, axis] += drift_pct / 100.0 * comp / math.sqrt(1.5)
    if arm_pct > 0:
        d[:, 0] += math.sqrt(2.0) * arm_pct / 100.0 * np.sin(2 * np.pi * arm_hz * t + rng.uniform(0, 2 * np.pi))
        d[:, 1] += 0.5 * math.sqrt(2.0) * arm_pct / 100.0 * np.sin(2 * np.pi * 0.7 * arm_hz * t + rng.uniform(0, 2 * np.pi))
    size = hand_frac   # wrist -> middle MCP, in frame heights
    centre = np.array([0.5 * aspect, 0.75])   # wrist position (frame-height units): hand held up in the middle of the view
    pos = centre[None, None, :] + size * (HAND_TEMPLATE[None, :, :] + d[:, None, :])
    pos = pos + rng.normal(0.0, jitter, pos.shape)
    lm = pos / np.array([aspect, 1.0])
    gone = np.zeros(n, dtype=bool)
    for a0, a1 in lost:
        gone |= (t >= a0) & (t < a1)
    score = np.where(gone, 0.0, 0.97)
    inp = {"w": w, "h": h, "fps": fps, "score": score.tolist(),
           "landmarks": [None if gone[i] else lm[i].tolist() for i in range(n)]}
    if t_jitter_s:
        inp["t"] = t.tolist()
    return inp
