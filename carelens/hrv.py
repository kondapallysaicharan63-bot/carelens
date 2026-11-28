"""Heart-rate variability (RMSSD, SDNN) from a fingertip contact segment (W6).

Python reference; ``web/src/core/hrv.ts`` mirrors it operation for operation and both read ``shared/params.json -> hrv``
(plus ``contact_ppg`` for the grid and the band-pass). Not a medical device: a wellbeing trend, not a diagnosis, and never a
stress or health score.

Unlike the heart rate, HRV needs the beat series of the whole measurement, not of a 10 s window:

1. ``ContactPPG`` buffers the raw samples of the current contact epoch after the settling period (so the segment duration counts from the end of settling; frozen at ``max_duration_s``) and counts its 1 s
   updates after settling and how many were *steady*: reported ``ok``, or every W1 gate passed except the spectral-SNR
   gate (HRV itself spreads the heart-rate peak and would otherwise under-rate high-HRV people; deliberate deviation).
2. Segment: resampled to ``fs`` Hz on the W1 grid (``n = floor(D*fs)`` points ending at the last sample), the channel
   with the higher whole-segment spectral SNR (W1 rule), linear detrend / relative AC, FFT band-pass (zero-padded to the
   next power of two), then W1 ``find_beats`` (independent of the spectral estimate) -> sub-sample beat times -> IBIs (ms).
3. Artifacts: IBIs outside [ibi_min, ibi_max] are dropped, then any IBI deviating more than ``max_dev_frac`` from the
   median of its +-``neighbours`` neighbours (range-valid ones, itself excluded). RMSSD uses successive differences where
   BOTH IBIs are kept (no bridging over a rejected beat); SDNN is the sample SD of the kept IBIs.
4. Withheld (``reported`` False, no numbers) with reason: ``too_short`` (< min_duration_s), ``low_fps`` (effective rate
   < min_rate_hz), ``unsteady`` (fewer than min_ok_frac of the segment's updates were ok), ``too_many_artifacts``
   (kept fraction < min_kept_frac, or too few beats once the segment is at max length). Fewer than min_kept_beats kept
   while the segment can still grow (low heart rate) is ``collecting``: not an artifact problem, keep holding still.

Output (full precision): ``rmssd_ms, sdnn_ms, mean_hr_bpm, n_beats, n_kept, kept_frac, duration_s, reported, reason``.
Known limit: timing error of a 30 fps camera inflates RMSSD (see tests/test_hrv.py bias table).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .dsp import bandpass, detrend_relative, fft_size, grid_times, power_spectrum, spectral_peak, spectral_snr
from .params import PARAMS, ContactPpg, Hrv
from .ppg import ContactPPG, find_beats, run_contact_ppg  # noqa: F401


COLLECT_MARGIN_S = 0.5   # a frozen segment is within a frame or two of max_duration_s


def _median(v: list) -> float:
    s = sorted(v)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def keep_mask(ibis: list, hp: Hrv) -> list:
    """True for IBIs that survive the range check and the local-median deviation check."""
    n, k = len(ibis), hp.neighbours
    rng = [hp.ibi_min_ms <= x <= hp.ibi_max_ms for x in ibis]
    keep = list(rng)
    for i in range(n):
        if not rng[i]:
            continue
        nb = [ibis[j] for j in range(max(0, i - k), min(n, i + k + 1)) if j != i and rng[j]]
        if not nb or abs(ibis[i] - _median(nb)) > hp.max_dev_frac * _median(nb):
            keep[i] = False
    return keep


def hrv_metrics(ibis: list, keep: list) -> tuple:
    """(rmssd, sdnn, mean_hr) over the kept IBIs; each None when it cannot be computed."""
    kept = [x for x, k in zip(ibis, keep) if k]
    d = [ibis[i + 1] - ibis[i] for i in range(len(ibis) - 1) if keep[i] and keep[i + 1]]
    rmssd = math.sqrt(sum(x * x for x in d) / len(d)) if d else None
    sdnn = None
    mean_hr = None
    if kept:
        mean = sum(kept) / len(kept)
        mean_hr = 60000.0 / mean
        if len(kept) >= 2:
            sdnn = math.sqrt(sum((x - mean) ** 2 for x in kept) / (len(kept) - 1))
    return rmssd, sdnn, mean_hr


def segment_ibis(ts, rs, gs, pp: ContactPpg, hp: Hrv = PARAMS.hrv) -> tuple:
    """Whole-segment beat series: ``(ibis_ms, n_beats)``."""
    ts, rs, gs = (np.asarray(a, dtype=np.float64) for a in (ts, rs, gs))
    n = int(math.floor((float(ts[-1]) - float(ts[0])) * pp.fs_hz + 1e-6))
    grid = grid_times(float(ts[-1]), n, pp.fs_hz)
    n_spec = fft_size(n, pp.zero_pad_factor)
    best = None
    for raw in (rs, gs):
        y = np.interp(grid, ts, raw)
        if not float(np.mean(y)) > 1e-6:
            continue
        rel, _ = detrend_relative(y)
        power = power_spectrum(rel, n_spec)
        snr = spectral_snr(power, n_spec, pp.fs_hz, spectral_peak(power, n_spec, pp.fs_hz, pp), pp)
        if best is None or snr > best[0]:   # R is tried first and wins ties
            best = (snr, rel)
    if best is None:
        return [], 0
    rel = best[1]
    bp = bandpass(rel, fft_size(n, 1), pp.fs_hz, pp)
    beats = find_beats(bp, pp.fs_hz, 60.0 / pp.beat_max_bpm, pp.min_prom * float(np.std(bp)))
    if beats:   # drop ghost peaks (ripple in a long gap): peak height below min_amp_frac x the median peak height
        hts = [float(bp[int(math.floor(b + 0.5))]) for b in beats]
        mh = _median(hts)
        beats = [b for b, v in zip(beats, hts) if v >= hp.min_amp_frac * mh]
    bt = [float(grid[0]) + b / pp.fs_hz for b in beats]
    return [(bt[i + 1] - bt[i]) * 1000.0 for i in range(len(bt) - 1)], len(bt)


def analyze_hrv(ts, rs, gs, n_upd: int, n_ok: int, pp: ContactPpg = PARAMS.contact_ppg, hp: Hrv = PARAMS.hrv) -> dict:
    out = {"rmssd_ms": None, "sdnn_ms": None, "mean_hr_bpm": None, "n_beats": 0, "n_kept": 0, "kept_frac": 0.0,
           "duration_s": 0.0, "reported": False, "reason": "too_short"}
    if len(ts) < 2:
        return out
    dur = float(ts[-1]) - float(ts[0])
    out["duration_s"] = dur
    if dur < hp.min_duration_s:
        return out
    if (len(ts) - 1) / dur < hp.min_rate_hz:
        out["reason"] = "low_fps"
        return out
    if n_upd <= 0 or n_ok < hp.min_ok_frac * n_upd:
        out["reason"] = "unsteady"
        return out
    ibis, n_beats = segment_ibis(ts, rs, gs, pp, hp)
    keep = keep_mask(ibis, hp)
    n_kept = sum(keep)
    out.update(n_beats=n_beats, n_kept=n_kept, kept_frac=(n_kept / len(ibis)) if ibis else 0.0)
    rmssd, sdnn, hr = hrv_metrics(ibis, keep)
    if out["kept_frac"] < hp.min_kept_frac or rmssd is None or sdnn is None:
        out["reason"] = "too_many_artifacts"
        return out
    if n_kept < hp.min_kept_beats:   # slow heart rate: more time is all it needs, until the segment is full
        out["reason"] = "collecting" if dur < hp.max_duration_s - COLLECT_MARGIN_S else "too_many_artifacts"
        return out
    out.update(rmssd_ms=rmssd, sdnn_ms=sdnn, mean_hr_bpm=hr, reported=True, reason="ok")
    return out


def hrv_result(cp: ContactPPG, hp: Optional[Hrv] = None) -> dict:
    """HRV of the contact segment buffered so far by ``cp``."""
    ts, rs, gs, n_upd, n_ok = cp.segment()
    return analyze_hrv(ts, rs, gs, n_upd, n_ok, cp.p, hp or cp.hrv_p or PARAMS.hrv)


def run_hrv(inp: dict) -> dict:
    """Fixture runner: replay ``{t, r, g, b, clip_frac}`` through ContactPPG, return the final HRV result."""
    cp = ContactPPG(hrv=PARAMS.hrv)
    for i, t in enumerate(inp["t"]):
        cp.update(t, inp["r"][i], inp["g"][i], inp["b"][i], inp["clip_frac"][i])
    return {"hrv": hrv_result(cp)}


# ------------------------------------------------------------------ synthetic RR series (tests + fixtures)

def truth_hrv(ibis_ms) -> tuple:
    """(rmssd, sdnn) of a known IBI series."""
    x = np.asarray(ibis_ms, dtype=np.float64)
    return float(np.sqrt(np.mean(np.diff(x) ** 2))), float(np.std(x, ddof=1))


def synth_rr(duration_s: float, rmssd_ms: float, *, mean_rr_ms: float = 850.0, seed: int = 0, resp_hz: float = 0.25,
             rsa_frac: float = 0.7, ar: float = 0.6) -> np.ndarray:
    """Beat times (s) whose true RMSSD is exactly ``rmssd_ms``: mean + k * (respiratory sinus arrhythmia sinusoid
    (``rsa_frac`` of the variance) + AR(1) noise), k solved so the RMSSD of the generated series matches."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * 1000.0 / mean_rr_ms) + 8
    k = np.arange(n)
    rsa = np.sin(2 * np.pi * resp_hz * k * mean_rr_ms / 1000.0 + rng.uniform(0, 2 * np.pi))
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = ar * e[i - 1] + rng.normal()
    e = (e - e.mean()) / e.std()
    dev = np.sqrt(rsa_frac) * rsa / rsa.std() + np.sqrt(1 - rsa_frac) * e
    dev = dev - dev.mean()
    scale = rmssd_ms / float(np.sqrt(np.mean(np.diff(dev) ** 2)))
    rr = mean_rr_ms + scale * dev
    return np.concatenate([[0.0], np.cumsum(rr / 1000.0)]) - rng.uniform(0.0, 0.5)


def with_ectopics(beat_times, idx, frac: float = 0.35) -> np.ndarray:
    """Premature beats: beat ``i`` in ``idx`` moves ``frac`` of its preceding interval earlier (short IBI, then the
    compensatory long one; the following beat keeps its time)."""
    bt = np.array(beat_times, dtype=np.float64)
    for i in idx:
        bt[i] -= frac * (bt[i] - bt[i - 1])
    return bt


def without_beats(beat_times, idx) -> np.ndarray:
    """Missed beats: the listed beats are not detected (the IBI across the gap doubles)."""
    return np.delete(np.asarray(beat_times, dtype=np.float64), list(idx))
