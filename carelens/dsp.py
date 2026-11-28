"""Shared DSP building blocks for the PPG pipelines (contact PPG W1, face rPPG W2).

Pure NumPy, only operations that are trivially reproducible in TypeScript (``web/src/core/dsp.ts``
mirrors this file): linear interpolation, a closed-form least-squares line, Hann, a radix-2 FFT,
a raised-cosine FFT-domain band-pass, a harmonic-sum fundamental selector with parabolic
refinement, and a band SNR. Functions take any params object exposing the ``SpectralParams`` fields
(``contact_ppg`` and ``rppg`` in ``shared/params.json`` both do). Moving code here must never change
W1 outputs: the W1 fixtures are the guard. Not a medical device.
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np

_EPS = 1e-30


class SpectralParams(Protocol):
    band_lo_hz: float
    band_hi_hz: float
    taper_stop_lo_hz: float
    taper_stop_hi_hz: float
    snr_halfwidth_hz: float
    harm_halfwidth_hz: float
    harm_w2: float
    harm_w3: float


def grid_times(t_end: float, n: int, fs: float) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return t_end - (n - 1 - k) / fs


def detrend_relative(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Subtract the closed-form least-squares line (against the sample index), divide by the raw mean."""
    n = y.size
    kk = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    ybar = float(np.mean(y))
    slope = float(np.sum(kk * (y - ybar)) / np.sum(kk * kk))
    return (y - (ybar + slope * kk)) / ybar, ybar


def percentile(x: np.ndarray, q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    s = np.sort(x)
    h = (s.size - 1) * q / 100.0
    lo = math.floor(h)
    hi = min(lo + 1, s.size - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (h - lo))


def hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def fft_size(n: int, factor: int) -> int:
    need, m = factor * n, 1
    while m < need:
        m *= 2
    return m


def power_spectrum(rel: np.ndarray, n_fft: int) -> np.ndarray:
    """|rfft(hann * rel, n_fft)|^2 (bins 0..n_fft/2)."""
    return np.abs(np.fft.rfft(rel * hann(rel.size), n=n_fft)) ** 2


def band_bins(n_fft: int, fs: float, lo: float, hi: float) -> tuple[int, int]:
    """Inclusive bin index range of [lo, hi] Hz."""
    return math.ceil(lo * n_fft / fs - 1e-9), math.floor(hi * n_fft / fs + 1e-9)


def parabolic_log(power: np.ndarray, k: int) -> float:
    """Fractional bin offset in [-0.5, 0.5] of the peak at k, parabola through log power."""
    a, b, c = (math.log(power[k - 1] + _EPS), math.log(power[k] + _EPS), math.log(power[k + 1] + _EPS))
    den = a - 2.0 * b + c
    if den == 0.0:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (a - c) / den))


def parabolic_lin(a: float, b: float, c: float) -> float:
    """Fractional offset of a local maximum from three linear samples (clamped to +-0.5)."""
    den = a - 2.0 * b + c
    if den == 0.0:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (a - c) / den))


def _peak_bin(power: np.ndarray, n_fft: int, fs: float, f: float, hw: float, k_lo: int, k_hi: int) -> int:
    """Index of the strongest bin within +-hw Hz of f, clipped to [k_lo, k_hi]; -1 if the range is empty."""
    lo = max(k_lo, math.ceil((f - hw) * n_fft / fs - 1e-9))
    hi = min(k_hi, math.floor((f + hw) * n_fft / fs + 1e-9))
    return -1 if hi < lo else lo + int(np.argmax(power[lo:hi + 1]))


def _harmonic_score(power: np.ndarray, n_fft: int, fs: float, f: float, p: SpectralParams) -> float:
    """S(f) = P(f) + w2 P(2f) + w3 P(3f); P = peak power within +-harm_halfwidth; terms at/above Nyquist are 0."""
    top = n_fft // 2
    sc = 0.0
    for mult, w in ((1, 1.0), (2, p.harm_w2), (3, p.harm_w3)):
        if mult * f < fs / 2.0:
            k = _peak_bin(power, n_fft, fs, mult * f, p.harm_halfwidth_hz, 0, top)
            if k >= 0:
                sc += w * float(power[k])
    return sc


def spectral_peak(power: np.ndarray, n_fft: int, fs: float, p: SpectralParams) -> float:
    """Refined in-band fundamental (Hz). Candidates f*, f*/2, 2f* (in band) are scored by the harmonic sum
    S(f); the best wins, ties go to the higher f. Guards against half- and double-rate picks."""
    k_lo, k_hi = band_bins(n_fft, fs, p.band_lo_hz, p.band_hi_hz)
    k0 = k_lo + int(np.argmax(power[k_lo:k_hi + 1]))
    f0 = k0 * fs / n_fft
    best_k, best_s = k0, _harmonic_score(power, n_fft, fs, f0, p)
    for fc in (f0 / 2.0, f0 * 2.0):   # evaluation order is irrelevant: ties resolved by f below
        if not (p.band_lo_hz <= fc <= p.band_hi_hz):
            continue
        kc = _peak_bin(power, n_fft, fs, fc, p.harm_halfwidth_hz, k_lo, k_hi)
        if kc < 0:
            continue
        sc = _harmonic_score(power, n_fft, fs, kc * fs / n_fft, p)
        if sc > best_s or (sc == best_s and kc > best_k):
            best_k, best_s = kc, sc
    return (best_k + parabolic_log(power, best_k)) * fs / n_fft


def spectral_snr(power: np.ndarray, n_fft: int, fs: float, f_star: float, p: SpectralParams) -> float:
    k_lo, k_hi = band_bins(n_fft, fs, p.band_lo_hz, p.band_hi_hz)
    f = np.arange(k_lo, k_hi + 1) * fs / n_fft
    pw = power[k_lo:k_hi + 1]
    sig = np.abs(f - f_star) <= p.snr_halfwidth_hz
    if 2.0 * f_star < fs / 2.0:
        sig |= np.abs(f - 2.0 * f_star) <= p.snr_halfwidth_hz
    return float(np.sum(pw[sig]) / (np.sum(pw[~sig]) + _EPS))


def taper(f: np.ndarray, p: SpectralParams) -> np.ndarray:
    """Raised-cosine band-pass gain: 0 <= stop_lo, ramp to 1 at band_lo, 1 to band_hi, ramp to 0 at stop_hi."""
    g = np.zeros_like(f, dtype=np.float64)
    up = (f >= p.taper_stop_lo_hz) & (f < p.band_lo_hz)
    g[up] = 0.5 - 0.5 * np.cos(np.pi * (f[up] - p.taper_stop_lo_hz) / (p.band_lo_hz - p.taper_stop_lo_hz))
    g[(f >= p.band_lo_hz) & (f <= p.band_hi_hz)] = 1.0
    dn = (f > p.band_hi_hz) & (f < p.taper_stop_hi_hz)
    g[dn] = 0.5 + 0.5 * np.cos(np.pi * (f[dn] - p.band_hi_hz) / (p.taper_stop_hi_hz - p.band_hi_hz))
    return g


def bandpass(rel: np.ndarray, n_fft: int, fs: float, p: SpectralParams) -> np.ndarray:
    """FFT-domain band-pass of the zero-padded signal, truncated back to the window length."""
    spec = np.fft.rfft(rel, n=n_fft)
    spec *= taper(np.arange(spec.size) * fs / n_fft, p)
    return np.fft.irfft(spec, n=n_fft)[:rel.size]


def detrend_linear(y: np.ndarray) -> np.ndarray:
    """Subtract the closed-form least-squares line (against the sample index); no scaling."""
    n = y.size
    kk = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    ybar = float(np.mean(y))
    slope = float(np.sum(kk * (y - ybar)) / np.sum(kk * kk))
    return y - (ybar + slope * kk)
