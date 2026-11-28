"""Contact PPG: heart rate from a fingertip over a phone's back camera (+ torch).

Reference implementation of the web "Heart rate (fingertip)" mode (W1). The TypeScript port
``web/src/core/ppg.ts`` mirrors this file operation for operation; both read
``shared/params.json -> contact_ppg`` and replay ``shared/fixtures/ppg_*.json``. Only operations
that are trivially reproducible in both languages are used (linear interpolation, a closed-form
least-squares line, Hann, a radix-2 FFT, a raised-cosine FFT-domain band-pass, hand-written peak
rules): no IIR design, no filtfilt, no scipy.

Input per frame: timestamp ``t`` (s) and the mean R, G, B of the central ROI of a small frame,
plus ``clip_frac`` (fraction of ROI pixels with R >= 250).

Pipeline per update (every ``step_s``):

1. Frame state. A frame is *no_finger* if R < finger_r_min, R/(G+1) < finger_ratio_min or
   |dR| to the previous frame > finger_dr_max (motion / pressure jolt); *saturated* if
   clip_frac > max_clip_frac. Either clears the sample buffer and restarts the contact timer.
2. *settling* until contact has lasted ``settle_s``.
3. Window: the last ``min(window_s, contact span)`` of samples, linearly interpolated onto a
   uniform ``fs`` grid. Grid: ``n = floor(W*fs)`` points, ``t_end - (n-1-k)/fs`` for k = 0..n-1
   (so the last point is exactly ``t_end``; endpoint included, start at ``t_end-(n-1)/fs``). The
   last real sample before the window start is kept as the left interpolation bracket. Window is
   rejected ("gaps") if the largest real-sample gap > max_gap_s or the real rate < min_rate_hz.
4. For R and for G: subtract the least-squares line, divide by the raw window mean (relative AC),
   perfusion index PI = (p95 - p5) of the relative AC x 100 (percent). Hann window, zero-pad to
   N = next power of two >= zero_pad_factor*length, power spectrum. Fundamental: the in-band peak f* and
   its half and double (if in band) are scored S(f) = P(f) + harm_w2 P(2f) + harm_w3 P(3f) (P = peak power
   within +-harm_halfwidth_hz); the best score wins, ties go to the higher f; parabolic interpolation on log
   power -> f_spec. SNR = power within +-snr_halfwidth of f_spec and of 2*f_spec (below Nyquist, in band) over
   the remaining in-band power. The channel with the higher SNR is used (R wins ties).
5. Beats (independent of f_spec): the relative AC (zero-padded to N) is band-passed in the FFT domain with a
   raised-cosine taper (0 below taper_stop_lo, ramp to 1 at band_lo, 1 to band_hi, ramp to 0 at taper_stop_hi);
   local maxima with prominence >= min_prom x std, spacing >= 60/beat_max_bpm, then within any 0.5 x median
   preliminary interval only the most prominent peak is kept; parabolic sub-sample timing -> IBIs (ms);
   f_ibi = 60000/median(IBI). A genuine cross-check: spectrum and beats must agree within agree_bpm.
6. Gates, in order: no_finger / saturated / settling / gaps / low_signal (SNR < min_snr, PI outside
   [pi_min, pi_max], or fewer than min_ibis IBIs) / disagree (|60 f_spec - f_ibi| > agree_bpm) / ok.

Quality in [0, 1] = q_w_snr*q_snr + q_w_agree*q_agree + q_w_pi*q_pi (weights sum to 1; each term is
monotone), with q_snr = clip((SNR_dB - 10log10(min_snr)) / (q_snr_full_db - 10log10(min_snr))),
q_agree = clip(1 - |60 f_spec - f_ibi| / agree_bpm) (0 without an IBI estimate), and q_pi a
trapezoid: 0 at pi_min, 1 on [q_pi_full_lo, q_pi_full_hi], 0 at pi_max. Quality is 0 when the window
was not analysed.

Wellbeing demonstration only. Not a medical device.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from .dsp import (_EPS, band_bins, bandpass, detrend_relative, fft_size, grid_times, hann,  # noqa: F401 (re-exported)
                  parabolic_lin, parabolic_log, percentile, power_spectrum, spectral_peak, spectral_snr, taper)
from .params import PARAMS, ContactPpg, Hrv

REASONS = ("ok", "no_finger", "saturated", "settling", "low_signal", "disagree", "gaps")


# ------------------------------------------------------------------ pure building blocks

def perfusion_index(rel: np.ndarray, p: ContactPpg) -> float:
    return (percentile(rel, p.pi_hi_pct) - percentile(rel, p.pi_lo_pct)) * 100.0


def _prominence(x: np.ndarray, i: int) -> float:
    n, v = x.size, x[i]
    lmin = v
    j = i - 1
    while j >= 0 and x[j] < v:
        lmin = min(lmin, x[j])
        j -= 1
    rmin = v
    j = i + 1
    while j < n and x[j] < v:
        rmin = min(rmin, x[j])
        j += 1
    return float(v - max(lmin, rmin))


def find_beats(x: np.ndarray, fs: float, spacing_s: float, min_prom_abs: float) -> list[float]:
    """Sub-sample beat positions (in samples), independent of the spectral estimate.

    Strict-left local maxima with prominence >= min_prom_abs; tallest-first greedy minimum spacing
    ``spacing_s`` (ties by earlier index); then secondary-peak rejection: with m = median of the
    preliminary peak-to-peak intervals, keep (most prominent first) only peaks with no kept peak
    closer than 0.5 m; parabolic refinement."""
    prom = {i: _prominence(x, i) for i in range(1, x.size - 1) if x[i - 1] < x[i] >= x[i + 1]}
    cand = [i for i, pr in prom.items() if pr >= min_prom_abs]
    min_d = spacing_s * fs
    kept: list[int] = []
    for i in sorted(cand, key=lambda i: (-x[i], i)):
        if all(abs(i - j) >= min_d for j in kept):
            kept.append(i)
    kept.sort()
    if len(kept) >= 3:
        d = sorted(float(kept[k + 1] - kept[k]) for k in range(len(kept) - 1))
        m = len(d) // 2
        med = d[m] if len(d) % 2 else 0.5 * (d[m - 1] + d[m])
        keep2: list[int] = []
        for i in sorted(kept, key=lambda i: (-prom[i], i)):
            if all(abs(i - j) >= 0.5 * med for j in keep2):
                keep2.append(i)
        kept = sorted(keep2)
    return [i + parabolic_lin(float(x[i - 1]), float(x[i]), float(x[i + 1])) for i in kept]


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def quality_score(snr: float, delta_bpm: Optional[float], pi: float, p: ContactPpg) -> float:
    gate_db = 10.0 * math.log10(p.min_snr)
    q_snr = _clip01((10.0 * math.log10(max(snr, _EPS)) - gate_db) / (p.q_snr_full_db - gate_db))
    q_agree = 0.0 if delta_bpm is None else _clip01(1.0 - abs(delta_bpm) / p.agree_bpm)
    q_pi = min(_clip01((pi - p.pi_min) / (p.q_pi_full_lo - p.pi_min)),
               _clip01((p.pi_max - pi) / (p.pi_max - p.q_pi_full_hi)))
    return p.q_w_snr * q_snr + p.q_w_agree * q_agree + p.q_w_pi * q_pi


# ------------------------------------------------------------------ window analysis

def analyze_window(ts, rs, gs, p: ContactPpg = PARAMS.contact_ppg) -> dict:
    """Analyse buffered real samples (``ts`` ascending, last = window end). Returns ``{"gaps": True}``
    if the window is unusable, else the estimate dict (see keys below)."""
    ts, rs, gs = (np.asarray(a, dtype=np.float64) for a in (ts, rs, gs))
    t_end = float(ts[-1])
    w = min(p.window_s, t_end - float(ts[0]))
    n = int(math.floor(w * p.fs_hz + 1e-6))
    if n < 16:
        return {"gaps": True}
    idx = int(np.searchsorted(ts, t_end - w, side="left"))
    lo = max(idx - 1, 0)
    ts, rs, gs = ts[lo:], rs[lo:], gs[lo:]
    if ts.size < 2:
        return {"gaps": True}
    rate = (ts.size - 1) / (ts[-1] - ts[0])
    if float(np.max(np.diff(ts))) > p.max_gap_s or rate < p.min_rate_hz:
        return {"gaps": True}
    grid = grid_times(t_end, n, p.fs_hz)
    n_fft = fft_size(n, p.zero_pad_factor)
    best = None
    for name, raw in (("R", rs), ("G", gs)):
        y = np.interp(grid, ts, raw)
        if not float(np.mean(y)) > 1e-6:
            continue
        rel, _ = detrend_relative(y)
        power = power_spectrum(rel, n_fft)
        f = spectral_peak(power, n_fft, p.fs_hz, p)
        snr = spectral_snr(power, n_fft, p.fs_hz, f, p)
        if best is None or snr > best["snr"]:   # R is tried first and wins ties
            best = {"channel": name, "rel": rel, "f_spec": f, "snr": snr}
    if best is None:
        return {"gaps": True}
    rel, f_spec = best["rel"], best["f_spec"]
    pi = perfusion_index(rel, p)
    bp = bandpass(rel, n_fft, p.fs_hz, p)
    std = float(np.std(bp))
    beats = find_beats(bp, p.fs_hz, 60.0 / p.beat_max_bpm, p.min_prom * std)
    bt = [float(grid[0]) + b / p.fs_hz for b in beats]
    ibis = [(bt[i + 1] - bt[i]) * 1000.0 for i in range(len(bt) - 1)]
    f_ibi = None
    if len(ibis) >= p.min_ibis:
        s = sorted(ibis)
        m = len(s) // 2
        med = s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])
        f_ibi = 60000.0 / med
    bpm = 60.0 * f_spec
    delta = None if f_ibi is None else bpm - f_ibi
    return {"gaps": False, "span_s": w, "channel": best["channel"], "bpm": bpm, "bpm_ibi": f_ibi, "snr": best["snr"],
            "pi": pi, "delta": delta, "ibis_ms": ibis, "quality": quality_score(best["snr"], delta, pi, p)}


# ------------------------------------------------------------------ streaming

class ContactPPG:
    """Feed frames with ``update(t, r, g, b, clip_frac)``; returns an output dict once per ``step_s``
    (else None): ``{t, bpm, bpm_ibi, quality, reported, reason, snr, pi, channel, contact_epoch, span_s}`` (``span_s`` = analysed window length, None when no window was analysed) (full precision; round in the UI only). ``contact_epoch`` counts contact
    restarts (a UI measurement must restart when it changes). ``bpm`` /
    ``bpm_ibi`` are filled whenever a window was analysed, but only ``reported`` outputs may be shown."""

    def __init__(self, params: ContactPpg = PARAMS.contact_ppg, hrv: Optional[Hrv] = None):
        self.p = params
        self.hrv_p = hrv   # None: do not record the contact segment (W6 HRV off)
        self.last_ibis_ms: list[float] = []   # IBIs of the most recent analysed window (W6 HRV)
        self.reset()

    def reset(self) -> None:
        self._t: deque = deque()
        self._r: deque = deque()
        self._g: deque = deque()
        self._prev_r: Optional[float] = None
        self._contact_since: Optional[float] = None
        self._epoch = 0
        self._state = "ok"
        self._next_out: Optional[float] = None
        self._last_t: Optional[float] = None
        self.last_ibis_ms = []
        self._reset_segment()

    def _reset_segment(self) -> None:
        # raw samples of the current contact epoch AFTER settling (W6 HRV), frozen at hrv.max_duration_s
        self._seg_t: list = []
        self._seg_r: list = []
        self._seg_g: list = []
        self._seg_upd = 0   # 1 s updates since settling ended (current epoch)
        self._seg_ok = 0    # ... of which reported "ok"

    def segment(self) -> tuple:
        """``(t, r, g, n_updates, n_ok)`` of the current contact epoch (W6 HRV input)."""
        return self._seg_t, self._seg_r, self._seg_g, self._seg_upd, self._seg_ok

    def _clear(self, t: float, r: float, state: str) -> None:
        self._t.clear(); self._r.clear(); self._g.clear()
        self._reset_segment()
        self._contact_since = None
        self._prev_r = r
        self._state = state

    def update(self, t: float, r: float, g: float, b: float = 0.0, clip_frac: float = 0.0) -> Optional[dict]:
        p = self.p
        if self._last_t is not None and not t > self._last_t:
            return None   # non-monotonic timestamp: ignore the frame
        self._last_t = t
        if self._next_out is None:
            self._next_out = t + p.step_s
        jolt = self._prev_r is not None and abs(r - self._prev_r) > p.finger_dr_max
        if r < p.finger_r_min or r / (g + 1.0) < p.finger_ratio_min or jolt:
            self._clear(t, r, "no_finger")
        elif clip_frac > p.max_clip_frac:
            self._clear(t, r, "saturated")
        else:
            self._state = "ok"
            self._prev_r = r
            if self._contact_since is None:
                self._contact_since = t
                self._epoch += 1
            self._t.append(t); self._r.append(r); self._g.append(g)
            if self.hrv_p is not None and t - self._contact_since >= p.settle_s \
                    and (not self._seg_t or t - self._seg_t[0] <= self.hrv_p.max_duration_s):
                self._seg_t.append(t); self._seg_r.append(r); self._seg_g.append(g)
            while len(self._t) >= 2 and self._t[1] < t - p.window_s:
                self._t.popleft(); self._r.popleft(); self._g.popleft()
        if t < self._next_out:
            return None
        while self._next_out <= t:
            self._next_out += p.step_s
        return self._emit(t)

    def _emit(self, t: float) -> dict:
        out = {"t": t, "bpm": None, "bpm_ibi": None, "quality": 0.0, "reported": False,
               "reason": "ok", "snr": None, "pi": None, "channel": None,
               "contact_epoch": self._epoch, "span_s": None}
        if self._state != "ok":
            out["reason"] = self._state
            return out
        if self._contact_since is None or t - self._contact_since < self.p.settle_s:
            out["reason"] = "settling"
            return out
        a = analyze_window(self._t, self._r, self._g, self.p)
        counting = self.hrv_p is not None and bool(self._seg_t) and t - self._seg_t[0] <= self.hrv_p.max_duration_s
        if counting:
            self._seg_upd += 1
        if a["gaps"]:
            out["reason"] = "gaps"
            return out
        p = self.p
        self.last_ibis_ms = a["ibis_ms"]
        out.update(bpm=a["bpm"], bpm_ibi=a["bpm_ibi"],
                   quality=a["quality"], snr=a["snr"], pi=a["pi"], channel=a["channel"], span_s=a["span_s"])
        pi_ok = p.pi_min <= a["pi"] <= p.pi_max
        if counting and a["snr"] < p.min_snr and pi_ok and a["delta"] is not None and abs(a["delta"]) <= p.agree_bpm:
            self._seg_ok += 1   # steady for HRV: every W1 gate passed except SNR (HRV itself broadens the spectral peak)
        if a["snr"] < p.min_snr or not pi_ok or a["delta"] is None:
            out["reason"] = "low_signal"
        elif abs(a["delta"]) > p.agree_bpm:
            out["reason"] = "disagree"
        else:
            out["reason"] = "ok"
            out["reported"] = True
            if counting:
                self._seg_ok += 1
        return out


def run_contact_ppg(inp: dict) -> dict:
    """Fixture runner: replay ``{t, r, g, b, clip_frac}`` arrays; returns the updates and the IBIs
    of the last analysed window."""
    cp = ContactPPG()
    ups = []
    for i, t in enumerate(inp["t"]):
        u = cp.update(t, inp["r"][i], inp["g"][i], inp["b"][i], inp["clip_frac"][i])
        if u is not None:
            ups.append(u)
    return {"updates": ups, "final_ibis_ms": list(cp.last_ibis_ms)}


# ------------------------------------------------------------------ synthetic traces (tests + fixtures)

def _pulse(tau: np.ndarray) -> np.ndarray:
    """One beat's waveform vs time since the beat (s): systolic peak plus a smaller dicrotic bump."""
    return np.exp(-0.5 * ((tau - 0.18) / 0.09) ** 2) + 0.4 * np.exp(-0.5 * ((tau - 0.45) / 0.12) ** 2)


def synth_trace(duration: float = 40.0, bpm: float = 72.0, *, beat_times=None, fs: float = 30.0,
                seed: int = 0, base_r: float = 200.0, base_g: float = 60.0, amp: float = 3.0,
                noise: float = 0.2, resp_mod: float = 0.0, wander: float = 0.0, jitter_s: float = 0.0,
                drop_frac: float = 0.0, hiccup: Optional[tuple] = None, harmonic: float = 0.0,
                events: tuple = ()) -> dict:
    """Synthetic fingertip trace. ``amp`` = approximate peak-to-peak pulse in R units; ``noise`` = white
    noise sd (R units); ``resp_mod`` = fractional respiratory amplitude modulation at 0.25 Hz;
    ``wander`` = slow baseline amplitude (R units, 0.1 Hz + drift); ``harmonic`` adds that fraction of a
    second harmonic (strong values make the 2nd harmonic dominate); ``events`` are
    ``(t0, t1, kind)`` with kind in burst | saturate | lift | lowperf. Returns ``t, r, g, b, clip_frac``
    lists, ``truth_bpm`` and the true ``beat_times``."""
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    if jitter_s:
        t = t + rng.uniform(-jitter_s, jitter_s, n)
        t[0] = max(t[0], 0.0)
        t = np.maximum.accumulate(t + np.arange(n) * 1e-6)
    if beat_times is None:
        period = 60.0 / bpm
        beat_times = np.arange(rng.uniform(0, period) - period, duration + period, period)
    beat_times = np.asarray(beat_times, dtype=np.float64)
    wave = np.zeros_like(t)
    for b0 in beat_times:
        m = (t >= b0) & (t < b0 + 1.2)
        wave[m] += _pulse(t[m] - b0)
    if harmonic:
        period = 60.0 / bpm
        wave += harmonic * np.sin(2 * np.pi * 2 * (t / period) + 1.0)
    wave = wave / 1.4                                   # template peak ~ 1.4 -> unit-ish p-p
    mod = 1.0 + resp_mod * np.sin(2 * np.pi * 0.25 * t)
    sig = amp * mod * wave
    base = (wander * np.sin(2 * np.pi * 0.1 * t + 0.7) + 0.02 * wander * (t - duration / 2)
            + 0.5 * resp_mod * amp * np.sin(2 * np.pi * 0.25 * t))
    r = base_r + sig + base + rng.normal(0, noise, n)
    g = base_g + 0.12 * sig + 0.2 * base + rng.normal(0, noise * 0.7, n)
    bch = np.full(n, 40.0) + rng.normal(0, 0.2, n)
    clip = np.zeros(n)
    for t0, t1, kind in events:
        m = (t >= t0) & (t < t1)
        if kind == "burst":
            r[m] += 25.0 * np.sin(2 * np.pi * 2.2 * t[m]) + rng.normal(0, 4.0, m.sum())
        elif kind == "saturate":
            r[m] = 255.0 - 0.1 * np.abs(sig[m]) - rng.uniform(0, 0.2, m.sum())
            clip[m] = 0.85
        elif kind == "lift":
            r[m] = 40.0 + rng.normal(0, 0.3, m.sum())
            g[m] = 38.0 + rng.normal(0, 0.3, m.sum())
        elif kind == "lowperf":
            r[m] = base_r + 0.04 * sig[m] + rng.normal(0, 0.02, m.sum())
    keep = np.ones(n, bool)
    if drop_frac:
        keep &= rng.uniform(size=n) >= drop_frac
    if hiccup:
        keep &= ~((t >= hiccup[0]) & (t < hiccup[0] + hiccup[1]))
    return {"t": t[keep].tolist(), "r": r[keep].tolist(), "g": g[keep].tolist(), "b": bch[keep].tolist(),
            "clip_frac": clip[keep].tolist(), "truth_bpm": bpm, "beat_times": beat_times.tolist()}
