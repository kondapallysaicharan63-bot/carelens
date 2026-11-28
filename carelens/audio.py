"""Audio Tier 1 (W4): breathing rate from breath sounds, and cough events from YAMNet scores.

Reference implementation of the web "Breathing (sound)" and "Cough sounds" cards. The TypeScript port
``web/src/core/audio.ts`` mirrors this file; both read ``shared/params.json -> audio`` and replay
``shared/fixtures/audio_*.json``. Audio is processed in memory only; it is never stored or sent anywhere.
Wellbeing demonstration only. Not a medical device, no diagnosis.

Two parity levels (the ML model is not ported; parity starts at its score outputs):

* **audio level** (``envelope``): mono 16 kHz samples -> a 50 Hz log band-power envelope. STFT frames of
  ``frame_len`` = 512 samples, hop 320 (20 ms), Hann (``dsp.hann``), radix-2 FFT, power summed over
  150-2000 Hz (``envelope.band_*``), ``log10(power + eps)``. Frame ``k`` ends at ``t = (k*hop + frame_len) / fs``.
  ``clip_frac`` = fraction of the frame's samples with ``|x| >= clip_level``.
* **envelope level** (``BreathSoundRate``): envelope + per-frame clip fraction (+ YAMNet "Speech" scores) -> a
  breathing-rate update every ``step_s``.
* **score level** (``detect_label_events``): YAMNet per-frame scores of ONE label -> events + running count.

Score-frame time convention: ``t`` is the END of the 0.975 s YAMNet window, i.e. the moment the result is
available (``classify_wav`` returns that; the browser stamps frames with the stream clock). The native hop is
0.48 s (``yamnet.hop_samples``).

Breathing from breath sounds (close microphone)
-----------------------------------------------
Last ``window_s`` (30 s) of envelope, linearly detrended, Hann, zero-padded FFT, harmonic-sum fundamental in
0.1-0.7 Hz (the W1 path, ``dsp.spectral_peak``), band SNR. Gates, in priority order: ``no_mic`` (no samples),
``settling`` (less than ``settle_s`` of envelope), ``clipping`` (mean clip fraction in the window above
``clip_frac_max``), ``too_quiet`` (envelope dynamic range p95 - p5 below ``min_range_log10``: a muted or empty
signal), ``talking`` (share of YAMNet Speech frames >= ``speech_thr`` in the window at least
``speech_frac_max``), ``low_signal`` (SNR below ``min_snr``), ``ambiguous`` (doubled-rate rule below), ``ok``.
Quality = clip((SNR_dB - gate_dB) / (q_snr_full_db - gate_dB)), 0 when no window was analysed.

Doubled-rate rule (never report 2x or 1/2x): the envelope shows ONE bump per breath if only one phase is audible, or
TWO per breath (inhale + exhale), and the spectrum alone cannot tell (the harmonic sum favours the strong line). So the
estimate is accepted only with positive evidence from the bumps themselves. Bumps = prominent local maxima of the 0.3 s-smoothed,
detrended envelope (prominence >= ``bump_prom_frac`` of its p5-p95 range; width = FWHM of the bump's own prominence),
not touching the window edges, at least ``min_bump_s`` wide; ``f_b`` = their mean repetition rate. The spectral pick f* must match ``f_b`` ("bump" class) or
``f_b/2`` ("pair" class) within ``rate_tol``, else ambiguous. Alternation = odd/even bump amplitude (or width) differ by
>= ``alt_min`` of the mean AND the mean neighbour difference |a_i - a_{i+1}| / mean a >= ``alt_min`` AND the successive
differences strictly alternate in sign (needs >= ``min_alt_bumps`` bumps).
  * pair class (f* is half the bump rate): report f* if bumps alternate, else ambiguous.
  * bump class, f*/2 below the band (a doubled reading is impossible in band): report f*.
  * bump class, f*/2 in band: (a) bumps alternate -> inhale+exhale pairs, report f*/2; (b) no alternation AND the bump
    spans most of the interval (mean FWHM / interval >= ``width_min``) AND no spectral line at f*/2 (``sub_ratio``) AND
    >= ``min_bumps`` bumps -> one bump per breath, report f*; (c) otherwise ``ambiguous`` (reason ``ambiguous``, no rate).
Equal, short inhale+exhale bumps are fundamentally ambiguous and are withheld; an equal pair so broad that the two
bumps merge into one wide bump is indistinguishable from slow single-bump breathing (documented limit, see the
accuracy report). Coverage is therefore low for fast breathing and for equal double bumps, by design.

Limits (do not over-claim): counts sounds the model recognises as coughs: a wellbeing trend only. Breath-sound
breathing needs a close microphone (10-20 cm), quiet surroundings and no talking. The synthetic generators
below are documented approximations, not evidence of performance on people. Not a medical device.
"""
from __future__ import annotations

import base64
import hashlib
import math
import os
import urllib.request
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .dsp import band_bins, detrend_linear, fft_size, hann, parabolic_log, peak_bin, percentile, power_spectrum, spectral_peak, spectral_snr
from .params import PARAMS, Audio, AudioBreathing, LabelEvents

REASONS = ("ok", "no_mic", "too_quiet", "clipping", "talking", "settling", "low_signal", "ambiguous")
YAMNET_WINDOW_S = 0.975
_EPS = 1e-30


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ------------------------------------------------------------------ audio -> envelope

def n_frames(n_samples: int, p: Audio = PARAMS.audio) -> int:
    e = p.envelope
    return 0 if n_samples < e.frame_len else 1 + (n_samples - e.frame_len) // e.hop


def envelope(x: np.ndarray, p: Audio = PARAMS.audio) -> tuple[np.ndarray, np.ndarray]:
    """``(env, clip_frac)`` per frame of mono samples ``x`` (float, nominal range [-1, 1], at ``p.fs_hz``)."""
    e = p.envelope
    x = np.asarray(x, dtype=np.float64)
    n = n_frames(x.size, p)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(e.frame_len)[None, :] + e.hop * np.arange(n)[:, None]
    frames = x[idx]
    k_lo, k_hi = band_bins(e.frame_len, p.fs_hz, e.band_lo_hz, e.band_hi_hz)
    spec = np.fft.rfft(frames * hann(e.frame_len)[None, :], axis=1)
    power = np.sum(np.abs(spec[:, k_lo:k_hi + 1]) ** 2, axis=1)
    clip = np.mean(np.abs(frames) >= p.breathing.clip_level, axis=1)
    return np.log10(power + e.eps), clip


class EnvelopeStream:
    """Incremental ``envelope``: ``push(samples)`` returns the ``(t, env, clip_frac)`` of every frame completed.
    Equal to the batch ``envelope`` (a test guards it). ``t`` = frame end time in stream seconds."""

    def __init__(self, p: Audio = PARAMS.audio):
        self.p = p
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0)
        self._k = 0

    def push(self, samples: np.ndarray) -> list[tuple[float, float, float]]:
        e = self.p.envelope
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float64)])
        out = []
        while self._buf.size >= e.frame_len:
            env, clip = envelope(self._buf[:e.frame_len], self.p)
            out.append(((self._k * e.hop + e.frame_len) / self.p.fs_hz, float(env[0]), float(clip[0])))
            self._buf = self._buf[e.hop:]
            self._k += 1
        return out


# ------------------------------------------------------------------ breathing from the envelope

def find_bumps(y: np.ndarray, fs: float, p: AudioBreathing) -> list[tuple[float, float, float]]:
    """``(t_peak_s, prominence, fwhm_s)`` of each complete bump of the (detrended) envelope window ``y``: local maxima of the
    ``smooth_s`` boxcar-smoothed envelope whose prominence is at least ``bump_prom_frac`` of the (p95 - p5) range, with the
    width measured at half of the bump's own prominence; bumps cut by the window edges or shorter than ``min_bump_s`` are dropped."""
    m = max(1, int(round(p.smooth_s * fs)))
    c = np.concatenate([[0.0], np.cumsum(y)])
    s = (c[m:] - c[:-m]) / m   # boxcar mean of m samples (valid positions only)
    n = s.size
    min_prom = p.bump_prom_frac * (percentile(s, 95.0) - percentile(s, 5.0))
    bumps = []
    for i in range(1, n - 1):
        if not (s[i] > s[i - 1] and s[i] >= s[i + 1]):
            continue
        lo_l, j = s[i], i - 1
        while j >= 0 and s[j] <= s[i]:
            lo_l = min(lo_l, s[j]); j -= 1
        lo_r, k = s[i], i + 1
        while k < n and s[k] <= s[i]:
            lo_r = min(lo_r, s[k]); k += 1
        prom = s[i] - max(lo_l, lo_r)
        if prom < min_prom:
            continue
        level = s[i] - 0.5 * prom
        l = i
        while l > 0 and s[l - 1] > level:
            l -= 1
        r = i
        while r < n - 1 and s[r + 1] > level:
            r += 1
        if l > 0 and r < n - 1 and (r - l + 1) / fs >= p.min_bump_s:
            bumps.append((i / fs, float(prom), (r - l + 1) / fs))
    return bumps


def _alt(v: Sequence[float]) -> float:
    return abs(float(np.mean(v[0::2])) - float(np.mean(v[1::2]))) / float(np.mean(v))


def has_sub_line(power: np.ndarray, n_fft: int, fs: float, f: float, p: AudioBreathing) -> bool:
    """A local spectral peak at f/2 (inside the band, a full search window above its edge) with >= ``sub_ratio`` of f's power."""
    if f / 2.0 < p.band_lo_hz + p.harm_halfwidth_hz:
        return False
    k_lo, k_hi = band_bins(n_fft, fs, p.band_lo_hz, p.band_hi_hz)
    kf = peak_bin(power, n_fft, fs, f, p.harm_halfwidth_hz, k_lo, k_hi)
    kh = peak_bin(power, n_fft, fs, f / 2.0, p.harm_halfwidth_hz, k_lo, k_hi)
    return bool(kf >= 0 and k_lo < kh < k_hi and power[kh] >= power[kh - 1] and power[kh] >= power[kh + 1]
                and power[kh] >= p.sub_ratio * power[kf])


def resolve_rate(y: np.ndarray, power: np.ndarray, n_fft: int, fs: float, f: float, p: AudioBreathing) -> tuple[Optional[float], int, Optional[float]]:
    """Doubled-rate rule (module docstring). Returns ``(rate_hz or None if ambiguous, n_bumps, alternation)``."""
    bumps = find_bumps(y, fs, p)
    nb = len(bumps)
    if nb < p.min_bumps:
        return None, nb, None
    amps = [b[1] for b in bumps]
    widths = [b[2] for b in bumps]
    interval = (bumps[-1][0] - bumps[0][0]) / (nb - 1)
    f_b = 1.0 / interval
    alt, alternating = None, False
    if nb >= p.min_alt_bumps:
        v = amps if _alt(amps) >= _alt(widths) else widths   # the series (amplitude or width) that alternates more
        d = np.diff(v)
        alt = _alt(v)
        asym = float(np.mean(np.abs(d))) / float(np.mean(v))
        flips = bool(np.all(d[1:] * d[:-1] < 0))   # strictly alternating up/down: a drift or trend never does
        alternating = alt >= p.alt_min and asym >= p.alt_min and flips
    if abs(f - f_b) <= p.rate_tol * f_b:       # the pick is the bump rate
        if f / 2.0 < p.band_lo_hz * (1.0 - p.rate_tol):   # a doubled reading would be below the band (with its edge tolerance)
            return f, nb, alt
        if alternating:
            return f / 2.0, nb, alt
        single = (float(np.mean(widths)) / interval >= p.width_min) and not has_sub_line(power, n_fft, fs, f, p)
        return (f if single else None), nb, alt
    if abs(f - f_b / 2.0) <= p.rate_tol * f_b / 2.0:   # the pick is half the bump rate (two bumps per breath)
        return (f if alternating else None), nb, alt
    return None, nb, alt


def quality_score(snr: float, p: AudioBreathing) -> float:
    gate_db = 10.0 * math.log10(p.min_snr)
    return _clip01((10.0 * math.log10(max(snr, _EPS)) - gate_db) / (p.q_snr_full_db - gate_db))


class BreathSoundRate:
    """Feed envelope frames with ``push(t, env, clip_frac)`` (``env=None``: no microphone signal) and YAMNet Speech
    scores with ``push_speech(t, score)``; ``push`` returns an output dict once per ``step_s`` (else None):
    ``{t, bpm, quality, reported, reason, snr, bumps, alt, range_log10, clip_frac, speech_frac}`` (full precision).
    ``bpm`` is filled whenever a window was analysed and its rate is unambiguous; only ``reported`` outputs may be shown."""

    def __init__(self, params: Audio = PARAMS.audio):
        self.params = params
        self.p = params.breathing
        self.env_fs = params.fs_hz / params.envelope.hop
        self.window_n = int(math.floor(self.p.window_s * self.env_fs + 1e-6))
        self.settle_n = int(math.floor(self.p.settle_s * self.env_fs + 1e-6))
        self.reset()

    def reset(self) -> None:
        self._env: deque = deque(maxlen=self.window_n)
        self._clip: deque = deque(maxlen=self.window_n)
        self._speech: deque = deque()
        self._next_out: Optional[float] = None
        self._last_t: Optional[float] = None
        self._no_mic = False

    def push_speech(self, t: float, score: float) -> None:
        self._speech.append((t, score))

    def push(self, t: float, env: Optional[float], clip_frac: float = 0.0) -> Optional[dict]:
        p = self.p
        if self._last_t is not None and not t > self._last_t:
            return None   # non-monotonic timestamp: ignore
        self._last_t = t
        if self._next_out is None:
            self._next_out = t + p.step_s
        if env is None or not math.isfinite(env):
            self._env.clear(); self._clip.clear(); self._speech.clear()
            self._no_mic = True
        else:
            self._no_mic = False
            self._env.append(env); self._clip.append(clip_frac)
        if t < self._next_out:
            return None
        while self._next_out <= t:
            self._next_out += p.step_s
        return self._emit(t)

    def _emit(self, t: float) -> dict:
        p = self.p
        out = {"t": t, "bpm": None, "quality": 0.0, "reported": False, "reason": "ok", "snr": None,
               "bumps": None, "alt": None, "range_log10": None, "clip_frac": None, "speech_frac": None,
               "span_s": None}
        if self._no_mic:
            out["reason"] = "no_mic"
            return out
        n = len(self._env)
        if n < self.settle_n:
            out["reason"] = "settling"
            return out
        y = np.fromiter(self._env, dtype=np.float64, count=n)
        out["span_s"] = n / self.env_fs   # analysed window length (s)
        out["clip_frac"] = float(np.mean(np.fromiter(self._clip, dtype=np.float64, count=n)))
        out["range_log10"] = percentile(y, 95.0) - percentile(y, 5.0)
        while self._speech and self._speech[0][0] <= t - p.window_s:
            self._speech.popleft()
        if self._speech:
            out["speech_frac"] = sum(1 for _, s in self._speech if s >= p.speech_thr) / len(self._speech)
        if out["clip_frac"] > p.clip_frac_max:
            out["reason"] = "clipping"
        elif out["range_log10"] < p.min_range_log10:
            out["reason"] = "too_quiet"
        elif out["speech_frac"] is not None and out["speech_frac"] >= p.speech_frac_max:
            out["reason"] = "talking"
        else:
            self._analyse(y, out)
        return out

    def _analyse(self, y: np.ndarray, out: dict) -> None:
        p = self.p
        y = detrend_linear(y)
        n_fft = fft_size(y.size, p.zero_pad_factor)
        power = power_spectrum(y, n_fft)
        f = spectral_peak(power, n_fft, self.env_fs, p)
        snr = spectral_snr(power, n_fft, self.env_fs, f, p)
        if not (math.isfinite(f) and math.isfinite(snr)) or snr < p.min_snr:
            out["reason"] = "low_signal"   # includes degenerate windows: never report, never emit NaN
            out["snr"] = snr if math.isfinite(snr) else None
            return
        rate, out["bumps"], out["alt"] = resolve_rate(y, power, n_fft, self.env_fs, f, p)
        if rate is None:
            out["reason"] = "ambiguous"
            out["snr"] = snr
            return
        snr = spectral_snr(power, n_fft, self.env_fs, rate, p)
        out["snr"], out["quality"], out["bpm"] = snr, quality_score(snr, p), 60.0 * rate
        out["reason"], out["reported"] = "ok", True


def run_breath_env(inp: dict) -> dict:
    """Fixture runner: replay ``{env, clip?, speech?}``; speech frames ``[t, score]`` are delivered once the envelope
    clock reaches their time. ``env`` entries may be null (no microphone). Frame ``i`` is at ``t = i / fs_env``."""
    est = BreathSoundRate()
    fs = inp["fs_env"]
    clip = inp.get("clip")
    speech = sorted(inp.get("speech", []))
    j, ups = 0, []
    for i, v in enumerate(inp["env"]):
        t = i / fs
        while j < len(speech) and speech[j][0] <= t:
            est.push_speech(speech[j][0], speech[j][1])
            j += 1
        u = est.push(t, v, 0.0 if clip is None else clip[i])
        if u is not None:
            ups.append(u)
    return {"updates": ups}


# ------------------------------------------------------------------ label events (cough, later snoring, ...)

def detect_label_events(frames: Sequence[Sequence[float]], p: LabelEvents = PARAMS.audio.cough) -> dict:
    """``frames`` = ascending ``(t, score)`` of one label. Returns ``{"events": [{t_start, t_end, peak_score}], "count"}``."""
    segs: list[list] = []   # [t_start, t_end, peak, n_frames]
    cur: Optional[list] = None
    for t, s in frames:
        if cur is None:
            if s >= p.on_thr:
                cur = [t, t, s, 1]
        elif s >= p.off_thr:
            cur[1] = t; cur[2] = max(cur[2], s); cur[3] += 1
        else:
            segs.append(cur); cur = None
    if cur is not None:
        segs.append(cur)
    events: list[list] = []
    for t0, t1, peak, nf in segs:
        if nf < p.min_frames:
            continue
        if events and (t0 - events[-1][1] < p.merge_gap_s - 1e-9 or t0 - events[-1][0] < p.refractory_s - 1e-9):
            ev = events[-1]
            ev[1] = t1; ev[2] = max(ev[2], peak)
        else:
            events.append([t0, t1, peak])
    return {"events": [{"t_start": a, "t_end": b, "peak_score": c} for a, b, c in events], "count": len(events)}


def run_label_events(inp: dict) -> dict:
    """Fixture runner: ``{label, frames: [[t, score], ...]}`` with the parameters of that label's section."""
    sections = {"Cough": PARAMS.audio.cough}
    return detect_label_events(inp["frames"], sections[inp["label"]])


# ------------------------------------------------------------------ audio-level fixture runner

def decode_pcm(inp: dict) -> np.ndarray:
    """int16 little-endian PCM (base64) -> float in [-1, 1)."""
    return np.frombuffer(base64.b64decode(inp["pcm_b64"]), dtype="<i2").astype(np.float64) / 32768.0


def encode_pcm(x: np.ndarray) -> str:
    q = np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def run_audio_envelope(inp: dict) -> dict:
    env, clip = envelope(decode_pcm(inp))
    return {"envelope": env.tolist(), "clip_frac": clip.tolist()}


# ------------------------------------------------------------------ synthetic signals (tests + fixtures)
#
# Documented SYNTHETIC approximations, not evidence about real breathing or coughing.
#  * Breath envelope: log10 band power = floor + amp * bump train + slow drift + white jitter. One bump per
#    breath (inhale only audible), or two per breath (``double``: inhale amp 1, exhale amp ``exhale``).
#  * Breath audio: white noise whose gain follows the same bump train (the envelope above, undone).
#  * Cough scores: a floor of small scores with cough-shaped bursts of high scores.

def _bump_train(t: np.ndarray, bpm: float, phase: float, width: float, double: bool, exhale: float) -> np.ndarray:
    """Sum of raised-cosine bumps (duration ``width`` x half-period) at the breath period."""
    period = 60.0 / bpm
    half = period / 2.0 if double else period
    out = np.zeros_like(t)
    k0 = int(math.floor((t[0] - phase) / half)) - 1
    k1 = int(math.ceil((t[-1] - phase) / half)) + 1
    dur = width * half
    for k in range(k0, k1 + 1):
        amp = 1.0 if (not double or k % 2 == 0) else exhale
        x = (t - (phase + k * half)) / dur
        m = (x >= 0) & (x < 1)
        out[m] += amp * 0.5 * (1.0 - np.cos(2 * np.pi * x[m]))
    return out


def synth_breath_envelope(duration: float = 50.0, bpm: float = 15.0, *, seed: int = 0, amp: float = 0.6,
                          floor: float = 1.0, jitter: float = 0.06, drift: float = 0.15, width: Optional[float] = None,
                          double: bool = False, exhale: float = 0.5, fs_env: Optional[float] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width = (0.6 if double else 0.9) if width is None else width   # one broad bump per breath, or two shorter ones
    fs = PARAMS.audio.fs_hz / PARAMS.audio.envelope.hop if fs_env is None else fs_env
    t = np.arange(int(duration * fs)) / fs
    env = floor + amp * _bump_train(t, bpm, rng.uniform(0.0, 60.0 / bpm), width, double, exhale)
    env = env + drift * np.sin(2 * np.pi * rng.uniform(0.02, 0.05) * t + rng.uniform(0, 6.28))
    return env + rng.normal(0.0, jitter, t.size)


def synth_shaped_noise(duration: float = 3.0, *, seed: int = 0, fs: Optional[int] = None, level: float = 0.05) -> np.ndarray:
    """Coloured noise (1/sqrt(f) amplitude slope) with a slow amplitude modulation and a short click (peak ~ ``level`` x 4)."""
    fs = PARAMS.audio.fs_hz if fs is None else fs
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    spec = np.fft.rfft(rng.normal(0.0, 1.0, n))
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / fs), 50.0)
    x = np.fft.irfft(spec / np.sqrt(f / 50.0), n=n)
    x = x / np.std(x) * level
    t = np.arange(n) / fs
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))
    c = int(0.8 * n)
    x[c:c + 40] += 4.0 * level * np.hanning(40)
    return np.clip(x, -1.0, 1.0)


def synth_breath_audio(duration: float = 60.0, bpm: float = 15.0, *, seed: int = 0, level: float = 0.02,
                       double: bool = False, exhale: float = 0.5, depth: float = 6.0, width: Optional[float] = None) -> np.ndarray:
    """Breath-like noise: coloured noise multiplied by (1 + ``depth`` x bump train)."""
    fs = PARAMS.audio.fs_hz
    width = (0.6 if double else 0.9) if width is None else width
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    gain = 1.0 + depth * _bump_train(t, bpm, rng.uniform(0.0, 60.0 / bpm), width, double, exhale)
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / fs), 50.0)
    noise = np.fft.irfft(np.fft.rfft(rng.normal(0.0, 1.0, n)) / np.sqrt(f / 50.0), n=n)
    return np.clip(noise / np.std(noise) * level * gain, -1.0, 1.0)


def synth_scores(duration: float, bursts: Sequence[Sequence[float]], *, seed: int = 0, hop: float = 0.48,
                 floor: float = 0.03, noise: float = 0.02, t0: float = YAMNET_WINDOW_S) -> list[list[float]]:
    """Per-frame ``[t, score]`` at ``hop``: ``floor`` + noise plus ``bursts`` = ``(t_start, n_frames, peak)`` with a
    rise/fall profile (0.45, 1, 0.8, 0.5, 0.25 x peak, then the floor)."""
    rng = np.random.default_rng(seed)
    n = int(duration / hop)
    s = floor + np.abs(rng.normal(0.0, noise, n))
    prof = [0.45, 1.0, 0.8, 0.5, 0.25]
    for ts, nf, peak in bursts:
        i0 = int(round((ts - t0) / hop))
        for j in range(nf):
            if 0 <= i0 + j < n:
                s[i0 + j] = max(s[i0 + j], peak * prof[min(j, len(prof) - 1)])
    return [[t0 + i * hop, float(np.clip(v, 0.0, 1.0))] for i, v in enumerate(s)]


# ------------------------------------------------------------------ offline YAMNet (recorded-sample tests)

YAMNET_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
YAMNET_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"   # same pin as web/scripts/fetch-models.mjs
YAMNET_PATH = Path(__file__).resolve().parent.parent / "web" / "public" / "models" / "yamnet.tflite"   # git-ignored


def ensure_yamnet(path: Path = YAMNET_PATH) -> Path:
    """The self-hosted ``yamnet.tflite`` (downloaded to the git-ignored ``web/public/models`` if absent, SHA-256 checked)."""
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == YAMNET_SHA256:
        return path
    data = urllib.request.urlopen(YAMNET_URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != YAMNET_SHA256:
        raise RuntimeError("yamnet.tflite checksum mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def read_wav_mono16k(path: str | Path) -> np.ndarray:
    """16-bit PCM WAV -> mono float at 16 kHz (linear resampling if needed). For recorded samples of consenting people only."""
    import wave
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        ch, sr, raw = w.getnchannels(), w.getframerate(), w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, ch).mean(axis=1) / 32768.0
    if sr != PARAMS.audio.fs_hz:
        n = int(round(x.size * PARAMS.audio.fs_hz / sr))
        x = np.interp(np.arange(n) * sr / PARAMS.audio.fs_hz, np.arange(x.size), x)
    return x


def classify_samples(x: np.ndarray, labels: Sequence[str] = ("Cough", "Speech"), hop_s: Optional[float] = None) -> dict:
    """Run the real YAMNet (MediaPipe Python ``AudioClassifier``) over mono 16 kHz samples with the browser's
    framing: 15600-sample windows every ``hop_samples`` (0.48 s). Returns ``{"t": [...], label: [...]}`` where ``t`` is the
    window END time. Needs ``mediapipe``; the model is fetched by ``ensure_yamnet``."""
    from mediapipe.tasks.python import BaseOptions, audio
    from mediapipe.tasks.python.components.containers import audio_data as ad

    y = PARAMS.audio.yamnet
    hop = y.hop_samples if hop_s is None else int(round(hop_s * PARAMS.audio.fs_hz))
    opts = audio.AudioClassifierOptions(base_options=BaseOptions(model_asset_path=str(ensure_yamnet())),
                                        running_mode=audio.RunningMode.AUDIO_CLIPS, max_results=-1)
    clf = audio.AudioClassifier.create_from_options(opts)
    try:
        out: dict = {"t": [], **{lab: [] for lab in labels}}
        x = np.asarray(x, dtype=np.float32)
        for s in range(0, x.size - y.window_samples + 1, hop):
            res = clf.classify(ad.AudioData.create_from_array(x[s:s + y.window_samples], PARAMS.audio.fs_hz))
            scores = {c.category_name: c.score for c in res[0].classifications[0].categories}
            out["t"].append((s + y.window_samples) / PARAMS.audio.fs_hz)
            for lab in labels:
                out[lab].append(float(scores.get(lab, 0.0)))
        return out
    finally:
        clf.close()


def classify_wav(path: str | Path, labels: Sequence[str] = ("Cough", "Speech"), hop_s: Optional[float] = None) -> dict:
    """Offline helper: per-frame YAMNet scores of a WAV (see ``classify_samples``); feed ``zip(t, scores)`` to
    ``detect_label_events``. Only use recordings of people who consented to it, and never commit them."""
    return classify_samples(read_wav_mono16k(path), labels, hop_s)
