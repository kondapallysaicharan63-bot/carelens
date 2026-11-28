"""W10 (Tier 2, Experimental): possible pauses in breathing sounds, and sound cues for a possible fall.

Reference implementation; ``web/src/core/sleepSounds.ts`` mirrors it and both replay ``shared/fixtures/sleep_*.json``.
Audio is processed in memory only. Wellbeing demonstration only. NOT a medical device and NOT a sleep test: it never
diagnoses anything. Wording: "possible pause in breathing sounds", never anything stronger.

Possible pauses (phone near the bed)
------------------------------------
* ``BumpStream``: the 50 Hz log band-power envelope of ``audio.py`` -> breath bumps. The envelope is boxcar-smoothed
  (``bump.smooth_s``), a floor is tracked as the rolling minimum over ``bump.floor_win_s``, and a bump is a stretch above
  ``floor + rise_on_log10`` (until it falls under ``rise_off_frac`` of that), at least ``min_bump_s`` long; its time is the
  smoothed maximum. A stretch longer than ``max_bump_s``, or whose rise exceeds ``spike_ratio`` x the median of the last bump
  rises (needs ``typ_bumps``), is a loud non-breath sound ("spike"), not a bump.
* Inhale and exhale of one breath may show as two bumps (or one merged bump): bumps closer than ``bump.merge_s`` (below the
  period of the fastest in-band breathing) are one breath, timed at the louder one.
  Envelope values under ``bump.silence_log10`` (digital silence, e.g. a microphone that has just started) are no signal at all:
  they restart the tracker's running state instead of setting a floor.
* ``PauseTracker``: the interval between consecutive bumps (peak to peak) is a POSSIBLE PAUSE when it is
  >= ``pause_min_s`` and <= ``pause_max_s`` (breathing resumed), the ``pre_regular_s`` before it held regular breathing
  (>= ``pre_min_bumps`` bumps covering the window, inter-bump CV <= ``pre_cv_max``, rate inside ``rate_min_bpm``..``rate_max_bpm``)
  and no YAMNet Speech frame (score >= ``speech_thr``) or spike overlaps it (``overlap_margin_s`` slack). Longer silences are
  counted as ``no_sound_gaps`` (the sleeper moved or left), never as pauses.
* ``regular_s`` = total time of intervals between bumps inside regular breathing, free of speech and spikes (a counted pause
  included). The per-hour rate is withheld (None) while ``regular_s`` < ``min_valid_regular_s`` (30 min).

Sound cue for a possible fall: ``sound_cue`` is true when a cue-label sound (Thump, thud / Screaming / Glass) starts within
``corroborate_s`` after, or ends within ``corroborate_s`` before, the fall's [t_onset, t_confirmed] interval.

Limits (do not over-claim): phone placement, bedding, a fan and movement all change what the microphone hears. The synthetic
generators in ``fixtures.py`` are documented approximations, not evidence of performance on people.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence

from .audio import YAMNET_WINDOW_S
from .params import PARAMS, AudioEvents, SleepSounds

MAX_KEEP = 500        # pauses kept in the view (the count itself is unbounded)
MAX_RECENT = 600      # bumps kept for the regularity window


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5 + 1e-9))


def _median(v: Sequence[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class BumpStream:
    """Envelope frames -> ``("bump", t_peak, rise)`` and ``("spike", t_start, t_end)`` records (see module docstring)."""

    def __init__(self, p: SleepSounds = PARAMS.sleep_sounds, fs_env: Optional[float] = None):
        self.p = p
        self.b = p.bump
        self.fs = PARAMS.audio.fs_hz / PARAMS.audio.envelope.hop if fs_env is None else fs_env
        self.m = max(1, _round_half_up(self.b.smooth_s * self.fs))
        self.floor_n = max(1, int(math.floor(self.b.floor_win_s * self.fs + 1e-6)))
        self.reset()

    def reset(self) -> None:
        self._ring: deque = deque(maxlen=self.m)
        self._mono: deque = deque()
        self._k = 0
        self._seg: Optional[list] = None   # [t0, peak, t_peak, peak_rise]
        self._rises: deque = deque(maxlen=16)

    def push(self, t: float, env: float) -> list[tuple]:
        b = self.b
        self._ring.append(env)
        if len(self._ring) < self.m:
            return []
        acc = 0.0
        for v in self._ring:
            acc += v
        s = acc / self.m
        ts = t - ((self.m - 1) / 2.0) / self.fs
        k = self._k
        self._k += 1
        while self._mono and self._mono[-1][1] >= s:
            self._mono.pop()
        self._mono.append((k, s))
        while self._mono[0][0] <= k - self.floor_n:
            self._mono.popleft()
        rise = s - self._mono[0][1]
        out: list[tuple] = []
        seg = self._seg
        if seg is None:
            if rise >= b.rise_on_log10:
                self._seg = [ts, s, ts, rise]
            return out
        if s > seg[1]:
            seg[1], seg[2], seg[3] = s, ts, rise
        if rise >= b.rise_on_log10 * b.rise_off_frac:
            return out
        self._seg = None
        dur = ts - seg[0]
        if dur > b.max_bump_s:
            out.append(("spike", seg[0], ts))
        elif dur >= b.min_bump_s:
            if len(self._rises) >= b.typ_bumps and seg[3] > b.spike_ratio * _median(self._rises):
                out.append(("spike", seg[0], ts))
            else:
                self._rises.append(seg[3])
                out.append(("bump", seg[2], seg[3]))
        return out


class PauseTracker:
    """Possible pauses in breathing sounds, incrementally and in bounded memory (module docstring).

    Feed ``push_frame(t, env)`` (envelope frames; ``env`` None/non-finite = no signal) and ``push_speech(t, score)``; or, for
    the bump-level fixtures, ``push_bump(t)`` / ``push_spike(t0, t1)``. ``view()`` returns
    ``{count, pauses: [{t_start, t_end, duration_s}], regular_s, rate_per_h (None while withheld), no_sound_gaps}``."""

    def __init__(self, p: SleepSounds = PARAMS.sleep_sounds, fs_env: Optional[float] = None):
        self.p = p
        self.stream = BumpStream(p, fs_env)
        self.reset()

    def reset(self) -> None:
        self.stream.reset()
        self.count = 0
        self.regular_s = 0.0
        self.no_sound_gaps = 0
        self.pauses: deque = deque(maxlen=MAX_KEEP)
        self._speech: deque = deque()
        self._spikes: deque = deque()
        self._recent: deque = deque(maxlen=MAX_RECENT)
        self._last_rise: Optional[float] = None
        self._now = 0.0

    def break_stream(self) -> None:
        """The signal broke (microphone lost): forget the running bump/regularity state, keep the totals."""
        self.stream.reset()
        self._recent.clear()
        self._last_rise = None
        self._speech.clear()
        self._spikes.clear()

    def _trim(self, t: float) -> None:
        self._now = max(self._now, t)
        horizon = self._now - self.p.pause_max_s - 10.0
        while self._speech and self._speech[0][0] < horizon:
            self._speech.popleft()
        while self._spikes and self._spikes[0][1] < horizon:
            self._spikes.popleft()

    def push_speech(self, t: float, score: float) -> None:
        self._speech.append((t, score))
        self._trim(t)

    def push_frame(self, t: float, env: Optional[float]) -> None:
        if env is None or not math.isfinite(env) or env < self.p.bump.silence_log10:
            self.break_stream()   # no microphone signal, or digital silence (a muted or just-started input): nothing to measure
            return
        self._trim(t)
        for rec in self.stream.push(t, env):
            if rec[0] == "bump":
                self.push_bump(rec[1], rec[2])
            else:
                self.push_spike(rec[1], rec[2])

    def push_spike(self, t0: float, t1: float) -> None:
        self._spikes.append((t0, t1))
        self._trim(t1)

    def _overlaps(self, a: float, b: float) -> bool:
        """Speech or a loud non-breath sound inside [a, b] (with the margin)."""
        p = self.p
        lo, hi = a - p.overlap_margin_s, b + p.overlap_margin_s
        for t, s in self._speech:
            if s >= p.speech_thr and t > lo and t - YAMNET_WINDOW_S < hi:
                return True
        for t0, t1 in self._spikes:
            if t1 > lo and t0 < hi:
                return True
        return False

    def _regular(self, t_end: float) -> bool:
        """Regular breathing in the ``pre_regular_s`` ending at ``t_end``: enough bumps covering the window, low CV, rate in band."""
        p = self.p
        w = [x for x in self._recent if x >= t_end - p.pre_regular_s]
        if len(w) < p.pre_min_bumps:
            return False
        iv = [w[i + 1] - w[i] for i in range(len(w) - 1)]
        mean = sum(iv) / len(iv)
        if not mean > 0:
            return False
        sd = math.sqrt(sum((x - mean) ** 2 for x in iv) / len(iv))
        rate = 60.0 / mean
        covers = w[0] <= t_end - p.pre_regular_s + 1.5 * mean
        return covers and sd / mean <= p.pre_cv_max and p.rate_min_bpm <= rate <= p.rate_max_bpm

    def push_bump(self, t: float, rise: Optional[float] = None) -> None:
        """A breath bump at time ``t`` (``rise`` = its height, to pick the louder of two bumps of one breath)."""
        p = self.p
        self._trim(t)
        if self._recent:
            tp = self._recent[-1]
            iv = t - tp
            if 0 < iv < p.bump.merge_s:   # the second bump of the same breath: keep the louder one's time
                if rise is not None and (self._last_rise is None or rise > self._last_rise):
                    self._recent[-1] = t
                    self._last_rise = rise
                return
            if iv >= p.pause_min_s:
                if iv <= p.pause_max_s:
                    if self._regular(tp) and not self._overlaps(tp, t):
                        self.count += 1
                        self.regular_s += iv
                        self.pauses.append((tp, t, iv))
                else:
                    self.no_sound_gaps += 1
            elif iv > 0:
                self._recent.append(t)
                self._last_rise = rise
                if self._regular(t) and not self._overlaps(tp, t):
                    self.regular_s += iv
                self._trim_recent(t)
                return
            else:
                return   # non-monotonic: ignore
        self._recent.append(t)
        self._last_rise = rise
        self._trim_recent(t)

    def _trim_recent(self, t: float) -> None:
        horizon = t - self.p.pre_regular_s - self.p.pause_max_s
        while self._recent and self._recent[0] < horizon:
            self._recent.popleft()

    def view(self) -> dict:
        p = self.p
        rate = self.count / (self.regular_s / 3600.0) if self.regular_s >= p.min_valid_regular_s else None
        return {"count": self.count,
                "pauses": [{"t_start": a, "t_end": b, "duration_s": d} for a, b, d in self.pauses],
                "regular_s": self.regular_s, "rate_per_h": rate, "no_sound_gaps": self.no_sound_gaps}


# ------------------------------------------------------------------ sound cue for a possible fall

def sound_cue(fall: dict, sounds: Sequence[dict], p: AudioEvents = PARAMS.audio_events) -> bool:
    """Pure: is there a cue-label sound (``p.cue_labels``) within ``p.corroborate_s`` of the fall's ``[t_onset, t_confirmed]``?
    ``sounds`` = ``{label, t_start, t_end}``."""
    lo, hi = fall["t_onset"] - p.corroborate_s, fall["t_confirmed"] + p.corroborate_s
    return any(s["label"] in p.cue_labels and s["t_end"] >= lo and s["t_start"] <= hi for s in sounds)


# ------------------------------------------------------------------ fixture runners

def run_sleep_bumps(inp: dict) -> dict:
    """Bump-level: ``{bumps: [t], spikes: [[t0, t1]], speech: [[t, score]]}`` replayed in arrival order (a speech frame at ``t``,
    a spike at its end, a bump at its time; speech first on ties)."""
    items = ([(t, 0, (t, s)) for t, s in inp.get("speech", [])] + [(b, 1, (a, b)) for a, b in inp.get("spikes", [])]
             + [(t, 2, t) for t in inp["bumps"]])
    tr = PauseTracker()
    for _, kind, v in sorted(items, key=lambda x: (x[0], x[1])):
        if kind == 0:
            tr.push_speech(*v)
        elif kind == 1:
            tr.push_spike(*v)
        else:
            tr.push_bump(v)
    return tr.view()


def run_sleep_env(inp: dict) -> dict:
    """Envelope-level: ``{fs_env, env, speech?}``; speech frames are delivered once the envelope clock reaches their time."""
    tr = PauseTracker(fs_env=inp["fs_env"])
    speech = sorted(inp.get("speech", []))
    j = 0
    for i, v in enumerate(inp["env"]):
        t = i / inp["fs_env"]
        while j < len(speech) and speech[j][0] <= t:
            tr.push_speech(*speech[j])
            j += 1
        tr.push_frame(t, v)
    return tr.view()


def run_sound_cue(inp: dict) -> dict:
    return {"cues": [sound_cue(f, inp["sounds"]) for f in inp["falls"]]}


# ------------------------------------------------------------------ synthetic signals (tests + fixtures)
#
# Documented SYNTHETIC approximations, not evidence about real sleep sounds: regular breath bumps (as audio.synth_breath_envelope)
# with silent stretches, optional irregular timing, optional loud bursts.

def synth_breath_starts(duration: float, bpm: float = 14.0, gaps: Sequence[Sequence[float]] = (), *, seed: int = 0,
                        irregular: Sequence[Sequence[float]] = (), irreg_frac: float = 0.45) -> list[float]:
    """Breath start times (s): one per ``60 / bpm`` s with the breaths of each ``gaps`` span (t0, t1) removed. Inside an
    ``irregular`` span each start is shifted by a uniform +-``irreg_frac`` x period."""
    import numpy as np
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    t = float(rng.uniform(0.0, period))
    out = []
    while t < duration:
        s = t
        if any(a <= t < b for a, b in irregular):
            s = t + float(rng.uniform(-irreg_frac, irreg_frac)) * period
        if not any(a <= t < b for a, b in gaps):
            out.append(s)
        t += period
    return out


def synth_envelope_from_starts(starts: Sequence[float], duration: float, *, seed: int = 0, bpm: float = 14.0, amp: float = 0.6,
                               floor: float = 1.0, jitter: float = 0.04, drift: float = 0.03, width: float = 0.5,
                               bursts: Sequence[Sequence[float]] = (), fs_env: Optional[float] = None,
                               pair: Optional[Sequence[float]] = None):
    """50 Hz log-envelope: ``floor`` + raised-cosine bumps (``width`` x period long) at ``starts`` + slow drift + jitter;
    ``bursts`` = (t0, t1, extra_log10) add a loud flat burst (cough / speech-like); ``pair`` = (offset_s, amp2, width2_s) adds a
    second, louder bump (the exhale) ``offset_s`` after each start, and ``width`` x period then means the first bump's length."""
    import numpy as np
    rng = np.random.default_rng(seed + 7)
    fs = PARAMS.audio.fs_hz / PARAMS.audio.envelope.hop if fs_env is None else fs_env
    t = np.arange(int(duration * fs)) / fs
    env = np.full(t.size, floor)
    dur = width * 60.0 / bpm
    for s in starts:
        x = (t - s) / dur
        m = (x >= 0) & (x < 1)
        env[m] += amp * 0.5 * (1.0 - np.cos(2 * np.pi * x[m]))
        if pair is not None:
            x2 = (t - (s + pair[0])) / pair[2]
            m2 = (x2 >= 0) & (x2 < 1)
            env[m2] += pair[1] * 0.5 * (1.0 - np.cos(2 * np.pi * x2[m2]))
    for a, b, extra in bursts:
        env[(t >= a) & (t < b)] += extra
    env = env + drift * np.sin(2 * np.pi * rng.uniform(0.02, 0.05) * t + rng.uniform(0, 6.28))
    return env + rng.normal(0.0, jitter, t.size)
