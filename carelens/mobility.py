"""Guided mobility tests from pose landmarks (W7, Experimental): 30 s chair stand, five-times sit-to-stand, Timed Up
and Go, walking speed.

Python reference; ``web/src/core/mobility.ts`` mirrors it operation for operation and both read
``shared/params.json -> mobility``. Not a medical device: a wellbeing trend to compare with your own earlier results,
never a norm, a cut-off, a pass/fail or a "fall risk" score. Plain Python (math only) so the TypeScript port can match it.

Chain: landmarks -> ``PresenceTracker`` (smoothing, short-dropout holding) -> ``sample_from_landmarks`` -> ``Sample``
series -> one of the four test functions. Coordinates are frame-height units (x * aspect, y as is).

Signals per frame (``Sample``)
  knee_deg  hip-knee-ankle angle; both legs averaged when equally visible (visibility within ``leg_vis_margin``), else the
            more visible leg alone; legs below ``min_visibility`` are ignored, no visible leg -> the frame is not ``ok``.
  hip_h     (ankle y - hip y) / (|hip-knee| + |knee-ankle|): hip height scaled by leg length (same leg choice).
  x         hip-centre x;  height  nose-to-ankle y extent (apparent body height);  ankle_dx  left minus right ankle x.

Sit/stand state machine (``RepMachine``), hysteresis on the knee angle:
  SEATED  (knee < sit_deg)  --knee > rise_deg-->  RISING  --knee >= stand_deg and hip_h >= stand_hip_frac for
  stand_dwell_s-->  STANDING  --knee < sit_deg for sit_dwell_s-->  SEATED (one completed rep).
  RISING back to SEATED without standing is a *partial rise* and does not count. Event times are the first frame of the
  condition that then held for its dwell ("stand time" = first frame of the held standing posture; "seat time" = first
  frame of the held seated posture). The rise onset is the first frame with knee > rise_deg.

Conventions
  chair_stand_30s: frames in [go, go + chair_stand_s]; reps = completed SEATED->STANDING->SEATED cycles (a rep still in
    progress at the end is not counted; flag ``rep_in_progress``); must start seated.
  sit_to_stand_5x: the clock starts at "Go" or, if the first rise began up to ``sts_early_s`` earlier, at that rise onset
    (an earlier start is not credited); it stops when the ``sts_reps``-th STANDING is reached (its stand time). Per-rep
    times are the intervals between successive stand times, the first measured from the clock start.
  tug: same clock start; ends at the seat time of the final sit. Phases: rise (start -> first stand time), walk_out (->
    turn_1 start), turn_1, walk_back (-> arrival), turn_2 (arrival -> sit onset), sit (sit onset -> seat time); they sum to
    the total. Turn_1 = the stretch around the farthest point from the chair where the hip-centre speed is below
    turn_speed_frac of the walking speed; the farthest point must be >= min_excursion_bodies body heights away and the
    person must come back >= min_return_frac of the way (else ``no_turn``). Arrival = next slow-down after turn_1 (or the
    sit onset). Sit onset = the last frame of upright posture (knee >= stand_deg, hip high) before sitting. Only side-on
    walks along the image x axis are supported (a walk along the camera axis shows no turn: ``no_turn``).
  walking_speed: see ``walking_speed``. Approximate: depends on the entered height, assumes a side-on view and a roughly
    constant body-to-camera distance; perspective and scale errors are not corrected.

Withheld (``reported`` False, no numbers) reasons: ``out_of_view`` (too few frames with visible legs / no scale),
``not_seated_at_start``, ``no_rise``, ``no_turn``, ``not_seated_at_end``, ``incomplete`` (5x: fewer than the required
standings), ``not_side_on``, ``no_walk``, ``too_short``, ``bad_height``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from . import pose as P
from .params import PARAMS, Mobility
from .tracking import PresenceTracker

MOBILITY_INDICES = [P.NOSE, P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP, P.L_KNEE, P.R_KNEE, P.L_ANKLE, P.R_ANKLE]

TESTS = ("chair_stand_30s", "sit_to_stand_5x", "tug", "walking_speed")


@dataclass
class Sample:
    t: float
    ok: bool = False            # a measurement with at least one visible leg (never true for held frames)
    held: bool = False          # tracker repeated the last pose (a dropout): not a measurement
    knee_deg: float = 0.0
    hip_h: float = 0.0
    x: float = 0.0
    height: Optional[float] = None
    ankle_dx: Optional[float] = None


def _leg(lm, hip: int, knee: int, ank: int, aspect: float):
    vis = min(lm[hip][2], lm[knee][2], lm[ank][2])
    ax, ay = lm[hip][0] * aspect - lm[knee][0] * aspect, lm[hip][1] - lm[knee][1]
    bx, by = lm[ank][0] * aspect - lm[knee][0] * aspect, lm[ank][1] - lm[knee][1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-6 or nb < 1e-6:
        return None
    c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return vis, math.degrees(math.acos(c)), (lm[ank][1] - lm[hip][1]) / (na + nb)


def sample_from_landmarks(t: float, lm, aspect: float, mp: Mobility = PARAMS.mobility) -> Sample:
    """One frame's signals from 33 normalised landmark rows [x, y, visibility] (None = no person)."""
    if lm is None:
        return Sample(t=t)
    legs = []
    for h, k, a in ((P.L_HIP, P.L_KNEE, P.L_ANKLE), (P.R_HIP, P.R_KNEE, P.R_ANKLE)):
        g = _leg(lm, h, k, a, aspect)
        if g is not None and g[0] >= mp.min_visibility:
            legs.append(g)
    if not legs:
        return Sample(t=t)
    if len(legs) == 2 and abs(legs[0][0] - legs[1][0]) > mp.leg_vis_margin:
        legs = [legs[0] if legs[0][0] > legs[1][0] else legs[1]]
    knee = sum(g[1] for g in legs) / len(legs)
    hip_h = sum(g[2] for g in legs) / len(legs)
    x = (lm[P.L_HIP][0] + lm[P.R_HIP][0]) / 2.0 * aspect
    ank = [lm[k] for k in (P.L_ANKLE, P.R_ANKLE) if lm[k][2] >= mp.min_visibility]
    height = None
    if ank and lm[P.NOSE][2] >= mp.min_visibility:
        height = sum(r[1] for r in ank) / len(ank) - lm[P.NOSE][1]
    dx = None
    if len(ank) == 2:
        dx = (lm[P.L_ANKLE][0] - lm[P.R_ANKLE][0]) * aspect
    return Sample(t=t, ok=True, knee_deg=knee, hip_h=hip_h, x=x, height=height, ankle_dx=dx)


def track_samples(ts, frames, aspect: float, mp: Mobility = PARAMS.mobility) -> list:
    """landmarks -> PresenceTracker -> Samples. ``frames[i]`` is 33 rows or None; dropouts bridged by the tracker come out
    as ``held`` samples (not ok)."""
    tracker = PresenceTracker()
    out = []
    for t, lm in zip(ts, frames):
        o, held = tracker.update(t, lm, aspect)
        if o is None or held:
            out.append(Sample(t=t, held=held))
        else:
            out.append(sample_from_landmarks(t, o.tolist(), aspect, mp))   # plain Python floats from here on
    return out


# ------------------------------------------------------------------ helpers

def _median(v):
    s = sorted(v)
    m = len(s) >> 1
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _pct(v, q: float) -> float:
    """Linear-interpolated percentile (numpy default), q in [0, 1]."""
    s = sorted(v)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def smoothed_velocity(ts, xs, win_s: float) -> list:
    """Per-sample velocity: displacement over the samples within +-win_s/2 divided by their time span (0 if one sample)."""
    n = len(ts)
    out = []
    for i in range(n):
        lo = i
        while lo > 0 and ts[lo - 1] >= ts[i] - win_s / 2:
            lo -= 1
        hi = i
        while hi < n - 1 and ts[hi + 1] <= ts[i] + win_s / 2:
            hi += 1
        out.append((xs[hi] - xs[lo]) / (ts[hi] - ts[lo]) if hi > lo and ts[hi] > ts[lo] else 0.0)
    return out


def _result(test: str, reported: bool, reason: str, **kw) -> dict:
    return {"test": test, "reported": reported, "reason": reason, **kw}


def _quality(window) -> float:
    return sum(1 for s in window if s.ok) / len(window) if window else 0.0


# ------------------------------------------------------------------ sit / stand state machine

class RepMachine:
    def __init__(self, mp: Mobility = PARAMS.mobility):
        self.mp = mp
        self.state = "init"
        self.start_seated = False
        self.stands: list = []        # stand time per STANDING reached (first frame of the held posture)
        self.rise_onsets: list = []   # rise onset of the rise that reached each STANDING
        self.seats: list = []         # seat time per completed rep
        self.sit_onsets: list = []    # last upright frame before each completed rep's sit
        self.partial_rises = 0
        self._rise_t = 0.0
        self._cand_stand: Optional[float] = None
        self._cand_sit: Optional[float] = None
        self._last_stand = 0.0
        self._armed = False

    def update(self, s: Sample) -> None:
        if not s.ok:
            return
        mp, t = self.mp, s.t
        is_sit = s.knee_deg < mp.sit_deg
        is_stand = s.knee_deg >= mp.stand_deg and s.hip_h >= mp.stand_hip_frac
        if self.state == "init":
            self.start_seated = is_sit
            self.state = "seated" if is_sit else "standing"
            self._last_stand = t
            return
        if self.state == "seated":
            if s.knee_deg > mp.rise_deg:
                self.state, self._rise_t, self._armed = "rising", t, True
                self._cand_stand = self._cand_sit = None
        elif self.state == "rising":
            if is_stand:
                if self._cand_stand is None:
                    self._cand_stand = t
                if t - self._cand_stand >= mp.stand_dwell_s:
                    self.state = "standing"
                    self.stands.append(self._cand_stand)
                    self.rise_onsets.append(self._rise_t)
                    self._last_stand, self._cand_sit = t, None
                return
            self._cand_stand = None
            if is_sit:
                if self._cand_sit is None:
                    self._cand_sit = t
                if t - self._cand_sit >= mp.sit_dwell_s:
                    self.state, self._cand_sit = "seated", None
                    self.partial_rises += 1
            else:
                self._cand_sit = None
        elif self.state == "standing":
            if is_stand:
                self._last_stand = t
            if is_sit:
                if self._cand_sit is None:
                    self._cand_sit = t
                if t - self._cand_sit >= mp.sit_dwell_s:
                    self.state = "seated"
                    if self._armed:
                        self.seats.append(self._cand_sit)
                        self.sit_onsets.append(self._last_stand)
                    self._armed, self._cand_sit = False, None
            else:
                self._cand_sit = None

    @property
    def in_progress(self) -> bool:
        return self.state in ("rising", "standing") and self._armed


# ------------------------------------------------------------------ the tests

def chair_stand_30s(samples, go_t: float, mp: Mobility = PARAMS.mobility) -> dict:
    """Completed sit->stand->sit cycles in ``chair_stand_s`` seconds from ``go_t``."""
    name = "chair_stand_30s"
    win = [s for s in samples if go_t <= s.t <= go_t + mp.chair_stand_s]
    q = _quality(win)
    if q < mp.min_quality:
        return _result(name, False, "out_of_view", reps=None, rep_times_s=None, partial_rises=None, rep_in_progress=None, quality=q)
    m = RepMachine(mp)
    for s in win:
        m.update(s)
    if not m.start_seated:
        return _result(name, False, "not_seated_at_start", reps=None, rep_times_s=None, partial_rises=None, rep_in_progress=None, quality=q)
    return _result(name, True, "ok", reps=len(m.seats), rep_times_s=[t - go_t for t in m.seats],
                   partial_rises=m.partial_rises, rep_in_progress=m.in_progress, quality=q)


def _clock_start(m: RepMachine, go_t: float) -> float:
    return min(go_t, m.rise_onsets[0]) if m.rise_onsets else go_t


def sit_to_stand_5x(samples, go_t: float, mp: Mobility = PARAMS.mobility) -> dict:
    """Time from the clock start (Go, or an up-to-``sts_early_s`` earlier rise onset) to the ``sts_reps``-th STANDING."""
    name = "sit_to_stand_5x"
    m = RepMachine(mp)
    end_t = None
    for s in samples:
        if s.t < go_t - mp.sts_early_s:
            continue
        m.update(s)
        if len(m.stands) >= mp.sts_reps:
            end_t = m.stands[mp.sts_reps - 1]
            break
    last_t = end_t if end_t is not None else (samples[-1].t if samples else go_t)
    q = _quality([s for s in samples if go_t <= s.t <= last_t])
    nul = dict(time_s=None, rep_times_s=None, stands=len(m.stands), quality=q)
    if q < mp.min_quality:
        return _result(name, False, "out_of_view", **nul)
    if not m.start_seated:
        return _result(name, False, "not_seated_at_start", **nul)
    if end_t is None:
        return _result(name, False, "incomplete", **nul)
    start = _clock_start(m, go_t)
    st = m.stands[:mp.sts_reps]
    reps = [st[0] - start] + [st[i] - st[i - 1] for i in range(1, len(st))]
    return _result(name, True, "ok", time_s=end_t - start, rep_times_s=reps, stands=len(st), quality=q)


PHASES = ("rise", "walk_out", "turn_1", "walk_back", "turn_2", "sit")


def tug(samples, go_t: float, mp: Mobility = PARAMS.mobility) -> dict:
    """Timed Up and Go phases from landmarks (side-on view, walking along the image x axis)."""
    name = "tug"
    tp = mp.tug
    m = RepMachine(mp)
    for s in samples:
        if s.t >= go_t - mp.sts_early_s:
            m.update(s)
    end_t = m.seats[-1] if m.seats else (samples[-1].t if samples else go_t)
    q = _quality([s for s in samples if go_t <= s.t <= end_t])

    def no(reason):
        return _result(name, False, reason, time_s=None, phases_s=None, quality=q)

    if q < mp.min_quality:
        return no("out_of_view")
    if not m.start_seated:
        return no("not_seated_at_start")
    if not m.stands:
        return no("no_rise")
    start = _clock_start(m, go_t)
    use = [s for s in samples if s.ok and s.t >= go_t - mp.sts_early_s]
    x_seat = use[0].x
    # the stand that starts the walk: the last stand at or before the farthest point reached after the first stand
    # (an earlier stand-up that was followed by sitting down again is part of the "rise" phase)
    first = [s for s in use if s.t >= m.stands[0]]
    far = max(range(len(first)), key=lambda i: (abs(first[i].x - x_seat), -i))
    stand_t = max(st for st in m.stands if st <= first[far].t)
    after = [s for s in use if s.t >= stand_t]
    hs = [s.height for s in after if s.height is not None]
    if not hs:
        return no("out_of_view")
    body = _median(hs)
    ts, xs = [s.t for s in after], [s.x for s in after]
    exc = [abs(x - x_seat) for x in xs]
    ext = max(range(len(exc)), key=lambda i: (exc[i], -i))   # first maximum
    if exc[ext] / body < tp.min_excursion_bodies or ext == len(exc) - 1:
        return no("no_turn")
    if exc[ext] - min(exc[ext + 1:]) < tp.min_return_frac * exc[ext]:
        return no("no_turn")
    v = smoothed_velocity(ts, xs, tp.vel_win_s)
    out_speeds = [abs(v[i]) for i in range(ext + 1) if abs(v[i]) >= mp.walk.min_speed_bodies_s * body]
    if not out_speeds:
        return no("no_turn")
    thr = tp.turn_speed_frac * _median(out_speeds)
    s0 = ext
    while s0 > 0 and abs(v[s0 - 1]) < thr:
        s0 -= 1
    e0 = ext
    while e0 < len(v) - 1 and abs(v[e0 + 1]) < thr:
        e0 += 1
    # the sit that finishes the test: first completed sit after the turn
    k = next((i for i, st in enumerate(m.seats) if st > ts[e0]), None)
    if k is None:
        return no("not_seated_at_end")
    seated_t, sit_onset = m.seats[k], m.sit_onsets[k]
    arrival = sit_onset
    for i in range(e0 + 1, len(v)):
        if ts[i] > sit_onset:
            break
        if abs(v[i]) < thr and abs(v[i - 1]) >= thr:
            arrival = ts[i]
            break
    a = stand_t
    b = min(max(ts[s0], a), sit_onset)
    c = min(max(ts[e0], b), sit_onset)
    d = min(max(arrival, c), sit_onset)
    cuts = [start, a, b, c, d, sit_onset, seated_t]
    phases = {PHASES[i]: cuts[i + 1] - cuts[i] for i in range(6)}
    return _result(name, True, "ok", time_s=seated_t - start, phases_s=phases, quality=q)


def walking_speed(samples, height_m: float = PARAMS.mobility.walk.height_default_m, mp: Mobility = PARAMS.mobility) -> dict:
    """Approximate walking speed (m/s) and cadence (steps/min) over steady SIDE-ON walking segments.

    Walking = hip-centre speed >= min_speed_bodies_s body heights/s (body height = nose-to-ankle extent), split at
    dropouts > gap_s and direction changes. Per segment the steady run (|v| >= steady_frac * median |v|, ends trimmed) must
    keep the apparent body height within +-height_tol_frac (5th/95th percentile vs median) or the walk is not side-on.
    Metres per frame-height unit = nose_ankle_frac * height_m / median apparent height (documented: nose-to-ankle of a
    standing person is about 0.87 of their height). Speed = |least-squares slope of x| * that scale, accepted for runs of at
    least min_dur_s seconds or min_dist_m metres; overall speed = total distance / total time. Quality = fraction of frames
    with visible legs between the first and last visible frame (time before/after the person is in view does not count). Cadence = foot-passing
    count (sign changes of the left-minus-right ankle x with a hysteresis of cadence_hyst_frac of its amplitude)."""
    name, w = "walking_speed", mp.walk
    nul = dict(speed_m_s=None, cadence_spm=None, distance_m=None, duration_s=None, n_segments=0, n_not_side_on=0, segments=[])
    oks = [i for i, s in enumerate(samples) if s.ok]
    q = _quality(samples[oks[0]:oks[-1] + 1]) if oks else 0.0   # quality over the in-view span only (first to last ok frame)
    if not w.height_min_m <= height_m <= w.height_max_m:
        return _result(name, False, "bad_height", quality=q, **nul)
    use = [s for s in samples if s.ok]
    hs_all = [s.height for s in use if s.height is not None]
    if q < mp.min_quality or len(use) < w.min_samples or not hs_all:
        return _result(name, False, "out_of_view", quality=q, **nul)
    hmed = _median(hs_all)
    ts, xs = [s.t for s in use], [s.x for s in use]
    v = smoothed_velocity(ts, xs, w.vel_win_s)
    thr = w.min_speed_bodies_s * hmed
    segs, cur = [], []
    for i in range(len(use)):
        walking = abs(v[i]) >= thr
        if cur and (not walking or ts[i] - ts[cur[-1]] > w.gap_s or (v[i] > 0) != (v[cur[-1]] > 0)):
            segs.append(cur)
            cur = []
        if walking:
            cur.append(i)
    if cur:
        segs.append(cur)
    out, n_cand, n_not_side, tot_d, tot_t = [], 0, 0, 0.0, 0.0
    cad_n, cad_t = 0, 0.0
    for seg in segs:
        if len(seg) < w.min_samples:
            continue
        vm = _median([abs(v[i]) for i in seg])
        steady = [i for i in seg if abs(v[i]) >= w.steady_frac * vm]
        run = [i for i in seg if steady[0] <= i <= steady[-1]]
        if len(run) < w.min_samples:
            continue
        n_cand += 1
        hr = [use[i].height for i in run if use[i].height is not None]
        if not hr:
            continue
        hm = _median(hr)
        if _pct(hr, 0.05) < (1 - w.height_tol_frac) * hm or _pct(hr, 0.95) > (1 + w.height_tol_frac) * hm:
            n_not_side += 1
            continue
        tr, xr = [ts[i] for i in run], [xs[i] for i in run]
        n = len(run)
        mt, mx = sum(tr) / n, sum(xr) / n
        sxx = sum((t - mt) * (t - mt) for t in tr)
        slope = sum((t - mt) * (x - mx) for t, x in zip(tr, xr)) / sxx if sxx > 0 else 0.0
        dur = tr[-1] - tr[0]
        scale = w.nose_ankle_frac * height_m / hm
        speed = abs(slope) * scale
        dist = speed * dur
        if dur < w.min_dur_s and dist < w.min_dist_m:
            continue
        out.append({"t_start": tr[0], "duration_s": dur, "distance_m": dist, "speed_m_s": speed})
        tot_d += dist
        tot_t += dur
        dd = [(use[i].t, use[i].ankle_dx) for i in run if use[i].ankle_dx is not None]
        if len(dd) >= 8:
            mean = sum(d for _, d in dd) / len(dd)
            amp = _pct([abs(d - mean) for _, d in dd], 0.9)
            hy, state, cross = w.cadence_hyst_frac * amp, 0, []
            for t, d in dd:
                dv = d - mean
                if dv > hy:
                    if state == -1:
                        cross.append(t)
                    state = 1
                elif dv < -hy:
                    if state == 1:
                        cross.append(t)
                    state = -1
            if len(cross) >= w.cadence_min_crossings and cross[-1] > cross[0]:
                cad_n += len(cross) - 1
                cad_t += cross[-1] - cross[0]
    base = dict(n_segments=len(out), n_not_side_on=n_not_side, segments=out)
    if not out:
        if n_not_side:
            reason = "not_side_on"
        elif n_cand:
            reason = "too_short"
        else:
            hp5, hp95 = _pct(hs_all, 0.05), _pct(hs_all, 0.95)
            reason = "not_side_on" if (hp5 < (1 - w.height_tol_frac) * hmed or hp95 > (1 + w.height_tol_frac) * hmed) else "no_walk"
        return _result(name, False, reason, speed_m_s=None, cadence_spm=None, distance_m=None, duration_s=None, quality=q, **base)
    return _result(name, True, "ok", speed_m_s=tot_d / tot_t, cadence_spm=(cad_n / cad_t * 60.0 if cad_t > 0 else None),
                   distance_m=tot_d, duration_s=tot_t, quality=q, **base)


# ------------------------------------------------------------------ fixture runner (parity)

def run_mobility(inp: dict) -> dict:
    """Parity runner: input = {test, fps, aspect, landmark_indices, landmarks[], go_t, height_m?, t?}."""
    from .fixtures import _landmarks, _round   # same expansion/rounding as the other fixtures (lazy: fixtures imports this module)
    aspect, idx = inp["aspect"], inp["landmark_indices"]
    ts = inp["t"] if "t" in inp else [i / inp["fps"] for i in range(len(inp["landmarks"]))]
    frames = [_landmarks(f, idx) if f is not None else None for f in inp["landmarks"]]
    samples = track_samples(ts, frames, aspect)
    test = inp["test"]
    if test == "chair_stand_30s":
        return _round(chair_stand_30s(samples, inp["go_t"]))
    if test == "sit_to_stand_5x":
        return _round(sit_to_stand_5x(samples, inp["go_t"]))
    if test == "tug":
        return _round(tug(samples, inp["go_t"]))
    if test == "walking_speed":
        return _round(walking_speed(samples, inp.get("height_m", PARAMS.mobility.walk.height_default_m)))
    raise ValueError(f"unknown mobility test {test!r}")
