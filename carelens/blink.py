"""Blink rate, eyes-closed time and long closures from MediaPipe FaceLandmarker eye points (W9, Experimental).

Python reference; ``web/src/core/blink.ts`` mirrors it operation for operation and both read ``shared/params.json -> blink``.
Not a medical device. It reports how often the eyes closed briefly (blinks per minute), what share of the last minute the eyes
were (almost) shut ("eyes-closed time", the PERCLOS measure) and how many longer closures there were. Screens, reading, dry air,
light, contact lenses and glasses glare all change these numbers: compare with your own earlier results, nothing more.

Per frame (landmark indices in ``params.blink.eye_landmarks`` are the MediaPipe FaceLandmarker eye contour in the usual
p1..p6 order: p1 outer corner, p2 / p3 the upper lid, p4 the inner corner, p5 / p6 the lower lid, so p2-p6 and p3-p5 are the
two vertical lid distances):

  EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)   per eye, in aspect-corrected coordinates (x * w/h, y);
  the frame's EAR is the mean of both eyes (both must be measurable).

Calibration is per person: the open-eye baseline is the ``baseline_pct`` (90th) percentile of the EAR over the last
``baseline_s`` (20 s) of VALID frames (frames without a face, with a turned head or at too low a rate are not valid and neither
enter nor age the baseline). Until ``baseline_s`` of valid data exist the reason is ``calibrating`` and nothing is counted.

  closed        EAR < ``closed_frac`` x baseline (0.55); the eye is open again once EAR > ``reopen_frac`` x baseline (0.70)
                (hysteresis, so jitter around the threshold does not split one blink into two)
  fully closed  EAR < ``perclos_frac`` x baseline (0.30), i.e. roughly 70 % or more lid closure; time spent so is PERCLOS
  blink         a closed run from its first closed frame to the first reopened frame lasting ``blink_min_ms``..``blink_max_ms``
                (50-500 ms); shorter runs are noise, longer ones are "long closures" (counted once, when they pass
                ``blink_max_ms``), never blinks
  window        everything is evaluated over the last ``rate_window_s`` (60 s) of counted time: blinks per minute = blinks /
                counted seconds x 60, eyes-closed % = fully-closed seconds / counted seconds, long closures = their count.
                Valid time only: a head turn or lost face does not dilute the rate. Withheld (``calibrating``) until
                ``min_window_s`` of counted time exist.

Gating (first match wins): ``no_face`` (no landmarks, an eye too small or implausible), ``low_fps`` (fewer than
``min_rate_hz`` DISTINCT frames per second over the last ``fps_window_s``: at 15 Hz a 100 ms blink spans 1-2 frames, so a
count would be a guess), ``head_turned`` (yaw or pitch estimated from the landmarks beyond the limits: EAR is unreliable),
``calibrating``, else ``ok``. Only ``ok`` is ``reported``.

Head pose (rough, from landmarks only, not an angle): yaw ratio = horizontal offset of the nose tip from the midpoint of the
outer eye corners / the distance between those corners (about 0.45 * sin(yaw) for a typical face, so the default 0.25 is a
turn of roughly 30-35 degrees); pitch ratio = (nose y - forehead y) / (chin y - forehead y), about 0.57 when facing the
camera. Frame gaps above ``max_dt_s`` are unobserved time: they are not counted and end any open closure.
A closure already under way when counting starts or resumes (end of calibration, face back after a turn or gap) has an
unknown start and is skipped. Known limits: not validated on real faces or eyewear; synthetic EAR traces in tests; 25-30 Hz cameras resolve a 100 ms blink in
2-3 frames, so durations are quantised to the frame period.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence

import numpy as np

from .params import PARAMS, Blink

REASONS = ("ok", "no_face", "low_fps", "head_turned", "calibrating")
LEFT_EYE = PARAMS.blink.eye_landmarks.left
RIGHT_EYE = PARAMS.blink.eye_landmarks.right
POSE = PARAMS.blink.pose_landmarks
EYE_CORNERS = (LEFT_EYE[0], RIGHT_EYE[3])          # outer corners: 33 and 263
FIXTURE_INDICES = sorted(set(LEFT_EYE + RIGHT_EYE + [POSE.nose, POSE.forehead, POSE.chin]))


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy's default), written out so the TypeScript port matches exactly."""
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _dist(a, b, aspect: float) -> float:
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def eye_ear(lm, idx: Sequence[int], aspect: float, bp: Blink = PARAMS.blink) -> Optional[float]:
    """EAR of one eye, or None when the eye is too small or the value is implausible."""
    p = [lm[i] for i in idx]
    width = _dist(p[0], p[3], aspect)
    if not width >= bp.min_eye_width:
        return None
    ear = (_dist(p[1], p[5], aspect) + _dist(p[2], p[4], aspect)) / (2.0 * width)
    return ear if 0.0 <= ear <= bp.max_ear else None


def frame_ear(lm, aspect: float, bp: Blink = PARAMS.blink) -> Optional[float]:
    left = eye_ear(lm, bp.eye_landmarks.left, aspect, bp)
    right = eye_ear(lm, bp.eye_landmarks.right, aspect, bp)
    if left is None or right is None:
        return None
    return 0.5 * (left + right)


def head_turned(lm, aspect: float, bp: Blink = PARAMS.blink) -> bool:
    a, b = lm[bp.eye_landmarks.left[0]], lm[bp.eye_landmarks.right[3]]
    iod = _dist(a, b, aspect)
    nose, fore, chin = lm[bp.pose_landmarks.nose], lm[bp.pose_landmarks.forehead], lm[bp.pose_landmarks.chin]
    height = chin[1] - fore[1]
    if not iod > 0 or not height > 0:
        return True
    yaw = (nose[0] - 0.5 * (a[0] + b[0])) * aspect / iod
    pitch = (nose[1] - fore[1]) / height
    return abs(yaw) > bp.yaw_max_ratio or abs(pitch - bp.pitch_neutral) > bp.pitch_tol


class BlinkTracker:
    """Streaming tracker: ``update(t, lm, aspect)`` per DISTINCT camera frame (``lm`` None = no face), returns the state dict."""

    def __init__(self, bp: Blink = PARAMS.blink):
        self.bp = bp
        self.last_t: Optional[float] = None
        self.times: deque = deque()           # recent frame times for the distinct-frame rate
        self.base: deque = deque()            # (dt, ear) of the valid frames in the baseline window
        self.base_sum = 0.0
        self.valid_total = 0.0
        self.closed = False
        self.need_open = True                 # a closure already under way when counting (re)starts has an unknown start: skip it
        self.run_start = 0.0
        self.run_long = False
        self.long_ev: Optional[dict] = None
        self.events: list = []                # {"t_start", "t_end", "long"} for every blink / long closure
        self.window: deque = deque()          # (t, dt, deep) of counted frames in the rate window
        self.first_counted: Optional[float] = None
        self.last = self._state("calibrating", None, None, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    # ---------------------------------------------------------------- helpers
    def _state(self, reason, ear, base, bpm, perclos, long_n, win_s, cov, rate) -> dict:
        return {"reason": reason, "reported": reason == "ok", "ear": ear, "baseline": base, "closed": self.closed,
                "blinks_per_min": bpm, "perclos": perclos, "long_closures": long_n, "window_s": win_s,
                "coverage": cov, "fps": rate, "n_blinks": sum(1 for e in self.events if not e["long"])}

    def _distinct_rate(self, t: float, gap: bool) -> Optional[float]:
        bp = self.bp
        if gap:
            self.times.clear()
        self.times.append(t)
        while self.times and self.times[0] < t - bp.fps_window_s:
            self.times.popleft()
        span = self.times[-1] - self.times[0]
        if len(self.times) < 3 or span < 0.5 * bp.fps_window_s:
            return None                       # not enough history yet to tell
        return (len(self.times) - 1) / span

    def _end_run(self, t: float, reopened: bool) -> None:
        bp = self.bp
        if self.long_ev is not None:
            self.long_ev["t_end"] = t
            self.long_ev = None
        if self.closed and reopened:
            dur_ms = (t - self.run_start) * 1000.0
            if bp.blink_min_ms <= dur_ms <= bp.blink_max_ms:
                self.events.append({"t_start": self.run_start, "t_end": t, "long": False})
            elif dur_ms > bp.blink_max_ms and not self.run_long:
                self.events.append({"t_start": self.run_start, "t_end": t, "long": True})
        self.closed = False
        self.run_long = False
        if not reopened:
            self.need_open = True

    # ---------------------------------------------------------------- per frame
    def update(self, t: float, lm, aspect: float) -> dict:
        bp = self.bp
        dt = 0.0
        gap = False
        if self.last_t is not None:
            raw = t - self.last_t
            gap = raw > bp.max_dt_s
            dt = 0.0 if gap else max(raw, 0.0)
        self.last_t = t
        if gap:
            self._end_run(t, False)
        rate = self._distinct_rate(t, gap)

        ear = frame_ear(lm, aspect, bp) if lm is not None else None
        if ear is None:
            self._end_run(t, False)
            self.last = self._finish("no_face", None, rate)
            return self.last
        if rate is None:                      # rate not yet verified (start, or after a frame gap): never count at an unverified rate
            self._end_run(t, False)
            self.last = self._finish("calibrating", ear, rate)
            return self.last
        if rate < bp.min_rate_hz:
            self._end_run(t, False)
            self.last = self._finish("low_fps", ear, rate)
            return self.last
        if head_turned(lm, aspect, bp):
            self._end_run(t, False)
            self.last = self._finish("head_turned", ear, rate)
            return self.last

        # valid frame: baseline over the last baseline_s of valid frames
        self.base.append((dt, ear))
        self.base_sum += dt
        while len(self.base) > 1 and self.base_sum - self.base[0][0] >= bp.baseline_s:
            self.base_sum -= self.base.popleft()[0]
        self.valid_total += dt
        calibrated = self.valid_total >= bp.baseline_s
        base = percentile([e for _, e in self.base], bp.baseline_pct)
        if not calibrated or not base > 0:
            self.need_open = True
            self.last = self._finish("calibrating", ear, rate, base if base > 0 else None)
            return self.last

        if self.first_counted is None:
            self.first_counted = t
        deep = ear < bp.perclos_frac * base
        self.window.append((t, dt, deep))
        if self.need_open:
            if ear >= bp.closed_frac * base:
                self.need_open = False
        elif not self.closed:
            if ear < bp.closed_frac * base:
                self.closed, self.run_start, self.run_long = True, t, False
        elif ear > bp.reopen_frac * base:
            self._end_run(t, True)
        elif not self.run_long and (t - self.run_start) * 1000.0 > bp.blink_max_ms:
            self.run_long = True
            self.long_ev = {"t_start": self.run_start, "t_end": t, "long": True}
            self.events.append(self.long_ev)
        self.last = self._finish("ok", ear, rate, base)
        return self.last

    def _finish(self, reason: str, ear, rate, base=None) -> dict:
        bp = self.bp
        t = self.last_t
        while self.window and self.window[0][0] < t - bp.rate_window_s:
            self.window.popleft()
        counted = sum(d for _, d, _ in self.window)
        deep_s = sum(d for _, d, k in self.window if k)
        lo = t - bp.rate_window_s
        n_blink = sum(1 for e in self.events if not e["long"] and e["t_start"] >= lo)
        n_long = sum(1 for e in self.events if e["long"] and e["t_start"] >= lo)
        span = min(bp.rate_window_s, t - self.first_counted) if self.first_counted is not None else 0.0
        cov = min(1.0, counted / span) if span > 0 else 0.0
        if reason == "ok" and counted < bp.min_window_s:
            reason = "calibrating"
        bpm = n_blink / counted * 60.0 if counted > 0 else 0.0
        perclos = deep_s / counted if counted > 0 else 0.0
        return self._state(reason, ear, base, bpm, perclos, n_long, counted, cov, rate if rate is not None else 0.0)


def run_blink(inp: dict) -> dict:
    """Parity runner: input = {w, h, fps, [t], landmark_indices, landmarks[]} (frames: rows [x, y] per listed index, or null)."""
    idx = inp["landmark_indices"]
    aspect = inp["w"] / inp["h"]
    tr = BlinkTracker()
    runs, updates, last_sec = [], [], None
    for i, frame in enumerate(inp["landmarks"]):
        t = inp["t"][i] if "t" in inp else i / inp["fps"]
        lm = None if frame is None else {k: row for k, row in zip(idx, frame)}
        u = tr.update(t, lm, aspect)
        if not runs or runs[-1]["reason"] != u["reason"]:
            runs.append({"t": round(t, 3), "reason": u["reason"]})
        if last_sec != math.floor(t):
            last_sec = math.floor(t)
            updates.append({"t": round(t, 3), "reason": u["reason"], "blinks_per_min": u["blinks_per_min"], "perclos": u["perclos"],
                            "long_closures": u["long_closures"], "ear": u["ear"], "baseline": u["baseline"]})
    return {"blinks": [{"t_start": e["t_start"], "t_end": e["t_end"]} for e in tr.events if not e["long"]],
            "long_closures": [{"t_start": e["t_start"], "t_end": e["t_end"]} for e in tr.events if e["long"]],
            "reason_runs": runs, "updates_1hz": updates, "final": {k: v for k, v in tr.last.items() if k != "closed"}}


# ------------------------------------------------------------------ synthetic faces (tests, fixtures)

# A frontal face in frame-height units around (0.5 * aspect, 0.5): eye contours as the 6 p1..p6 points of an eye of width EYE_W
EYE_W = 0.085
EYE_CENTRES = {"left": (-0.075, -0.02), "right": (0.075, -0.02)}   # relative to the face centre, x in frame-height units
NOSE, FOREHEAD, CHIN = (0.0, 0.05), (0.0, -0.20), (0.0, 0.25)       # pitch ratio (0.05 + 0.20) / 0.45 = 0.556


def ear_trace(duration: float, fps: float, blinks_per_min: float = 0.0, blink_ms: float = 150.0, open_ear: float = 0.30, seed: int = 0,
              closures: Sequence[tuple] = (), blink_ms_range: Optional[tuple] = None, ramp_ms: float = 45.0,
              drift: float = 0.03, t_jitter_s: float = 0.0):
    """A synthetic EAR trace with KNOWN eye closures. Returns (t, ear, truth) where truth lists each closure's
    (t_start, t_end) as the span below 50 % of the open EAR.

    Blinks are placed at a regular rate (0 = none) with a random +-25 % interval jitter, each lasting ``blink_ms`` (or a duration
    drawn from ``blink_ms_range``) between its two half-open crossings; ``closures`` are extra (t0, t1) long closures (eyes shut
    from t0 to t1). The lids move linearly over ``ramp_ms`` to a closed EAR of 8 % of the open one and back. ``drift`` is a slow
    +-fraction wobble of the open EAR (posture, squinting), ``t_jitter_s`` perturbs the frame times."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * fps))
    t = np.arange(n) / fps
    if t_jitter_s:
        t = np.maximum.accumulate(np.maximum(0.0, t + rng.uniform(-t_jitter_s, t_jitter_s, n)))
    spans = []
    if blinks_per_min > 0:
        period = 60.0 / blinks_per_min
        x = period * rng.uniform(0.3, 0.9)
        while x < duration - 1.0:
            ms = rng.uniform(*blink_ms_range) if blink_ms_range else blink_ms
            spans.append((x, x + ms / 1000.0))
            x += period * rng.uniform(0.75, 1.25)
    spans += [tuple(c) for c in closures]
    spans.sort()
    ramp = ramp_ms / 1000.0
    openness = np.ones(n)                       # 1 = open, closed level = 0.08
    for a, b in spans:
        # half-way crossings at a and b: the ramp is centred on them
        down = np.clip((t - (a - ramp / 2)) / ramp, 0.0, 1.0)
        up = np.clip((t - (b - ramp / 2)) / ramp, 0.0, 1.0)
        openness = np.minimum(openness, 1.0 - 0.92 * (down - up))
    wob = 1.0 + drift * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6.28))
    return t, open_ear * wob * openness, spans


def eye_points(cx: float, cy: float, ear: float, w: float = EYE_W) -> list:
    """The 6 contour points p1..p6 of one eye (frame-height units) for a given EAR; lids symmetric about the eye centre."""
    h = ear * w                                 # |p2-p6| = |p3-p5| = EAR * |p1-p4|
    return [(cx - w / 2, cy), (cx - w / 6, cy - h / 2), (cx + w / 6, cy - h / 2), (cx + w / 2, cy), (cx + w / 6, cy + h / 2), (cx - w / 6, cy + h / 2)]


def synth_face(duration: float = 80.0, fps: float = 25.0, aspect: float = 4 / 3, jitter_px: float = 0.0, seed: int = 0,
               open_ear: float = 0.30, yaw_segments: Sequence[tuple] = (), lost: Sequence[tuple] = (), h_px: int = 480, **ear_kw) -> tuple:
    """A synthetic face as the parity input dict plus the truth spans. ``jitter_px``: independent Gaussian landmark noise in
    pixels of a ``h_px``-high frame (MediaPipe-like landmark jitter is about 0.3-0.5 px at a steady face).
    ``yaw_segments``: (t0, t1, ratio) spans where the face is turned (nose shifted by ``ratio`` x the eye distance; the eyes
    keep closing and opening, so a wrong count there would show). ``lost``: spans without a face."""
    rng = np.random.default_rng(seed + 1000)
    t, ear, spans = ear_trace(duration, fps, open_ear=open_ear, seed=seed, **ear_kw)
    n = t.size
    centre = np.array([0.5 * aspect, 0.5])
    iod = 2 * EYE_CENTRES["right"][0]
    sigma = jitter_px / h_px
    frames = []
    for i in range(n):
        if any(a <= t[i] < b for a, b in lost):
            frames.append(None)
            continue
        pts = {}
        for side, idx in (("left", PARAMS.blink.eye_landmarks.left), ("right", PARAMS.blink.eye_landmarks.right)):
            ex, ey = EYE_CENTRES[side]
            for k, (x, y) in zip(idx, eye_points(centre[0] + ex, centre[1] + ey, float(ear[i]))):
                pts[k] = [x, y]
        nose = np.array(NOSE)
        for a, b, ratio in yaw_segments:
            if a <= t[i] < b:
                nose = nose + np.array([ratio * iod, 0.0])
        pts[POSE.nose] = list(centre + nose)
        pts[POSE.forehead] = list(centre + np.array(FOREHEAD))
        pts[POSE.chin] = list(centre + np.array(CHIN))
        rows = []
        for k in FIXTURE_INDICES:
            x, y = pts[k]
            x, y = x + rng.normal(0, sigma), y + rng.normal(0, sigma)
            rows.append([round(x / aspect, 4), round(y, 4)])        # normalised by frame width / height
        frames.append(rows)
    w = int(round(h_px * aspect))
    inp = {"w": w, "h": h_px, "fps": fps, "landmark_indices": FIXTURE_INDICES, "landmarks": frames}
    if ear_kw.get("t_jitter_s"):
        inp["t"] = [round(float(x), 4) for x in t]
    return inp, spans
