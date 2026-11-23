"""Fall detection, inactivity alerts, activity timeline and gait metrics.

Wellness wording only: these produce trends and "check-in suggested" events,
never a diagnosis.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from .features import BodyState


@dataclass
class Event:
    kind: str            # "fall" | "inactivity"
    t_onset: float       # seconds into the clip
    t_confirmed: float
    message: str
    details: dict = field(default_factory=dict)
    clip: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class FallDetector:
    """Sudden drop in torso angle or hip height, then little movement for
    `still_seconds`. Standing back up cancels a pending fall."""

    def __init__(self, drop_window_s=0.9, upright_deg=40, lying_deg=60, drop_frac=0.35,
                 min_angle_rate=40.0, still_seconds=20.0, still_motion=0.08,
                 pending_timeout_s=120.0):
        self.drop_window_s = drop_window_s
        self.upright_deg = upright_deg
        self.lying_deg = lying_deg
        self.drop_frac = drop_frac
        self.min_angle_rate = min_angle_rate   # deg/s: separates a fall from a slow lie-down
        self.still_seconds = still_seconds
        self.still_motion = still_motion
        self.pending_timeout_s = pending_timeout_s
        self._hist: deque = deque()
        self._pending: Optional[dict] = None
        self._armed = True   # re-armed once the person is upright again

    def _sudden_drop(self, s: BodyState) -> Optional[dict]:
        for (t0, ang0, hip0, size0) in self._hist:
            if ang0 > self.upright_deg:
                continue
            dt = max(s.t - t0, 1e-3)
            if s.torso_angle > self.lying_deg and (s.torso_angle - ang0) / dt >= self.min_angle_rate:
                return {"trigger": "torso_angle", "from_deg": round(ang0, 1),
                        "to_deg": round(s.torso_angle, 1), "drop_time_s": round(dt, 2),
                        "angle_rate_deg_s": round((s.torso_angle - ang0) / dt, 1)}
            if (s.hip_y - hip0) > self.drop_frac * size0 and (s.hip_y - hip0) / size0 / dt >= 0.3:
                return {"trigger": "hip_drop", "drop_body_frac": round((s.hip_y - hip0) / size0, 2),
                        "drop_time_s": round(dt, 2)}
        return None

    def update(self, s: BodyState, held: bool = False) -> Optional[Event]:
        """held=True: `s` is a carried-forward state (pose dropout bridged by the tracker).
        It is not a measurement, so it never enters the drop history or triggers a drop;
        its carried motion still feeds the stillness timer."""
        if s.present and not held:
            self._hist.append((s.t, s.torso_angle, s.hip_y, s.extent))
        while self._hist and self._hist[0][0] < s.t - self.drop_window_s:
            self._hist.popleft()

        if (s.present and not held and s.torso_angle < self.upright_deg
                and s.posture in ("standing", "walking")):
            if self._pending is not None:
                self._pending = None   # recovered
            self._armed = True

        if self._pending is None:
            if self._armed and s.present and not held:
                info = self._sudden_drop(s)
                if info:
                    self._pending = {"t_onset": s.t, "still_start": s.t, "info": info}
                    self._armed = False
            return None

        p = self._pending
        if s.t - p["t_onset"] > self.pending_timeout_s:
            self._pending = None
            return None
        if s.present and s.motion > self.still_motion:
            p["still_start"] = s.t        # moving: restart stillness timer
        if s.t - p["still_start"] >= self.still_seconds:
            self._pending = None
            return Event(kind="fall", t_onset=round(p["t_onset"], 2), t_confirmed=round(s.t, 2),
                         message="Possible fall: sudden drop followed by little movement. "
                                 "Check-in suggested.",
                         details={**p["info"], "still_for_s": round(s.t - p["still_start"], 1)})
        return None


class InactivityDetector:
    """No meaningful movement for `seconds` while the person is present."""

    def __init__(self, seconds=1800.0, still_motion=0.08):
        self.seconds = seconds
        self.still_motion = still_motion
        self._start: Optional[float] = None
        self._fired = False

    def update(self, s: BodyState) -> Optional[Event]:
        if not s.present or s.motion > self.still_motion:
            self._start, self._fired = None, False
            return None
        if self._start is None:
            self._start = s.t
        if not self._fired and s.t - self._start >= self.seconds:
            self._fired = True
            return Event(kind="inactivity", t_onset=round(self._start, 2), t_confirmed=round(s.t, 2),
                         message=f"No meaningful movement for {self.seconds/60:.1f} min while "
                                 f"present ({s.posture}). Check-in suggested.",
                         details={"posture": s.posture})
        return None


class ActivityTimeline:
    """Seconds spent in each posture, bucketed by wall-clock hour."""

    POSTURES = ("sitting", "standing", "walking", "lying", "absent")

    def __init__(self, start_time: datetime):
        self.start_time = start_time
        self.buckets: dict = defaultdict(lambda: defaultdict(float))
        self._last_t: Optional[float] = None

    def update(self, s: BodyState):
        if self._last_t is not None:
            dt = min(s.t - self._last_t, 1.0)
            hour = (self.start_time + timedelta(seconds=s.t)).strftime("%Y-%m-%d %H:00")
            self.buckets[hour][s.posture] += dt
        self._last_t = s.t

    def to_dict(self):
        return {h: {p: round(v.get(p, 0.0), 1) for p in self.POSTURES}
                for h, v in sorted(self.buckets.items())}


class GaitAnalyzer:
    """Walking speed and steadiness for each crossing of the room.

    Speed is converted to m/s using the person's height, so it is approximate and
    assumes a roughly side-on view. Use it as a trend over days, not an absolute.
    """

    def __init__(self, person_height_m=1.65, min_walk_s=2.0, gap_s=0.5):
        self.height_m = person_height_m
        self.min_walk_s = min_walk_s
        self.gap_s = gap_s
        self.segments: list = []
        self._cur: list = []
        self._last_walk_t: Optional[float] = None

    def update(self, s: BodyState):
        if s.present and s.posture == "walking":
            self._cur.append(s)
            self._last_walk_t = s.t
        elif self._cur and (self._last_walk_t is None or s.t - self._last_walk_t > self.gap_s):
            self._close()

    def _close(self):
        seg, self._cur = self._cur, []
        if len(seg) < 3 or seg[-1].t - seg[0].t < self.min_walk_s:
            return
        dur = seg[-1].t - seg[0].t
        size = float(np.median([b.extent for b in seg]))
        dist = abs(seg[-1].center[0] - seg[0].center[0]) / size        # body heights
        speeds = np.array([b.speed_x for b in seg])
        angles = np.array([b.torso_angle for b in seg])
        self.segments.append({
            "t_start": round(seg[0].t, 2), "duration_s": round(dur, 2),
            "speed_m_s_approx": round(dist / dur * self.height_m, 2),
            "speed_cv": round(float(speeds.std() / (speeds.mean() + 1e-6)), 3),
            "trunk_sway_deg": round(float(angles.std()), 2),
        })

    def finish(self):
        if self._cur:
            self._close()
        return self.segments
