"""Per-frame body features from pose landmarks: height, torso angle, posture, motion."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import pose as P

CORE = [P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP]


@dataclass
class BodyState:
    t: float
    present: bool
    height: float = 0.0          # bbox height of visible landmarks (fraction of frame)
    extent: float = 0.0          # max(bbox w, h): body size that survives lying down
    torso_angle: float = 0.0     # degrees from vertical (0 upright, 90 horizontal)
    hip_y: float = 0.0           # hip midpoint y (0 top .. 1 bottom)
    center: tuple = (0.0, 0.0)   # hip midpoint (x, y)
    motion: float = 0.0          # landmark speed, body-sizes per second (smoothed)
    speed_x: float = 0.0         # horizontal hip speed, body-sizes per second
    posture: str = "absent"      # standing | sitting | lying | walking | absent
    landmarks: Optional[np.ndarray] = None


def _mid(lm, a, b):
    return (lm[a, :2] + lm[b, :2]) / 2.0


class FeatureExtractor:
    def __init__(self, min_visibility: float = 0.5, walk_speed: float = 0.25,
                 smooth_s: float = 1.0, aspect: float = 4 / 3):
        # aspect = frame width / height. x is rescaled so both axes are in
        # "frame-height units"; pixel coords are then coord * frame_height.
        self.aspect = aspect
        self.min_vis = min_visibility
        self.walk_speed = walk_speed
        self.smooth_s = smooth_s
        self._prev: Optional[tuple[float, np.ndarray]] = None
        self._motion_hist: deque = deque()
        self._x_hist: deque = deque()

    def update(self, t: float, lm: Optional[np.ndarray]) -> BodyState:
        if lm is None or lm[CORE, 2].mean() < self.min_vis:
            self._prev = None
            self._x_hist.clear()
            return BodyState(t=t, present=False)
        lm = lm.copy()
        lm[:, 0] *= self.aspect

        vis = lm[:, 2] >= self.min_vis
        pts = lm[vis, :2]
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        extent = max(w, h, 1e-3)

        sh = _mid(lm, P.L_SHOULDER, P.R_SHOULDER)
        hip = _mid(lm, P.L_HIP, P.R_HIP)
        v = sh - hip
        torso_len = float(np.linalg.norm(v)) + 1e-6
        angle = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]))))

        # Motion: mean landmark displacement per second, normalised by body size
        inst = 0.0
        if self._prev is not None:
            pt, plm = self._prev
            dt = max(t - pt, 1e-3)
            both = vis & (plm[:, 2] >= self.min_vis)
            if both.any():
                d = np.linalg.norm(lm[both, :2] - plm[both, :2], axis=1).mean()
                inst = float(d / extent / dt)
        self._prev = (t, lm.copy())
        self._motion_hist.append((t, inst))
        while self._motion_hist and self._motion_hist[0][0] < t - self.smooth_s:
            self._motion_hist.popleft()
        motion = float(np.mean([m for _, m in self._motion_hist]))

        self._x_hist.append((t, float(hip[0])))
        while self._x_hist and self._x_hist[0][0] < t - self.smooth_s:
            self._x_hist.popleft()
        speed_x = 0.0
        if len(self._x_hist) >= 2 and self._x_hist[-1][0] > self._x_hist[0][0]:
            (t0, x0), (t1, x1) = self._x_hist[0], self._x_hist[-1]
            speed_x = abs(x1 - x0) / (t1 - t0) / extent

        # Posture
        if angle > 60:
            posture = "lying"
        else:
            ank = _mid(lm, P.L_ANKLE, P.R_ANKLE)
            ank_ok = lm[[P.L_ANKLE, P.R_ANKLE], 2].mean() >= self.min_vis
            leg_ratio = (ank[1] - hip[1]) / torso_len if ank_ok else 1.6
            posture = "standing" if leg_ratio > 1.2 else "sitting"
            if posture == "standing" and speed_x > self.walk_speed:
                posture = "walking"

        return BodyState(t=t, present=True, height=h, extent=extent, torso_angle=angle,
                         hip_y=float(hip[1]), center=(float(hip[0]), float(hip[1])),
                         motion=motion, speed_x=speed_x, posture=posture, landmarks=lm)
