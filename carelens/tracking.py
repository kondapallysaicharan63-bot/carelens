"""Presence debouncing and short-term single-person tracking for pose dropouts.

Sits between the PoseProvider and FeatureExtractor. It fills brief dropouts (marked
``held``), debounces presence, and smooths keypoints with a One Euro filter
(Casiez, Roussel, Vogel, CHI 2012).

Kept as its own module because it is a distinct pipeline stage (the PRD's
"Perception (pose, track)" box), not part of pose inference or feature extraction.

Single-person heuristic only: a large jump of the body centroid starts a new track.
This is NOT identity tracking -- no face recognition, by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import pose as P

CENTROID_KPS = [P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP]


@dataclass
class TrackingConfig:
    hold_s: float = 0.5          # max dropout that is bridged with the last pose
    enter_s: float = 0.2         # consecutive-detection span needed to become present (>= 2 frames
                                 # at low fps; a lone detection never creates presence)
    max_jump: float = 0.25       # displacement (frame-height units) treated as a new track / raw passthrough
    min_cutoff: float = 1.0      # One Euro: Hz, smoothing of slow/static motion
    beta: float = 1.0            # One Euro: speed coefficient (coords are in frame units/s)
    d_cutoff: float = 1.0        # One Euro: Hz, derivative smoothing


class OneEuroFilter:
    """Vectorised One Euro filter over a numpy array, time-based (uses sample times)."""

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._t = None
        self._x = None
        self._dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._t is None:
            self._t, self._x, self._dx = t, x.copy(), np.zeros_like(x)
            return x.copy()
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x.copy()


class PresenceTracker:
    """update(t, lm, aspect) -> (landmarks or None, held). `held` frames repeat the last
    smoothed pose and are NOT new measurements; callers must not treat them as such.

    Jumps never cause absence: a large move between consecutive detections (e.g. a fall
    at low fps) passes raw landmarks through and re-seeds the smoother, so the event
    detectors see the full-speed motion. Only a gap longer than hold_s drops presence."""

    def __init__(self, cfg: Optional[TrackingConfig] = None):
        self.cfg = cfg or TrackingConfig()
        self._filt = OneEuroFilter(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)
        self.present = False
        self._first_t: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_out: Optional[np.ndarray] = None
        self._core: Optional[np.ndarray] = None

    def _reset(self):
        self._filt.reset()
        self.present = False
        self._first_t = None
        self._last_out = None
        self._core = None

    def skip_gap(self, gap_s: float):
        """Camera unobserved for `gap_s`: reset the smoothing filters but keep presence state,
        as if no time had passed (no re-entry debounce). If the person really left, the normal
        hold_s -> absent path applies on the following frames."""
        self._filt.reset()
        self._core = None
        if self._last_t is not None:
            self._last_t += gap_s
        if self._first_t is not None:
            self._first_t += gap_s

    def update(self, t: float, lm: Optional[np.ndarray], aspect: float = 1.0):
        c = self.cfg
        if lm is None:
            if self.present and self._last_t is not None and t - self._last_t <= c.hold_s:
                return self._last_out.copy(), True
            self._reset()
            return None, False

        lm = np.asarray(lm, dtype=np.float64)
        core = lm[CENTROID_KPS, :2] * (aspect, 1.0)   # aspect-corrected, frame-height units
        dt = c.enter_s
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > c.hold_s:
                self._reset()
        if self._core is not None and np.linalg.norm(core - self._core, axis=1).max() > c.max_jump:
            self._filt.reset()   # fast move or re-acquired elsewhere: no smoothing, keep presence
        self._core, self._last_t = core, t
        if self._first_t is None:
            self._first_t = t

        out = lm.copy()
        out[:, :2] = self._filt(t, lm[:, :2])   # visibility passes through unsmoothed
        out = out.astype(np.float32)
        self._last_out = out
        if not self.present:
            if (t - self._first_t) + dt < c.enter_s - 1e-6 or t == self._first_t:
                return None, False
            self.present = True
        return out.copy(), False
