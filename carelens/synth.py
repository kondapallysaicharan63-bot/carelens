"""Synthetic test video with known ground truth: no real person is filmed.

A simple figure walks, sits, falls and lies still. Its face patch carries a
pulsatile skin colour (POS skin-model vector, strongest in green) at a known heart rate, and its shirt texture moves up and
down at a known breathing rate. Matching pose landmarks are returned through
ScriptedPose, because MediaPipe cannot see a stick figure. That means these tests
check the event and signal-processing logic. Pose accuracy has to be checked on
real, consented clips.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import pose as P
from .pose import ScriptedPose

W, H = 640, 480
ASPECT = W / H


def _skeleton(posture: str, x: float, phase: float = 0.0) -> np.ndarray:
    """33x3 landmarks in frame-height units (x, y, visibility)."""
    lm = np.zeros((P.N_LANDMARKS, 3), np.float32)
    lm[:, 2] = 0.2   # landmarks we do not model are low-visibility
    pts = {}
    if posture in ("stand", "walk"):
        swing = 0.05 * np.sin(phase) if posture == "walk" else 0.0
        pts = {P.NOSE: (x, 0.25), P.L_EYE: (x + 0.02, 0.235), P.R_EYE: (x - 0.02, 0.235),
               P.L_EAR: (x + 0.04, 0.24), P.R_EAR: (x - 0.04, 0.24),
               P.L_SHOULDER: (x + 0.07, 0.33), P.R_SHOULDER: (x - 0.07, 0.33),
               P.L_HIP: (x + 0.05, 0.55), P.R_HIP: (x - 0.05, 0.55),
               P.L_KNEE: (x + 0.05 + swing / 2, 0.72), P.R_KNEE: (x - 0.05 - swing / 2, 0.72),
               P.L_ANKLE: (x + 0.05 + swing, 0.90), P.R_ANKLE: (x - 0.05 - swing, 0.90)}
    elif posture == "sit":
        pts = {P.NOSE: (x, 0.32), P.L_EYE: (x + 0.02, 0.305), P.R_EYE: (x - 0.02, 0.305),
               P.L_EAR: (x + 0.04, 0.31), P.R_EAR: (x - 0.04, 0.31),
               P.L_SHOULDER: (x + 0.07, 0.40), P.R_SHOULDER: (x - 0.07, 0.40),
               P.L_HIP: (x + 0.05, 0.62), P.R_HIP: (x - 0.05, 0.62),
               P.L_KNEE: (x + 0.07, 0.64), P.R_KNEE: (x - 0.07, 0.64),
               P.L_ANKLE: (x + 0.07, 0.86), P.R_ANKLE: (x - 0.07, 0.86)}
    elif posture == "lie":
        y = 0.84
        pts = {P.NOSE: (x + 0.30, y - 0.02), P.L_EYE: (x + 0.29, y - 0.035),
               P.R_EYE: (x + 0.31, y - 0.035), P.L_EAR: (x + 0.28, y - 0.01),
               P.R_EAR: (x + 0.32, y - 0.01),
               P.L_SHOULDER: (x + 0.22, y - 0.03), P.R_SHOULDER: (x + 0.22, y + 0.03),
               P.L_HIP: (x, y - 0.03), P.R_HIP: (x, y + 0.03),
               P.L_KNEE: (x - 0.17, y - 0.03), P.R_KNEE: (x - 0.17, y + 0.03),
               P.L_ANKLE: (x - 0.34, y - 0.03), P.R_ANKLE: (x - 0.34, y + 0.03)}
    for k, (px, py) in pts.items():
        lm[k] = (px, py, 0.95)
    return lm


@dataclass
class Segment:
    kind: str                 # walk | still | transition | absent
    dur: float
    posture: str = "stand"    # for still
    x0: float = 0.6
    x1: float = 0.6           # for walk
    to: str = "stand"         # for transition (from = posture)


class Scenario:
    def __init__(self, segments: list[Segment], hr_bpm=72.0, br_bpm=15.0, jitter=0.0008,
                 seed=0):
        self.segments = segments
        self.hr_hz, self.br_hz = hr_bpm / 60.0, br_bpm / 60.0
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self.duration = sum(s.dur for s in segments)
        self.truth = {"hr_bpm": hr_bpm, "br_bpm": br_bpm, "segments": []}
        t = 0.0
        for s in segments:
            self.truth["segments"].append({"t0": round(t, 2), "t1": round(t + s.dur, 2),
                                           **s.__dict__})
            t += s.dur

    def landmarks_hu(self, t: float) -> Optional[np.ndarray]:
        t0 = 0.0
        for s in self.segments:
            if t < t0 + s.dur or s is self.segments[-1]:
                u = np.clip((t - t0) / s.dur, 0, 1)
                if s.kind == "absent":
                    return None
                if s.kind == "walk":
                    x = s.x0 + (s.x1 - s.x0) * u
                    return _skeleton("walk", x, phase=2 * np.pi * 1.8 * (t - t0))
                if s.kind == "still":
                    return _skeleton(s.posture, s.x0)
                if s.kind == "transition":
                    a, b = _skeleton(s.posture, s.x0), _skeleton(s.to, s.x0)
                    return a + (b - a) * u
            t0 += s.dur
        return None

    def pose_landmarks(self, t: float) -> Optional[np.ndarray]:
        """Normalized landmarks (x / width, y / height) as MediaPipe would return."""
        lm = self.landmarks_hu(t)
        if lm is None:
            return None
        lm = lm.copy()
        lm[:, :2] += self.rng.normal(0, self.jitter, (P.N_LANDMARKS, 2))
        lm[:, 0] /= ASPECT
        return lm

    def provider(self) -> ScriptedPose:
        return ScriptedPose(self.pose_landmarks)

    @classmethod
    def from_truth(cls, path: str | Path) -> "Scenario":
        """Rebuild a scenario from the ``.truth.json`` written next to a rendered clip."""
        truth = json.loads(Path(path).read_text())
        fields = Segment.__dataclass_fields__
        segs = [Segment(**{k: v for k, v in seg.items() if k in fields}) for seg in truth["segments"]]
        return cls(segs, hr_bpm=truth["hr_bpm"], br_bpm=truth["br_bpm"])


# ---------------------------------------------------------------- side-on mobility scripts (W7)
# A side-on figure for the chair-stand / TUG / walking tests (synthetic only). Thigh = shank = LEG; the ankles are fixed on
# the ground line while the hip rises from the chair (knee 90 deg) to standing (knee ~169 deg) by inverse kinematics.
# Nose-to-ankle is 0.652 frame heights at scale 1, i.e. 0.87 of a 1.65 m person: 1 frame-height unit = M_PER_U / scale metres.

LEG, GROUND, TORSO = 0.20, 0.90, 0.217
M_PER_U = 0.87 * 1.65 / 0.652


def _prog_geometry(prog: float):
    dy = 0.20 + 0.198 * prog
    dx = 0.20 * (1.0 - prog)
    return dx, dy


def side_knee_deg(prog: float) -> float:
    """Knee angle of the side-on figure at sit->stand progress ``prog`` (0 seated .. 1 standing, both legs straight down)."""
    dx, dy = _prog_geometry(prog)
    return float(np.degrees(2 * np.arcsin(min(1.0, np.hypot(dx, dy) / (2 * LEG)))))


def _side_skeleton(prog: float, fx: float, facing: int = 1, scale: float = 1.0, gait: float = 0.0, stride: float = 0.0,
                   turn: float = 0.0) -> np.ndarray:
    """33x3 landmarks (frame-height units) of the side-on figure. ``fx`` = ankle-centre x, ``gait`` stride phase (rad) with
    ankle swing ``stride`` (frame heights), ``turn`` 0..1 rotates the shoulders towards the camera."""
    lm = np.zeros((P.N_LANDMARKS, 3), np.float32)
    lm[:, 2] = 0.2
    dx, dy = _prog_geometry(prog)
    hip = np.array([-dx, -dy])
    sw = np.sin(gait)
    pts = {}
    for side, (hk, kk, ak) in {1: (P.L_HIP, P.L_KNEE, P.L_ANKLE), -1: (P.R_HIP, P.R_KNEE, P.R_ANKLE)}.items():
        ank = np.array([side * stride * sw, 0.0])
        v = ank - hip
        d = float(np.hypot(*v))
        mid = (hip + ank) / 2
        hgt = np.sqrt(max(LEG ** 2 - (d / 2) ** 2, 0.0))
        knee = mid + np.array([v[1], -v[0]]) / d * hgt
        pts[hk] = (hip[0] + side * 0.008, hip[1])
        pts[kk] = tuple(knee)
        pts[ak] = tuple(ank)
    lean = np.radians(18.0 * (1.0 - prog))
    sh = hip + np.array([np.sin(lean) * TORSO, -np.cos(lean) * TORSO])
    half = (0.012 + 0.07 * turn) / 2
    pts[P.L_SHOULDER] = (sh[0] + half, sh[1])
    pts[P.R_SHOULDER] = (sh[0] - half, sh[1])
    pts[P.NOSE] = (sh[0] + 0.03, sh[1] - 0.037)
    for k, (px, py) in pts.items():
        lm[k] = (fx + facing * px * scale, GROUND + py * scale, 0.95)
    return lm


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _interp_knots(t: float, knots) -> float:
    """Value at t of (time, value) knots, smoothstep between successive knots."""
    if t <= knots[0][0]:
        return knots[0][1]
    for (ta, va), (tb, vb) in zip(knots, knots[1:]):
        if t < tb:
            return va + (vb - va) * float(smoothstep((t - ta) / (tb - ta)))
    return knots[-1][1]


class SideScript:
    """Time functions of the side-on figure: ``prog(t)``, ``x(t)`` (ankle-centre), ``facing(t)``, ``turn(t)``, ``speed(t)`` (m/s,
    drives the stride), plus an optional ``scale(t)``. ``frames`` samples them with landmark jitter and dropouts."""

    def __init__(self, duration, prog, x, facing=lambda t: 1, turn=lambda t: 0.0, speed=lambda t: 0.0,
                 scale=lambda t: 0.6, stride_hz=0.9, jitter=0.0008, seed=0):
        self.duration, self.prog, self.x, self.facing, self.turn = duration, prog, x, facing, turn
        self.speed, self.scale, self.stride_hz, self.jitter = speed, scale, stride_hz, jitter
        self.rng = np.random.default_rng(seed)
        self._gait_t = np.linspace(0, duration, int(duration * 200) + 1)
        rate = np.array([min(1.0, abs(speed(t)) / 0.3) for t in self._gait_t])
        self._gait = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(self._gait_t))]) * 2 * np.pi * stride_hz

    def landmarks_hu(self, t: float) -> np.ndarray:
        sc = self.scale(t)
        amp = 0.05 * min(1.0, abs(self.speed(t)) / 0.8)
        gait = float(np.interp(t, self._gait_t, self._gait))
        return _side_skeleton(self.prog(t), self.x(t), self.facing(t), sc, gait, amp, self.turn(t))

    def frames(self, fps: float = 15.0, dropouts=()):
        """(times, normalised landmarks or None) for a camera of aspect ASPECT, like ``Scenario.pose_landmarks``."""
        ts, out = [], []
        for i in range(int(self.duration * fps)):
            t = i / fps
            lm = self.landmarks_hu(t)
            lm[:, :2] += self.rng.normal(0, self.jitter, (P.N_LANDMARKS, 2))
            lm[:, 0] /= ASPECT
            ts.append(t)
            out.append(None if any(a <= t < b for a, b in dropouts) else lm)
        return ts, out


def _chair_knots(n_reps: int, partial_after=(), t0: float = 2.0, cycle: float = 2.0, rise: float = 0.5, hold: float = 0.35,
                 descend: float = 0.5, peak_partial: float = 0.9, first_delay: float = 0.0):
    """prog knots: seated until t0 (+first_delay); then ``n_reps`` full cycles (rise, hold standing, descend, seated for the
    rest of ``cycle``) with a partial rise (to ``peak_partial``, no stand) after each rep index in ``partial_after``."""
    knots, t = [(0.0, 0.0), (t0 + first_delay, 0.0)], t0 + first_delay
    rest = cycle - rise - hold - descend
    for r in range(n_reps):
        for peak, h in ((1.0, hold),) + (((peak_partial, 0.0),) if r in partial_after else ()):
            knots += [(t + rise, peak), (t + rise + h, peak), (t + rise + h + descend, 0.0)]
            t += rise + h + descend
            knots += [(t + rest, 0.0)]
            t += rest
    return knots, t


def chair_stand_script(n_reps: int, partial_after=(), cycle: float = 2.0, t0: float = 2.0, duration: float = 36.0, **kw) -> tuple[SideScript, list]:
    """Chair stand: seated lead-in until ``t0`` ('Go'), then ``n_reps`` full reps (plus partial rises after the listed rep
    indices), seated afterwards. The figure faces +x, ankles at x = 0.55."""
    knots, _ = _chair_knots(n_reps, partial_after, t0=t0, cycle=cycle, **{k: v for k, v in kw.items() if k in ("rise", "hold", "descend", "peak_partial", "first_delay")})
    return SideScript(duration, lambda t: _interp_knots(t, knots), lambda t: 0.55,
                      jitter=kw.get("jitter", 0.0008), seed=kw.get("seed", 0)), knots


def _speed_profile(segs, x0: float, m_per_u: float):
    """Piecewise-linear speed knots [(t, v m/s)] -> (speed(t), x(t)) with x integrated analytically (x in frame heights)."""
    ts = [s[0] for s in segs]
    vs = [s[1] for s in segs]
    xs = [x0]
    for i in range(1, len(ts)):
        xs.append(xs[-1] + 0.5 * (vs[i] + vs[i - 1]) * (ts[i] - ts[i - 1]) / m_per_u)

    def speed(t):
        return float(np.interp(t, ts, vs))

    def x(t):
        if t <= ts[0]:
            return x0
        if t >= ts[-1]:
            return xs[-1]
        i = int(np.searchsorted(ts, t, side="right")) - 1
        dt = t - ts[i]
        slope = (vs[i + 1] - vs[i]) / (ts[i + 1] - ts[i])
        return xs[i] + (vs[i] * dt + 0.5 * slope * dt * dt) / m_per_u
    return speed, x


def tug_script(distance_m: float = 3.0, speed: float = 1.0, t0: float = 2.0, reaction: float = 0.5, rise: float = 1.2,
               pivot: float = 1.0, descend: float = 1.2, ramp: float = 0.5, tail: float = 2.0, scale: float = 0.6,
               return_trip: bool = True, sit_at_end: bool = True, jitter: float = 0.0008, seed: int = 0):
    """TUG: seated until ``t0`` (Go), rise, walk ``distance_m`` out, pivot, walk back, pivot, sit. Returns (script, events)
    where events holds the scripted times: go, rise_start, rise_end, turn1 (pivot window), turn2, sit_start, sit_end.
    ``return_trip`` False: no walk back (sits down at the far end). ``sit_at_end`` False: stays standing at the chair."""
    m_per_u = M_PER_U / scale
    tr = t0 + reaction
    t_rise_end = tr + rise
    walk_t = distance_m / speed + ramp
    seg = [(0.0, 0.0), (t_rise_end, 0.0), (t_rise_end + ramp, speed), (t_rise_end + walk_t - ramp, speed), (t_rise_end + walk_t, 0.0)]
    t_p1 = t_rise_end + walk_t
    ev = {"go": t0, "rise_start": tr, "rise_end": t_rise_end}
    t_end_walk = t_p1
    if return_trip:
        t_p1e = t_p1 + pivot
        seg += [(t_p1e, 0.0), (t_p1e + ramp, -speed), (t_p1e + walk_t - ramp, -speed), (t_p1e + walk_t, 0.0)]
        ev["turn1"] = (t_p1, t_p1e)
        t_p2 = t_p1e + walk_t
        t_p2e = t_p2 + pivot
        ev["turn2"] = (t_p2, t_p2e)
        t_end_walk = t_p2e
    else:
        t_p2e = t_p1 + 0.5
    ev["sit_start"] = t_p2e
    ev["sit_end"] = t_p2e + descend
    duration = t_p2e + descend + tail
    seg.append((duration, 0.0))
    spd, x = _speed_profile(seg, 0.30, m_per_u)
    knots = [(0.0, 0.0), (tr, 0.0), (t_rise_end, 1.0), (t_end_walk, 1.0)]
    if sit_at_end:
        knots += [(t_p2e, 1.0), (t_p2e + descend, 0.0), (duration, 0.0)]
    else:
        knots += [(duration, 1.0)]
    t_flip = t_p1 + pivot / 2

    def facing(t):
        return 1 if (not return_trip or t < t_flip) else -1

    def turn(t):
        for a, b in (ev.get("turn1"), ev.get("turn2")) if return_trip else ((t_p1, t_p1 + 0.5),):
            if a <= t < b:
                return float(np.sin(np.pi * (t - a) / (b - a)))
        return 0.0
    return SideScript(duration, lambda t: _interp_knots(t, knots), x, facing, turn, spd, scale=lambda t: scale,
                      jitter=jitter, seed=seed), ev, knots


def walk_script(speed: float, seconds: float = 6.0, scale: float = 0.6, x0: float = 0.1, lead: float = 0.0, jitter: float = 0.0008, seed: int = 0,
                tail: float = 0.0) -> SideScript:
    """Side-on walk to the right at constant ``speed`` m/s (after a standing ``lead``, before a standing ``tail``)."""
    m_per_u = M_PER_U / scale
    spd, x = _speed_profile([(0.0, 0.0), (lead, 0.0), (lead + 1e-3, speed), (lead + seconds, speed), (lead + seconds + 1e-3, 0.0), (lead + seconds + tail + 1e-3, 0.0)], x0, m_per_u)
    return SideScript(lead + seconds + tail, lambda t: 1.0, x, speed=spd, scale=lambda t: scale, jitter=jitter, seed=seed)


def approach_script(seconds: float = 5.0, drift_u: float = 0.0, s0: float = 0.45, s1: float = 0.9, jitter: float = 0.0008, seed: int = 0) -> SideScript:
    """Walks towards the camera: apparent size grows s0 -> s1 while the hip drifts ``drift_u`` frame heights sideways."""
    return SideScript(seconds, lambda t: 1.0, lambda t: 0.45 + drift_u * t / seconds, speed=lambda t: 1.0,
                      scale=lambda t: s0 + (s1 - s0) * t / seconds, jitter=jitter, seed=seed)


def _shirt_texture(rng, size=220):
    tex = np.full((size, size, 3), (150, 90, 60), np.uint8)
    for y in range(0, size, 9):
        cv2.line(tex, (0, y), (size, y), (200, 140, 100), 3)
    noise = rng.normal(0, 18, (size, size, 1))
    return np.clip(tex + noise, 0, 255).astype(np.uint8)


def render(scn: Scenario, path: str | Path, fps: float = 15.0, br_amp_px: float = 1.6,
           hr_amp: float = 0.8, light: float = 1.0, seed: int = 1, noise_sd: float = 4.0) -> Path:
    rng = np.random.default_rng(seed)
    path = Path(path)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    bg = np.full((H, W, 3), (95, 105, 110), np.uint8)
    cv2.rectangle(bg, (0, int(0.93 * H)), (W, H), (70, 80, 90), -1)
    shirt = _shirt_texture(rng)
    n = int(scn.duration * fps)
    for i in range(n):
        t = i / fps
        img = bg.copy()
        lm = scn.landmarks_hu(t)
        if lm is not None:
            px = lambda k: (int(lm[k, 0] * H), int(lm[k, 1] * H))
            # torso with moving shirt texture (breathing)
            sh = (lm[P.L_SHOULDER, :2] + lm[P.R_SHOULDER, :2]) / 2 * H
            hp = (lm[P.L_HIP, :2] + lm[P.R_HIP, :2]) / 2 * H
            length = np.linalg.norm(sh - hp)
            ang = np.degrees(np.arctan2(sh[0] - hp[0], -(sh[1] - hp[1])))
            dy = br_amp_px * np.sin(2 * np.pi * scn.br_hz * t)
            c = (sh + hp) / 2
            M = cv2.getRotationMatrix2D((110, 110), -ang, 1.0)
            M[:, 2] += (c[0] - 110, c[1] - 110 + dy)
            mask = np.zeros((220, 220), np.uint8)
            cv2.rectangle(mask, (110 - 45, int(110 - length / 2) - 5),
                          (110 + 45, int(110 + length / 2) + 5), 255, -1)
            Mm = M.copy(); Mm[1, 2] -= dy   # body outline does not move, only the shirt
            warped = cv2.warpAffine(shirt, M, (W, H))
            m = cv2.warpAffine(mask, Mm, (W, H)) > 0
            img[m] = warped[m]
            for a, b in [(P.L_HIP, P.L_KNEE), (P.L_KNEE, P.L_ANKLE), (P.R_HIP, P.R_KNEE),
                         (P.R_KNEE, P.R_ANKLE)]:
                cv2.line(img, px(a), px(b), (60, 60, 140), 14, cv2.LINE_AA)
            # face: skin ellipse whose colour pulses at the heart rate
            nose = np.array(px(P.NOSE)); eyes = (np.array(px(P.L_EYE)) + np.array(px(P.R_EYE))) / 2
            fc = tuple(((nose + eyes) / 2).astype(int))
            pulse = hr_amp * np.sin(2 * np.pi * scn.hr_hz * t)
            # BGR skin colour; the pulsatile colour vector is roughly proportional to the POS-paper skin
            # model [0.33, 0.77, 0.53] (relative R, G, B modulation), blood-volume negative-going. The green
            # amplitude (hr_amp levels) is unchanged, so the green-only estimator sees the same signal.
            rel = pulse / 150.0
            skin = (110.0 * (1.0 - rel * 0.53 / 0.77), 150.0 * (1.0 - rel), 205.0 * (1.0 - rel * 0.33 / 0.77))
            cv2.ellipse(img, fc, (30, 38), 0, 0, 360, skin, -1, cv2.LINE_AA)
            # eyes sit just above the landmark so the pose-derived face ROI (which starts just below the eye
            # line) holds skin only; otherwise one pixel of ROI-edge jitter swaps dark eye pixels in and out
            for k in (P.L_EYE, P.R_EYE):
                cv2.circle(img, (px(k)[0], px(k)[1] - 4), 2, (40, 40, 40), -1)
        f = img.astype(np.float32) * light * (1 + 0.01 * np.sin(2 * np.pi * 0.03 * t))
        f += rng.normal(0, noise_sd, f.shape)   # per-pixel sensor noise (levels)
        vw.write(np.clip(f, 0, 255).astype(np.uint8))
    vw.release()
    return path


def demo_scenario(inactivity_s: float = 60.0, **kw) -> Scenario:
    """Walk in, sit still (vitals), walk, sudden fall, lie still, get up,
    sit very still (inactivity), leave the room."""
    return Scenario([
        Segment("walk", 6, x0=0.15, x1=0.65),
        Segment("transition", 1.5, posture="stand", to="sit", x0=0.65),
        Segment("still", 40, posture="sit", x0=0.65),
        Segment("transition", 1.5, posture="sit", to="stand", x0=0.65),
        Segment("walk", 4, x0=0.65, x1=0.85),
        Segment("transition", 0.6, posture="stand", to="lie", x0=0.85),   # sudden fall
        Segment("still", 30, posture="lie", x0=0.85),
        Segment("transition", 3, posture="lie", to="stand", x0=0.85),     # slow get-up
        Segment("walk", 3, x0=0.85, x1=0.6),
        Segment("transition", 1.5, posture="stand", to="sit", x0=0.6),
        Segment("still", inactivity_s + 8, posture="sit", x0=0.6),
        Segment("transition", 1.5, posture="sit", to="stand", x0=0.6),
        Segment("walk", 4, x0=0.6, x1=1.4),
        Segment("absent", 4),
    ], **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render a synthetic demo clip with ground truth")
    ap.add_argument("--out", default="output/synthetic_demo.mp4")
    ap.add_argument("--hr", type=float, default=72.0)
    ap.add_argument("--br", type=float, default=15.0)
    a = ap.parse_args(argv)
    scn = demo_scenario(hr_bpm=a.hr, br_bpm=a.br)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    render(scn, a.out)
    Path(a.out).with_suffix(".truth.json").write_text(json.dumps(scn.truth, indent=2))
    print(f"wrote {a.out} ({scn.duration:.0f}s)")


if __name__ == "__main__":
    main()
