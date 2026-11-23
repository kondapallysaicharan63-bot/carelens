"""Synthetic test video with known ground truth: no real person is filmed.

A simple figure walks, sits, falls and lies still. Its face patch carries a
green-channel pulse at a known heart rate, and its shirt texture moves up and
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


def _shirt_texture(rng, size=220):
    tex = np.full((size, size, 3), (150, 90, 60), np.uint8)
    for y in range(0, size, 9):
        cv2.line(tex, (0, y), (size, y), (200, 140, 100), 3)
    noise = rng.normal(0, 18, (size, size, 1))
    return np.clip(tex + noise, 0, 255).astype(np.uint8)


def render(scn: Scenario, path: str | Path, fps: float = 15.0, br_amp_px: float = 1.6,
           hr_amp: float = 0.8, light: float = 1.0, seed: int = 1) -> Path:
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
            # face: skin ellipse, green channel pulses at the heart rate
            nose = np.array(px(P.NOSE)); eyes = (np.array(px(P.L_EYE)) + np.array(px(P.R_EYE))) / 2
            fc = tuple(((nose + eyes) / 2).astype(int))
            pulse = hr_amp * np.sin(2 * np.pi * scn.hr_hz * t)
            skin = (110.0, 150.0 + pulse, 205.0)
            cv2.ellipse(img, fc, (30, 38), 0, 0, 360, skin, -1, cv2.LINE_AA)
            cv2.circle(img, px(P.L_EYE), 3, (40, 40, 40), -1)
            cv2.circle(img, px(P.R_EYE), 3, (40, 40, 40), -1)
        f = img.astype(np.float32) * light * (1 + 0.01 * np.sin(2 * np.pi * 0.03 * t))
        f += rng.normal(0, 4, f.shape)
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
