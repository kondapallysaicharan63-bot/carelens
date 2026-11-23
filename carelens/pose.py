"""Pose providers. Each returns a (33, 3) array of normalized (x, y, visibility)
in MediaPipe Pose landmark order, or None when no person is found."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

# MediaPipe Pose landmark indices used in this project
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 25, 26, 27, 28
N_LANDMARKS = 33

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"


class PoseProvider(Protocol):
    def process(self, image_bgr: np.ndarray, t: float) -> Optional[np.ndarray]: ...


class MediaPipePose:
    """MediaPipe Pose Landmarker (Apache-2.0) in VIDEO mode."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, image_bgr, t):
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ms = max(int(t * 1000), self._last_ms + 1)  # timestamps must increase
        self._last_ms = ms
        res = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        return np.array([[p.x, p.y, p.visibility if p.visibility is not None else 1.0]
                         for p in lm], dtype=np.float32)

    def close(self):
        self._landmarker.close()


class ScriptedPose:
    """Returns landmarks from a function of time. Used with synthetic video so
    the event and vitals logic can be tested without filming anyone."""

    def __init__(self, fn: Callable[[float], Optional[np.ndarray]]):
        self.fn = fn

    def process(self, image_bgr, t):
        return self.fn(t)

    def close(self):
        pass
