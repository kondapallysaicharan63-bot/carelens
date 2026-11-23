"""Video input layer: one interface for files, USB webcams and streams.

The Roku camera is treated as just another file source: export a saved clip
from the app as a normal customer would, then pass the file path here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Union

import cv2
import numpy as np


def jetson_csi_pipeline(sensor_id: int = 0, width: int = 1280, height: int = 720,
                        fps: int = 30, out_width: int = 640, out_height: int = 360) -> str:
    """GStreamer pipeline for a CSI camera on NVIDIA Jetson (needs JetPack's OpenCV
    built with GStreamer)."""
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw,width={out_width},height={out_height},format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2")


@dataclass
class Frame:
    t: float          # seconds since start of source
    image: np.ndarray  # BGR uint8
    index: int


class VideoSource:
    """Iterate frames from a file path, a webcam index, or a stream URL.

    Examples:
        VideoSource("clip.mp4")      # exported clip
        VideoSource(0)               # first USB webcam
        VideoSource("rtsp://...")    # only if the device officially offers one
    """

    def __init__(self, source: Union[str, int], max_seconds: float | None = None,
                 resize_width: int | None = 640):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.is_live = isinstance(source, int) or (
            isinstance(source, str) and source.split(":", 1)[0] in ("rtsp", "http", "https", "csi"))
        self.max_seconds = max_seconds
        self.resize_width = resize_width
        if isinstance(source, str) and source.startswith("csi://"):
            # Jetson CSI camera (CAM0/CAM1) through GStreamer, e.g. "csi://0"
            self._cap = cv2.VideoCapture(jetson_csi_pipeline(int(source[6:] or 0)),
                                         cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video source: {source!r}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 15.0

    def __iter__(self) -> Iterator[Frame]:
        start = time.monotonic()
        idx = 0
        try:
            while True:
                ok, img = self._cap.read()
                if not ok:
                    break
                if self.is_live:
                    t = time.monotonic() - start
                else:
                    # Frame index / fps is more reliable than POS_MSEC across codecs.
                    t = idx / self.fps
                if self.max_seconds is not None and t > self.max_seconds:
                    break
                if self.resize_width and img.shape[1] > self.resize_width:
                    h = int(img.shape[0] * self.resize_width / img.shape[1])
                    img = cv2.resize(img, (self.resize_width, h), interpolation=cv2.INTER_AREA)
                yield Frame(t=t, image=img, index=idx)
                idx += 1
        finally:
            self._cap.release()
