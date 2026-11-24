"""Video input layer: one interface for files, USB webcams and streams.

The Roku camera is treated as just another file source: export a saved clip
from the app as a normal customer would, then pass the file path here.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union

import cv2
import numpy as np

from .config import redact_url

log = logging.getLogger(__name__)


def jetson_csi_pipeline(sensor_id: int = 0, width: int = 1280, height: int = 720,
                        fps: int = 30, out_width: int = 640, out_height: int = 360) -> str:
    """GStreamer pipeline for a CSI camera on NVIDIA Jetson (needs JetPack's OpenCV
    built with GStreamer)."""
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw,width={out_width},height={out_height},format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2")


_ENV_LOCK = threading.Lock()
_OPTS_KEY = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


def _merge_ffmpeg_options(prev: str | None, transport: str) -> str:
    """Existing options plus our rtsp_transport (the camera's setting wins over a stale one)."""
    kept = [o for o in (prev or "").split("|") if o and o.split(";", 1)[0] != "rtsp_transport"]
    return "|".join(kept + [f"rtsp_transport;{transport}"])


def _open_rtsp(url: str, transport: str, timeout_s: float) -> "cv2.VideoCapture":
    """Open an RTSP URL with FFmpeg, forcing the transport and bounding open/read time.

    Falls back to OpenCV's default backend when FFmpeg is unavailable (e.g. GStreamer-only builds).
    """
    if transport not in ("tcp", "udp"):
        raise ValueError(f"rtsp_transport must be 'tcp' or 'udp', not {transport!r}")
    ms = max(1, int(timeout_s * 1000))
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms]
    cap, t0 = None, time.monotonic()
    with _ENV_LOCK:
        prev = os.environ.get(_OPTS_KEY)
        os.environ[_OPTS_KEY] = _merge_ffmpeg_options(prev, transport)
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        except Exception:
            cap = None
        finally:
            if prev is None:
                os.environ.pop(_OPTS_KEY, None)
            else:
                os.environ[_OPTS_KEY] = prev
    if cap is not None and cap.isOpened():
        return cap
    if cap is not None:
        cap.release()
    if time.monotonic() - t0 >= 0.8 * timeout_s:
        # FFmpeg genuinely timed out on a reachable-looking host; don't wait twice.
        return cv2.VideoCapture()
    return cv2.VideoCapture(url)


@dataclass
class Frame:
    t: float          # seconds since start of source
    image: np.ndarray  # BGR uint8
    index: int
    # Unobserved seconds immediately before this frame (live outage / reconnect); 0 for
    # contiguous frames. Detectors must not count this time as stillness.
    gap_before_s: float = 0.0


def _open_capture(source, rtsp_transport: str = "tcp", open_timeout_s: float = 10.0):
    """Open a cv2.VideoCapture for a webcam index, CSI pipeline, RTSP URL or file path."""
    if isinstance(source, str) and source.startswith("csi://"):
        # Jetson CSI camera (CAM0/CAM1) through GStreamer, e.g. "csi://0"
        return cv2.VideoCapture(jetson_csi_pipeline(int(source[6:] or 0)), cv2.CAP_GSTREAMER)
    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://")):
        return _open_rtsp(source, rtsp_transport, open_timeout_s)
    return cv2.VideoCapture(source)


class VideoSource:
    """Iterate frames from a file path, a webcam index, or a stream URL.

    Examples:
        VideoSource("clip.mp4")      # exported clip
        VideoSource(0)               # first USB webcam
        VideoSource("rtsp://...")    # only if the device officially offers one
    """

    def __init__(self, source: Union[str, int], max_seconds: float | None = None,
                 resize_width: int | None = 640, rtsp_transport: str = "tcp",
                 open_timeout_s: float = 10.0):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.is_live = isinstance(source, int) or (
            isinstance(source, str)
            and source.split(":", 1)[0].lower() in ("rtsp", "rtsps", "http", "https", "csi"))
        self.max_seconds = max_seconds
        self.resize_width = resize_width
        self._cap = _open_capture(source, rtsp_transport, open_timeout_s)
        if not self._cap.isOpened():
            self._cap.release()
            raise IOError(f"Could not open video source: {redact_url(str(source))}")
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



class LiveSource:
    """Robust live input (usb / rtsp / csi): threaded capture, drop-old buffer, auto-reconnect.

    Same iteration contract as ``VideoSource`` (iterable of ``Frame``; ``.fps``, ``.source``,
    ``.is_live``) but a capture thread owns the ``cv2.VideoCapture`` and pushes into a tiny
    latest-frame buffer; when the consumer (inference) is slower than the camera the OLDEST
    frame is dropped (``frames_dropped``), so latency stays bounded. File clips keep using the
    synchronous ``VideoSource`` (deterministic, never drops frames).

    Frame ``t`` is monotonic seconds since iteration start, stamped at capture time, so
    ``last_latency_s`` (consume time - capture time) is measurable.

    Reconnect: a failed read, a failed open, or no frame for ``stall_s`` releases the capture
    and retries forever (until ``stop()``, ``max_seconds`` or the iterator is closed) with
    exponential backoff (initial 0.5 s, x2, cap 30 s, +-20 % jitter); backoff resets after a
    good frame. The first frame after an outage carries ``gap_before_s`` = unobserved seconds.

    Limitation: the stall check runs in the capture thread after ``read()`` returns, so a
    ``read()`` that blocks forever is only bounded by the backend's own read timeout (RTSP
    sets ``CAP_PROP_READ_TIMEOUT_MSEC`` in ``_open_rtsp``). USB/CSI backends have none; a
    wedged driver cannot be interrupted from Python (the thread is a daemon, so it never blocks
    process exit). Releasing a capture from another thread mid-read is unsafe, so we don't.
    """

    def __init__(self, source: Union[str, int], max_seconds: float | None = None,
                 resize_width: int | None = 640, rtsp_transport: str = "tcp",
                 open_timeout_s: float = 10.0, stall_s: float = 5.0, queue_size: int = 2,
                 backoff_initial_s: float = 0.5, backoff_factor: float = 2.0,
                 backoff_max_s: float = 30.0, jitter: float = 0.2,
                 open_fn: Optional[Callable] = None, sleep_fn: Optional[Callable] = None,
                 clock: Optional[Callable] = None, rng: Optional[random.Random] = None,
                 on_status: Optional[Callable[[dict], None]] = None,
                 strict_open: bool = True, join_timeout_s: float = 3.0,
                 max_outage_s: Optional[float] = None):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self._target = source                                 # may hold credentials: never log
        self.source = redact_url(str(source))
        self.is_live = True
        self.max_seconds, self.resize_width = max_seconds, resize_width
        self.stall_s, self.max_outage_s = stall_s, max_outage_s
        self.gave_up = False
        self._event_hook: Optional[Callable[[dict], None]] = None
        self._outage_t, self._announced, self._ever_up = 0.0, False, False
        self.queue_size = max(1, queue_size)
        self.backoff_initial_s, self.backoff_factor = backoff_initial_s, backoff_factor
        self.backoff_max_s, self.jitter = backoff_max_s, jitter
        self.join_timeout_s = join_timeout_s
        self._open_fn = open_fn or (lambda src: _open_capture(src, rtsp_transport, open_timeout_s))
        self._clock = clock or time.monotonic
        self._stop = threading.Event()
        self._sleep = sleep_fn or (lambda d: self._stop.wait(d))
        self._rng = rng or random.Random()
        self.on_status = on_status
        self.status = "connecting"
        self.reconnects = 0
        self.frames_dropped = 0
        self.last_latency_s = 0.0
        self._buf: deque = deque()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._t0 = self._clock()
        self._last_cap_t = 0.0
        self._outage = False
        self._cap = None
        self.fps = 15.0
        if strict_open:   # same fail-fast behaviour as VideoSource for a bad camera/URL
            cap = self._open_fn(self._target)
            if not cap.isOpened():
                cap.release()
                raise IOError(f"Could not open video source: {self.source}")
            self._cap = cap
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 1 else 15.0

    # ---- status ------------------------------------------------------------------------
    def _set_status(self, status: str):
        if status == self.status:
            return
        self.status = status
        if self.on_status:
            try:
                self.on_status({"status": status, "reconnects": self.reconnects,
                                "frames_dropped": self.frames_dropped,
                                "t": round(self._clock() - self._t0, 2)})
            except Exception:   # a callback must never kill capture
                log.warning("on_status callback failed", exc_info=False)

    def set_event_hook(self, hook: Optional[Callable[[dict], None]]):
        """Outage notifications, called from the capture thread (keep it cheap, thread-safe):
        ``{"type": "outage_started", "t_onset"}`` once an outage has lasted ``stall_s`` (so brief
        blips never alert), and ``{"type": "outage_ended", "t_onset", "duration_s"}`` on
        recovery. Alerting hooks here, so a dead camera is reported at onset, not at recovery."""
        self._event_hook = hook

    def elapsed_s(self) -> float:
        """Seconds since iteration start, on the frame-timestamp clock."""
        return self._clock() - self._t0

    def _emit(self, ev: dict):
        if self._event_hook:
            try:
                self._event_hook(ev)
            except Exception:
                log.warning("event hook failed", exc_info=False)

    # ---- capture thread ----------------------------------------------------------------
    def _backoff_delay(self, base: float) -> float:
        return base * (1.0 + self._rng.uniform(-self.jitter, self.jitter))

    def _capture_loop(self):
        cap, backoff = self._cap, self.backoff_initial_s
        self._cap = None
        got_first = False       # per capture: has it produced a frame yet?
        last_good = self._clock()
        try:
            while not self._stop.is_set():
                if cap is None:
                    try:
                        cap = self._open_fn(self._target)
                        if not cap.isOpened():
                            cap.release()
                            cap = None
                    except Exception as e:   # type only: messages can echo the URL
                        log.warning("camera open failed (%s)", type(e).__name__)
                        cap = None
                    if cap is None:
                        self._fail()
                        backoff = self._wait_backoff(backoff)
                        continue
                    got_first, last_good = False, self._clock()
                try:
                    # The first read on a new capture may legitimately wait for a keyframe
                    # (bounded by the backend's own timeouts); only later reads can "stall".
                    ok, img = cap.read()
                except Exception as e:
                    log.warning("camera read failed (%s)", type(e).__name__)
                    ok, img = False, None
                now = self._clock()
                if not ok or img is None:
                    cap.release()
                    cap = None
                    self._fail()
                    backoff = self._wait_backoff(backoff)
                    continue
                stalled = got_first and now - last_good > self.stall_s
                got_first, last_good, backoff = True, now, self.backoff_initial_s
                self._push(img, now - self._t0, stalled)   # a returned frame is never discarded
                if stalled:   # it arrived, but the stream froze: tear down and reconnect
                    cap.release()
                    cap = None
                    self._fail()
                    backoff = self._wait_backoff(backoff)
        except BaseException as e:
            self._error = e
            log.error("capture thread crashed (%s)", type(e).__name__)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._stop.set()
            with self._cond:
                self._cond.notify_all()

    def _fail(self):
        if not self._outage:
            self._outage = True
            self._outage_t = self._last_cap_t     # outage onset: last frame seen (or start)
            self._announced = False
        self._set_status("down")
        self._check_outage()

    def _check_outage(self):
        lasted = self.elapsed_s() - self._outage_t
        if not self._announced and lasted >= self.stall_s:
            self._announced = True
            self._emit({"type": "outage_started", "t_onset": self._outage_t})
        if self.max_outage_s is not None and lasted >= self.max_outage_s:
            self.gave_up = True
            self._stop.set()

    def _wait_backoff(self, backoff: float) -> float:
        self._sleep(self._backoff_delay(backoff))
        if self._outage:
            self._check_outage()
        return min(backoff * self.backoff_factor, self.backoff_max_s)

    def _push(self, img, t: float, stalled: bool = False):
        gap = 0.0
        if stalled:
            gap = max(0.0, t - self._last_cap_t)
        elif self._outage:
            gap = max(0.0, t - self._last_cap_t)   # before the first frame: from iteration start
            if self._announced:
                self._emit({"type": "outage_ended", "t_onset": self._outage_t, "duration_s": gap})
            self._outage = False
            if self._ever_up:
                self.reconnects += 1
        self._ever_up = True
        self._last_cap_t = t
        self._set_status("up")
        fr = Frame(t=t, image=img, index=-1, gap_before_s=gap)
        with self._cond:
            while len(self._buf) >= self.queue_size:
                old = self._buf.popleft()      # drop OLDEST, keep latest
                self.frames_dropped += 1
                fr.gap_before_s += old.gap_before_s   # a dropped frame must not hide an outage
            self._buf.append(fr)
            self._cond.notify()

    # ---- consumer ----------------------------------------------------------------------
    def __iter__(self) -> Iterator[Frame]:
        if self._thread is not None:
            raise RuntimeError("LiveSource can only be iterated once")
        self._t0 = self._clock()
        self._last_cap_t = 0.0
        self._thread = threading.Thread(target=self._capture_loop, name="carelens-capture",
                                        daemon=True)
        self._thread.start()
        idx = 0
        try:
            while True:
                with self._cond:
                    while not self._buf and not self._stop.is_set():
                        if self.max_seconds is not None and self._clock() - self._t0 > self.max_seconds:
                            return
                        self._cond.wait(0.1)
                    if not self._buf:
                        break
                    fr = self._buf.popleft()
                if self.max_seconds is not None and fr.t > self.max_seconds:
                    return
                self.last_latency_s = self._clock() - (self._t0 + fr.t)
                img = fr.image
                if self.resize_width and img.shape[1] > self.resize_width:
                    h = int(img.shape[0] * self.resize_width / img.shape[1])
                    img = cv2.resize(img, (self.resize_width, h), interpolation=cv2.INTER_AREA)
                fr.image, fr.index = img, idx
                idx += 1
                yield fr
            if self._error is not None:
                raise RuntimeError("live capture thread failed") from self._error
        finally:
            self.stop()

    def stop(self):
        """Stop capture, release the camera and join the thread (bounded wait). Idempotent."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        th = self._thread
        if th is not None and th is not threading.current_thread():
            th.join(self.join_timeout_s)
        if self._cap is not None:      # opened but never iterated
            self._cap.release()
            self._cap = None

    close = stop
