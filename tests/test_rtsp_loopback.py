"""Real RTSP-over-TCP loopback: ffmpeg encodes RTP to a local UDP port, and a tiny in-test
RTSP server (OPTIONS/DESCRIBE/SETUP/PLAY, interleaved TCP) relays it to VideoSource."""
import shutil
import time

import pytest

from carelens.sources import VideoSource

from rtsp_helper import MiniRtsp, free_port, start_ffmpeg_rtp

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_loopback_rtsp_stream(tmp_path):
    proc, rtp_port, sdp = start_ffmpeg_rtp(tmp_path)
    server = None
    try:
        server = MiniRtsp(sdp, rtp_port)
        server.start()
        try:
            src = VideoSource(f"rtsp://127.0.0.1:{server.port}/live", max_seconds=2,
                              resize_width=None, rtsp_transport="tcp", open_timeout_s=8)
        except IOError:
            pytest.skip("OpenCV/FFmpeg could not open the loopback RTSP stream")
        frames = list(src)
        assert len(frames) >= 5
        assert frames[0].image.shape == (240, 320, 3)
        assert src.is_live and 5 <= src.fps <= 30
    finally:
        if server:
            server.stop.set()
        proc.kill()
        proc.wait()


def test_unreachable_rtsp_error_has_no_credentials():
    port = free_port()  # closed
    t0 = time.monotonic()
    with pytest.raises(IOError) as ei:
        VideoSource(f"rtsp://admin:hunter2@127.0.0.1:{port}/x", open_timeout_s=2)
    assert time.monotonic() - t0 < 8
    assert "hunter2" not in str(ei.value) and "admin" not in str(ei.value)


def test_ffmpeg_fallback_and_env_merge(monkeypatch):
    import os
    import cv2
    from carelens import sources
    calls, seen_env = [], []

    class Cap:
        def __init__(self, ok): self.ok = ok
        def isOpened(self): return self.ok
        def release(self): pass
        def get(self, *_): return 10.0

    def fake(url, *args):
        calls.append(args)
        seen_env.append(os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"))
        return Cap(bool(not args))  # FFmpeg-forced open fails, plain open works
    monkeypatch.setattr(cv2, "VideoCapture", fake)
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "foo;1|rtsp_transport;udp")
    src = VideoSource("RTSPS://h/x", rtsp_transport="tcp", open_timeout_s=5)
    assert len(calls) == 2 and calls[1] == () and src.is_live
    assert seen_env[0] == "foo;1|rtsp_transport;tcp"
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "foo;1|rtsp_transport;udp"
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    sources._open_rtsp("rtsp://h/x", "udp", 0.0001)
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ and calls[2][1][1] == 1
    assert VideoSource.__init__ and sources._merge_ffmpeg_options(None, "udp") == "rtsp_transport;udp"


def test_open_timeout_bounds():
    from carelens.config import CameraConfig
    base = dict(id="c1", room="r", type="rtsp", source="rtsp://h/s")
    assert CameraConfig(**base, open_timeout_s=300).open_timeout_s == 300
    with pytest.raises(Exception):
        CameraConfig(**base, open_timeout_s=301)
