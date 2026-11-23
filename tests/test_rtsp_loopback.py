"""Real RTSP-over-TCP loopback: ffmpeg encodes RTP to a local UDP port, and a tiny in-test
RTSP server (OPTIONS/DESCRIBE/SETUP/PLAY, interleaved TCP) relays it to VideoSource."""
import shutil
import socket
import subprocess
import threading
import time

import pytest

from carelens.sources import VideoSource

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _free_port(kind=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _MiniRtsp(threading.Thread):
    def __init__(self, sdp: str, rtp_port: int):
        super().__init__(daemon=True)
        self.sdp = sdp.replace("\r\n", "\n").rstrip("\n") + "\na=control:track0\n"
        self.lsock = socket.socket()
        self.lsock.bind(("127.0.0.1", 0))
        self.lsock.listen(1)
        self.lsock.settimeout(15)
        self.port = self.lsock.getsockname()[1]
        self.rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp.bind(("127.0.0.1", rtp_port))
        self.rtp.settimeout(0.5)
        self.stop = threading.Event()

    def _reply(self, conn, cseq, headers="", body=""):
        msg = f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n{headers}"
        if body:
            msg += f"Content-Type: application/sdp\r\nContent-Length: {len(body)}\r\n"
        conn.sendall((msg + "\r\n" + body).encode())

    def run(self):
        try:
            conn, _ = self.lsock.accept()
            conn.settimeout(10)
            buf = b""
            while not self.stop.is_set():
                while b"\r\n\r\n" not in buf:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    buf += chunk
                req, buf = buf.split(b"\r\n\r\n", 1)
                lines = req.decode(errors="replace").split("\r\n")
                method = lines[0].split()[0]
                cseq = next((l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("cseq")), "0")
                if method == "OPTIONS":
                    self._reply(conn, cseq, "Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n")
                elif method == "DESCRIBE":
                    self._reply(conn, cseq, "Content-Base: rtsp://127.0.0.1/live/\r\n", self.sdp)
                elif method == "SETUP":
                    self._reply(conn, cseq, "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n"
                                            "Session: 1234\r\n")
                elif method == "PLAY":
                    self._reply(conn, cseq, "Session: 1234\r\n")
                    break
                else:
                    self._reply(conn, cseq)
            while not self.stop.is_set():
                try:
                    pkt = self.rtp.recv(65535)
                except socket.timeout:
                    continue
                conn.sendall(b"$\x00" + len(pkt).to_bytes(2, "big") + pkt)
        except OSError:
            pass
        finally:
            self.rtp.close()
            self.lsock.close()


def test_loopback_rtsp_stream(tmp_path):
    rtp_port = _free_port(socket.SOCK_DGRAM)
    sdp_file = tmp_path / "s.sdp"
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-re", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
         "-t", "15", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-bf", "0",
         "-g", "10", "-pix_fmt", "yuv420p", "-f", "rtp", "-sdp_file", str(sdp_file),
         f"rtp://127.0.0.1:{rtp_port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    server = None
    try:
        for _ in range(40):
            if sdp_file.exists() and sdp_file.stat().st_size > 0:
                break
            if proc.poll() is not None:
                pytest.skip("ffmpeg could not produce an RTP stream")
            time.sleep(0.1)
        else:
            pytest.skip("ffmpeg produced no SDP")
        server = _MiniRtsp(sdp_file.read_text(), rtp_port)
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
    port = _free_port()  # closed
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
