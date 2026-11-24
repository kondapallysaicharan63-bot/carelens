"""Shared RTSP loopback helpers: ffmpeg encodes RTP to a local UDP port and a tiny in-test
RTSP server (OPTIONS/DESCRIBE/SETUP/PLAY, interleaved TCP) relays it to OpenCV/FFmpeg clients."""
import socket
import subprocess
import threading
import time

import pytest


def free_port(kind=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MiniRtsp(threading.Thread):
    def __init__(self, sdp: str, rtp_port: int, port: int = 0):
        super().__init__(daemon=True)
        self.sdp = sdp.replace("\r\n", "\n").rstrip("\n") + "\na=control:track0\n"
        self.lsock = socket.socket()
        self.lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.lsock.bind(("127.0.0.1", port))
        self.lsock.listen(1)
        self.lsock.settimeout(15)
        self.conn = None
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
            self.conn = conn
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
            if self.conn is not None:
                self.conn.close()   # a killed server must drop the client's connection
            self.rtp.close()
            self.lsock.close()

    def kill(self):
        """Simulate the camera/server dying: stop relaying and close the connection."""
        self.stop.set()
        self.join(timeout=5)


def start_ffmpeg_rtp(tmp_path, seconds: int = 15, rate: int = 10, size: str = "320x240"):
    """Run ffmpeg sending a test pattern as RTP to a free UDP port. Returns (proc, rtp_port,
    sdp_text) or calls pytest.skip. Caller must proc.kill()."""
    rtp_port = free_port(socket.SOCK_DGRAM)
    sdp_file = tmp_path / "s.sdp"
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-re", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}",
         "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
         "-bf", "0", "-g", str(rate), "-pix_fmt", "yuv420p", "-f", "rtp", "-sdp_file", str(sdp_file),
         f"rtp://127.0.0.1:{rtp_port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        if sdp_file.exists() and sdp_file.stat().st_size > 0:
            return proc, rtp_port, sdp_file.read_text()
        if proc.poll() is not None:
            pytest.skip("ffmpeg could not produce an RTP stream")
        time.sleep(0.1)
    proc.kill()
    pytest.skip("ffmpeg produced no SDP")
