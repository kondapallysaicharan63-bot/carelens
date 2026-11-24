"""LiveSource: threaded capture, drop-old buffer, reconnect/backoff, stall watchdog, gap-safe
event timing. Offline (fake captures); the loopback RTSP disconnect test needs ffmpeg."""
import shutil
import threading
import time
from dataclasses import replace
from datetime import datetime

import random

import numpy as np
import pytest

from carelens.events import FallDetector, InactivityDetector
from carelens.pipeline import Config, run
from carelens.sources import Frame, LiveSource
from carelens.store import Store
from carelens.synth import Scenario, Segment

from test_events import run_states


@pytest.fixture(autouse=True)
def no_leaked_capture_threads():
    yield
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and any(t.name == "carelens-capture" for t in threading.enumerate()):
        time.sleep(0.05)
    assert not [t for t in threading.enumerate() if t.name == "carelens-capture"]


class FakeCam:
    """Scripted camera. `script` items, one per open(): 'fail' (open fails) or an int n (n good
    frames then read failures); once exhausted every open yields endless good frames."""

    def __init__(self, script=(), period=0.01, read_delay=None):
        self.script, self.period, self.read_delay = list(script), period, read_delay
        self.opens, self.caps = 0, []

    def open(self, _source):
        self.opens += 1
        item = self.script.pop(0) if self.script else None
        cap = _Cap(self, None if item == "fail" else item, item == "fail")
        self.caps.append(cap)
        return cap


class _Cap:
    def __init__(self, cam, n, fail_open):
        self.cam, self.left, self.fail_open, self.released = cam, n, fail_open, False

    def isOpened(self):
        return not self.fail_open

    def get(self, *_):
        return 100.0

    def read(self):
        if self.cam.read_delay is not None and self.left is not None and self.left <= 0:
            time.sleep(self.cam.read_delay)       # a stalled connection
            return True, np.zeros((24, 32, 3), np.uint8)
        if self.left is not None:
            if self.left <= 0:
                return False, None
            self.left -= 1
        time.sleep(self.cam.period)
        return True, np.zeros((24, 32, 3), np.uint8)

    def release(self):
        self.released = True


def make(cam, **kw):
    kw.setdefault("strict_open", False)
    kw.setdefault("resize_width", None)
    return LiveSource("rtsp://user:secret@cam.local/s", open_fn=cam.open, **kw)


def test_frames_flow_and_source_is_redacted():
    cam = FakeCam()
    src = make(cam)
    assert src.is_live and "secret" not in src.source
    it = iter(src)
    frames = [next(it) for _ in range(5)]
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    assert all(b.t > a.t for a, b in zip(frames, frames[1:]))
    assert all(f.gap_before_s == 0 for f in frames) and src.status == "up"
    it.close()
    assert cam.caps[0].released


def test_slow_consumer_drops_oldest_and_latency_is_bounded():
    cam = FakeCam(period=0.01)              # ~100 fps camera, 10 fps consumer
    src = make(cam, queue_size=2)
    lat = []
    for i, fr in enumerate(src):
        lat.append(src.last_latency_s)
        time.sleep(0.1)
        if i >= 12:
            break
    assert src.frames_dropped > 20
    assert max(lat) < 2 * 0.1


def test_backoff_schedule_and_reset_after_good_frame():
    # A: 3 frames then fail -> 0.5; open fails x2 -> 1, 2; B: 2 frames then fail -> reset 0.5
    cam = FakeCam([3, "fail", "fail", 2], period=0.001)
    delays = []
    src = make(cam, sleep_fn=lambda d: (delays.append(d), time.sleep(0.01)), rng=random.Random(1))
    gaps = []
    for fr in src:
        gaps.append(fr.gap_before_s)
        if len(gaps) >= 12:
            break
    assert len(delays) == 4
    for got, base in zip(delays, [0.5, 1, 2, 0.5]):
        assert base * 0.8 <= got <= base * 1.2
    assert any(g > 0 for g in gaps) and src.reconnects >= 2


def test_backoff_is_capped():
    cam = FakeCam(["fail"] * 9)
    delays = []
    src = make(cam, sleep_fn=lambda d: (delays.append(d), time.sleep(0.001)))
    it = iter(src)
    next(it)
    it.close()
    assert len(delays) == 9
    assert max(delays) <= 30 * 1.2 and delays[-1] >= 30 * 0.8


def test_stall_triggers_reconnect_and_releases_capture():
    cam = FakeCam([2], period=0.001, read_delay=0.35)    # 2 frames, then reads hang 0.35 s
    src = make(cam, stall_s=0.2, sleep_fn=lambda d: time.sleep(0.01))
    seen = []
    for fr in src:
        seen.append(fr)
        if len(seen) >= 5:
            break
    assert src.reconnects >= 1 and cam.opens >= 2 and cam.caps[0].released
    assert any(f.gap_before_s >= 0.2 for f in seen)


def test_status_transitions_and_callback_never_raises():
    changes = []

    def cb(ch):
        changes.append(ch["status"])
        raise RuntimeError("bad callback")   # must not kill capture

    cam = FakeCam([3, "fail", None], period=0.001)
    src = make(cam, on_status=cb, sleep_fn=lambda d: time.sleep(0.02))
    n = 0
    for fr in src:
        n += 1
        if n >= 8:
            break
    assert changes[:3] == ["up", "down", "up"]
    assert src.status in ("up", "down")


def test_gap_before_s_marks_first_frame_after_outage():
    cam = FakeCam([3, None], period=0.01)
    src = make(cam, sleep_fn=lambda d: time.sleep(0.3))
    frames = []
    for fr in src:
        frames.append(fr)
        if len(frames) >= 8:
            break
    gapped = [f for f in frames if f.gap_before_s > 0]
    assert len(gapped) == 1 and gapped[0].gap_before_s >= 0.3


def test_clean_shutdown_on_consumer_exception_no_leaked_threads():
    base = threading.active_count()
    cam = FakeCam()
    src = make(cam)
    with pytest.raises(ValueError):
        for fr in src:
            raise ValueError("consumer blew up")
    assert cam.caps[0].released
    assert threading.active_count() == base


def test_strict_open_fails_fast_without_leaking_url():
    cam = FakeCam(["fail"])
    with pytest.raises(IOError) as ei:
        LiveSource("rtsp://admin:hunter2@cam/s", open_fn=cam.open)
    assert "hunter2" not in str(ei.value)


def test_max_seconds_ends_iteration_even_during_outage():
    cam = FakeCam(["fail"] * 50)
    src = make(cam, max_seconds=0.5, sleep_fn=lambda d: time.sleep(0.05))
    t0 = time.monotonic()
    assert list(src) == []
    assert time.monotonic() - t0 < 3


# ---- event timing across an outage -------------------------------------------------------
def test_outage_does_not_count_as_stillness_for_inactivity():
    st = run_states(Scenario([Segment("still", 100, posture="sit")]), fps=5)
    det = InactivityDetector(seconds=60)
    ev = []
    for s in st:
        if s.t >= 30:
            if abs(s.t - 30) < 1e-6:
                det.pause(40)
            s = replace(s, t=s.t + 40)     # 40 s unobserved
        if (e := det.update(s)):
            ev.append(e)
    # observed stillness reaches 60 s only at original t=60 -> shifted 100; never early
    assert len(ev) == 1 and ev[0].t_confirmed >= 99.9
    # control: same jump without pause() fires early (outage miscounted as stillness)
    det2, early = InactivityDetector(seconds=60), []
    for s in st:
        s = replace(s, t=s.t + 40) if s.t >= 30 else s
        if (e := det2.update(s)):
            early.append(e)
    assert early and early[0].t_confirmed < 80


def test_pending_fall_survives_outage_and_confirms_on_observed_stillness():
    scn = Scenario([Segment("still", 5, posture="stand"),
                    Segment("transition", 0.6, posture="stand", to="lie"),
                    Segment("still", 30, posture="lie")])
    st = run_states(scn, fps=10)
    det = FallDetector(still_seconds=20)
    ev = []
    for s in st:
        if s.t >= 8:                          # drop at ~5 s, 10 s outage at t=8
            if abs(s.t - 8) < 1e-6:
                det.pause(10)
            s = replace(s, t=s.t + 10)
        if (e := det.update(s)):
            ev.append(e)
    assert len(ev) == 1
    assert ev[0].details["still_for_s"] < 21           # outage excluded from stillness
    assert 5 < ev[0].t_onset < 6.5 and ev[0].t_confirmed > 33   # not confirmed early at ~18 s


# ---- run() level ------------------------------------------------------------------------
class GapSource:
    fps, source, is_live, stall_s = 15.0, "rtsp://cam/redacted", True, 5.0

    def __iter__(self):
        for i in range(60):
            gap = 12.0 if i == 30 else 0.0
            yield Frame(t=i / 15 + (12.0 if i >= 30 else 0.0), image=np.zeros((48, 64, 3), np.uint8),
                        index=i, gap_before_s=gap)


def test_run_records_camera_offline_and_store_accepts_it(tmp_path):
    scn = Scenario([Segment("still", 60, posture="sit")])
    store = Store(tmp_path / "t.db")
    res = run(GapSource(), scn.provider(), tmp_path, Config(data_label="simulated"),
              start_time=datetime(2025, 1, 1, 8, 0, 0), clip_name="gap", store=store)
    off = [e for e in res["events"] if e["kind"] == "camera_offline"]
    assert len(off) == 1 and off[0]["details"]["duration_s"] == 12.0
    assert abs(off[0]["t_onset"] - 1.93) < 0.1 and "offline for 12 s" in off[0]["message"]
    roll = store.daily_rollups()
    assert roll and roll[0]["n_falls"] == 0 and roll[0]["n_inactivity"] == 0
    assert store.history()[0]["n_inactivity"] == 0
    rid = store.history()[0]["id"]
    assert [e["kind"] for e in store.get_run(rid)["events"]] == ["camera_offline"]
    store.close()


# ---- real RTSP loopback: kill the server mid-run, restart it ------------------------------
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_loopback_rtsp_survives_server_restart(tmp_path):
    from rtsp_helper import MiniRtsp, start_ffmpeg_rtp
    proc, rtp_port, sdp = start_ffmpeg_rtp(tmp_path, seconds=60)
    s1 = s2 = None
    try:
        s1 = MiniRtsp(sdp, rtp_port)
        s1.start()
        port = s1.port
        try:
            src = LiveSource(f"rtsp://127.0.0.1:{port}/live", resize_width=None,
                             open_timeout_s=5, stall_s=3.0, backoff_max_s=2.0)
        except IOError:
            pytest.skip("OpenCV/FFmpeg could not open the loopback RTSP stream")
        frames, steady_lat, killed, deadline = [], [], False, time.monotonic() + 40
        for fr in src:
            frames.append(fr)
            if fr.gap_before_s == 0 and fr.index > 2:
                steady_lat.append(src.last_latency_s)
            if len(frames) == 15 and not killed:
                s1.kill()
                s2 = MiniRtsp(sdp, rtp_port, port=port)   # same port, as a rebooted camera
                s2.start()
                killed = True
            if killed and any(f.gap_before_s > 0 for f in frames) and \
                    sum(1 for f in frames if f.t > next(g.t for g in frames if g.gap_before_s > 0)) >= 5:
                break
            if time.monotonic() > deadline:
                break
        assert src.reconnects >= 1, "no reconnect after the server was killed"
        gapped = [f for f in frames if f.gap_before_s > 0]
        assert gapped and gapped[0].gap_before_s > 0
        assert sum(f.t > gapped[0].t for f in frames) >= 5      # frames resumed
        assert steady_lat and max(steady_lat) < 1.0
    finally:
        for s in (s1, s2):
            if s:
                s.stop.set()
        proc.kill()
        proc.wait()


# ---- review follow-ups ------------------------------------------------------------------
class TimedSource:
    """Frames at scripted times; `gaps` maps frame index -> gap_before_s."""

    def __init__(self, times, gaps=None, fps=10.0, live=False):
        self.times, self.gaps, self.fps, self.is_live = times, gaps or {}, fps, live
        self.source, self.stall_s = "rtsp://cam/redacted", 5.0

    def __iter__(self):
        for i, t in enumerate(self.times):
            yield Frame(t=t, image=np.zeros((48, 64, 3), np.uint8), index=i,
                        gap_before_s=self.gaps.get(i, 0.0))


class Remap:
    """Scenario provider whose clock excludes the outage (the person kept doing their thing)."""

    def __init__(self, scn, gap_t, gap):
        self.scn, self.gap_t, self.gap = scn, gap_t, gap

    def process(self, image, t):
        return self.scn.pose_landmarks(t - (self.gap if t >= self.gap_t else 0.0))

    def close(self):
        pass


def test_short_gap_does_not_reset_inactivity_and_fires_once(tmp_path):
    scn = Scenario([Segment("still", 60, posture="sit")])
    times = [i / 10 + (1.0 if i >= 100 else 0.0) for i in range(400)]
    src = TimedSource(times, {100: 1.0})
    res = run(src, Remap(scn, 11.0, 1.0), tmp_path, Config(inactivity_seconds=20, data_label="simulated"),
              clip_name="g1")
    ev = [e for e in res["events"] if e["kind"] == "inactivity"]
    assert len(ev) == 1
    assert abs((ev[0]["t_confirmed"] - ev[0]["t_onset"]) - 21.0) < 0.5   # 20 s observed + 1 s gap
    assert ev[0]["t_onset"] < 1.0                                           # stillness not restarted


def test_fall_not_confirmed_in_empty_room_after_long_outage_but_confirmed_if_person_still_there():
    def feed(segments, shift=3 * 3600):
        det, out = FallDetector(still_seconds=20), []
        for s in run_states(Scenario(segments), fps=10):
            if s.t >= 9:
                if abs(s.t - 9) < 1e-6:
                    det.pause(shift)
                s = replace(s, t=s.t + shift)
            if (e := det.update(s)):
                out.append(e)
        return out

    base = [Segment("still", 5, posture="stand"), Segment("transition", 0.6, posture="stand", to="lie")]
    assert feed(base + [Segment("still", 3.4, posture="lie"), Segment("absent", 60)]) == []
    ev = feed(base + [Segment("still", 45, posture="lie")])
    assert len(ev) == 1 and ev[0].t_confirmed > 3 * 3600


def test_live_clip_fps_and_posture_sampling_follow_consumed_rate(tmp_path):
    import cv2
    scn = Scenario([Segment("still", 5, posture="stand"),
                    Segment("transition", 0.6, posture="stand", to="lie"),
                    Segment("still", 30, posture="lie")])
    src = TimedSource([i / 10 for i in range(356)], fps=30.0, live=True)   # 30 fps cam, 10 fps consumed
    res = run(src, scn.provider(), tmp_path, Config(data_label="simulated"), clip_name="lv")
    falls = [e for e in res["events"] if e["kind"] == "fall"]
    assert len(falls) == 1 and falls[0]["clip"]
    fps = cv2.VideoCapture(str(tmp_path / "clips" / falls[0]["clip"])).get(cv2.CAP_PROP_FPS)
    assert 9 <= fps <= 11
    ts = [p["t"] for p in res["posture_track"]]
    assert all(0.9 <= b - a <= 1.2 for a, b in zip(ts, ts[1:])) and len(ts) >= 30


def test_slow_first_read_is_accepted_not_a_stall():
    class SlowFirst(FakeCam):
        def open(self, _s):
            cap = super().open(_s)
            orig = cap.read
            state = {"n": 0}

            def read():
                state["n"] += 1
                if state["n"] == 1:
                    time.sleep(0.4)          # keyframe wait > stall_s
                return orig()
            cap.read = read
            return cap

    cam = SlowFirst()
    src = make(cam, stall_s=0.2, open_timeout_s=1.0)
    n = 0
    for fr in src:
        n += 1
        if n >= 5:
            break
    assert cam.opens == 1 and src.reconnects == 0


def test_outage_before_first_frame_is_measured_from_start_and_not_a_reconnect():
    cam = FakeCam(["fail", "fail"])
    src = make(cam, sleep_fn=lambda d: time.sleep(0.1))
    first = next(iter(src))
    assert first.gap_before_s >= 0.2 and first.gap_before_s == pytest.approx(first.t, abs=1e-6)
    assert src.reconnects == 0


def test_dead_camera_reported_at_onset_run_ends_at_max_outage_and_is_saved(tmp_path):
    cam = FakeCam(["fail"] * 500)
    seen = []
    src = make(cam, stall_s=0.2, max_outage_s=0.8, sleep_fn=lambda d: time.sleep(0.05))
    store = Store(tmp_path / "d.db")
    t0 = time.monotonic()
    res = run(src, Scenario([Segment("still", 5, posture="sit")]).provider(), tmp_path,
              Config(data_label="live"), start_time=datetime(2025, 1, 1, 8), clip_name="dead",
              store=store, on_event=lambda e: seen.append((time.monotonic(), dict(e))))
    assert time.monotonic() - t0 < 5 and src.gave_up
    off = [e for e in res["events"] if e["kind"] == "camera_offline"]
    assert len(off) == 1 and off[0]["t_onset"] == 0.0
    assert off[0]["details"]["duration_s"] >= 0.8 and "offline for" in off[0]["message"]
    assert len(seen) == 1 and seen[0][1]["kind"] == "camera_offline"   # hook fired at onset
    assert res["duration_s"] >= 0.8
    rid = store.history()[0]["id"]
    assert [e["kind"] for e in store.get_run(rid)["events"]] == ["camera_offline"]
    assert store.daily_rollups()[0]["n_falls"] == 0
    store.close()


def test_run_ending_while_down_without_max_outage_finalizes_event(tmp_path):
    cam = FakeCam(["fail"] * 500)
    src = make(cam, stall_s=0.2, max_seconds=0.7, sleep_fn=lambda d: time.sleep(0.05))
    res = run(src, Scenario([Segment("still", 5, posture="sit")]).provider(), tmp_path,
              Config(data_label="live"), clip_name="dead2")
    off = [e for e in res["events"] if e["kind"] == "camera_offline"]
    assert len(off) == 1 and off[0]["details"]["duration_s"] >= 0.5 and not src.gave_up


def test_recovered_outage_updates_event_duration_via_hook(tmp_path):
    cam = FakeCam([20, "fail", "fail", "fail", "fail", "fail"], period=0.01)
    src = make(cam, stall_s=0.2, max_seconds=3.0, sleep_fn=lambda d: time.sleep(0.15))
    res = run(src, Scenario([Segment("still", 10, posture="sit")]).provider(), tmp_path,
              Config(data_label="live"), clip_name="rec")
    off = [e for e in res["events"] if e["kind"] == "camera_offline"]
    assert len(off) == 1 and off[0]["details"]["duration_s"] >= 0.5
    assert off[0]["t_confirmed"] == pytest.approx(off[0]["t_onset"] + off[0]["details"]["duration_s"], abs=0.02)


def test_max_outage_config_field():
    from carelens.config import CameraConfig
    base = dict(id="c1", room="r", type="rtsp", source="rtsp://h/s")
    assert CameraConfig(**base).max_outage_s is None
    assert CameraConfig(**base, max_outage_s=600).max_outage_s == 600
    with pytest.raises(Exception):
        CameraConfig(**base, max_outage_s=0)


def test_gap_closes_gait_segment_and_resets_motion_without_tracker(tmp_path):
    scn = Scenario([Segment("walk", 4, x0=0.2, x1=1.1), Segment("walk", 4, x0=1.1, x1=0.2)])
    times = [i / 15 + (3.0 if i >= 60 else 0.0) for i in range(120)]
    src = TimedSource(times, {60: 3.0}, fps=15.0)
    res = run(src, Remap(scn, 4.0 + 3.0, 3.0), tmp_path,
              Config(tracking=None, data_label="simulated"), clip_name="gait")
    assert len(res["gait_segments"]) == 2
