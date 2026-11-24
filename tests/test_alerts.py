import http.server
import json
import logging
import re
import threading
import time
from datetime import datetime, time as dtime, timezone
from pathlib import Path

import pytest

from carelens import alerts as A
from carelens.alerts import (AlertDispatcher, DesktopSink, PermanentSinkError, WebhookSink,
                             backoff_delay, build_payload, build_sinks, in_quiet_hours, redact_error)
from carelens.config import AlertsConfig, AlertSinkConfig, AppConfig, QuietHours
from carelens.store import Store

T0 = 1_800_000_000.0   # arbitrary fixed instant


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


class FakeSink:
    def __init__(self, name="fake", kinds=A.ALERT_KINDS, script=()):
        self.name, self.kinds, self.script, self.sent = name, tuple(kinds), list(script), []

    def send(self, alert):
        if self.script:
            r = self.script.pop(0)
            if isinstance(r, BaseException):
                raise r
        self.sent.append(alert)


def cfg(**kw):
    kw.setdefault("enabled", True)
    return AlertsConfig(**kw)


def mk(tmp_path, sinks, clock=None, start=False, **kw):
    clock = clock or Clock()
    d = AlertDispatcher(tmp_path / "a.db", cfg(**kw), sinks, {"cam": "kitchen", "other": "hall"},
                        clock=clock, start=start, jitter=lambda: 0.0,
                        local_time=lambda t: datetime.fromtimestamp(t, timezone.utc))
    return d, clock


def ev(kind="fall", **extra):
    return {"kind": kind, "t_onset": 1.0, "t_confirmed": 2.0, "message": "m", "clip": "c.mp4", **extra}


def rows(tmp_path):
    with Store(tmp_path / "a.db") as s:
        return [dict(r) for r in s.db.execute("SELECT * FROM alerts ORDER BY id")]


def test_disabled_does_nothing(tmp_path):
    sink = FakeSink()
    d = AlertDispatcher(tmp_path / "a.db", AlertsConfig(), [sink], {}, start=True)
    d.submit("cam", ev())
    d.close()
    assert sink.sent == [] and not (tmp_path / "a.db").exists()


def test_kind_filter_and_unknown_kinds(tmp_path):
    fall_only, all_ = FakeSink("f", ["fall"]), FakeSink("a")
    d, _ = mk(tmp_path, [fall_only, all_])
    d.submit("cam", ev("inactivity"))
    d.submit("cam", ev("vitals_note"))      # not an alert kind
    d.drain_once()
    assert fall_only.sent == [] and len(all_.sent) == 1 and len(rows(tmp_path)) == 1
    d.close()


def test_no_matching_sink_is_recorded_for_the_banner(tmp_path):
    d, _ = mk(tmp_path, [FakeSink("f", ["fall"])])
    d.submit("cam", ev("camera_offline"))
    d.drain_once()
    (r,) = rows(tmp_path)
    assert r["status"] == "suppressed" and r["sink"] == ""
    d.close()


def test_dedupe_window_and_restart(tmp_path):
    s1 = FakeSink()
    d, clock = mk(tmp_path, [s1], dedupe_window_s=900)
    d.submit("cam", ev()); d.submit("cam", ev()); d.drain_once()
    assert len(s1.sent) == 1
    clock.t += 899
    d.submit("cam", ev()); d.drain_once()
    assert len(s1.sent) == 1                      # still inside the window
    d.close()
    s2 = FakeSink()                               # new dispatcher, same DB: dedupe survives restart
    d2, _ = mk(tmp_path, [s2], clock=clock, dedupe_window_s=900)
    d2.submit("cam", ev()); d2.submit("other", ev()); d2.drain_once()
    assert len(s2.sent) == 1 and s2.sent[0]["camera_id"] == "other"   # other camera not deduped
    clock.t += 2
    d2.submit("cam", ev()); d2.drain_once()
    assert len(s2.sent) == 2                      # window passed (901 s since the first)
    d2.submit("cam", ev("inactivity")); d2.drain_once()
    assert len(s2.sent) == 3                      # different kind is not deduped
    d2.close()


def test_dedupe_zero_disables(tmp_path):
    s = FakeSink()
    d, _ = mk(tmp_path, [s], dedupe_window_s=0)
    d.submit("cam", ev()); d.submit("cam", ev()); d.drain_once()
    assert len(s.sent) == 2
    d.close()


QH = {"cam": QuietHours(start="22:00", end="07:00")}


@pytest.mark.parametrize("hhmm,inside", [("21:59", False), ("22:00", True), ("23:30", True),
                                         ("00:00", True), ("06:59", True), ("07:00", False),
                                         ("12:00", False)])
def test_quiet_hours_wrap_midnight(hhmm, inside):
    h, m = map(int, hhmm.split(":"))
    assert in_quiet_hours(QH["cam"], dtime(h, m)) is inside
    assert in_quiet_hours(QuietHours(start="09:00", end="17:00"), dtime(h, m)) is (9 <= h < 17)


def at(hour):
    return datetime(2026, 1, 15, hour, 30, tzinfo=timezone.utc).timestamp()


def test_quiet_hours_suppress_only_inactivity_per_camera(tmp_path):
    s = FakeSink()
    d, clock = mk(tmp_path, [s], clock=Clock(at(23)), quiet_hours=QH, dedupe_window_s=0)
    d.submit("cam", ev("inactivity"))
    d.submit("cam", ev("fall"))
    d.submit("cam", ev("camera_offline"))
    d.submit("other", ev("inactivity"))     # no quiet hours configured for this camera
    d.drain_once()
    assert sorted((a["camera_id"], a["kind"]) for a in s.sent) == [
        ("cam", "camera_offline"), ("cam", "fall"), ("other", "inactivity")]
    assert [(r["kind"], r["status"]) for r in rows(tmp_path) if r["status"] == "suppressed"] == [
        ("inactivity", "suppressed")]
    clock.t = at(12)                         # daytime: delivered
    d.submit("cam", ev("inactivity")); d.drain_once()
    assert len(s.sent) == 4
    d.close()


def test_backoff_schedule_and_failed_after_max(tmp_path):
    s = FakeSink(script=[RuntimeError("x")] * 10)
    d, clock = mk(tmp_path, [s], max_attempts=4, backoff_base_s=30, backoff_max_s=100)
    d.submit("cam", ev())
    waits = []
    for _ in range(6):
        d.drain_once()
        (r,) = rows(tmp_path)
        if r["status"] == "failed":
            break
        waits.append(r["next_attempt_ts"] - clock.t)
        clock.t = r["next_attempt_ts"]
    assert waits == [60, 100, 100] and r["status"] == "failed" and r["attempts"] == 4
    assert backoff_delay(1, 30, 3600) == 60 and backoff_delay(10, 30, 3600) == 3600
    assert backoff_delay(2, 30, 3600, jitter=0.1) == pytest.approx(132)
    # not due yet -> no extra attempts
    n = len(s.script)
    d.drain_once()
    assert len(s.script) == n
    d.close()


def test_retry_then_success(tmp_path):
    s = FakeSink(script=[RuntimeError("down")])
    d, clock = mk(tmp_path, [s])
    d.submit("cam", ev()); d.drain_once()
    assert s.sent == [] and rows(tmp_path)[0]["status"] == "pending"
    d.drain_once(); assert s.sent == []              # backoff not elapsed
    clock.t += 61
    d.drain_once()
    assert len(s.sent) == 1 and rows(tmp_path)[0]["status"] == "sent"
    d.close()


def test_last_error_is_redacted(tmp_path):
    secret_url = "http://user:pw@10.0.0.5/hook/SECRETPATH?k=abc"
    s = FakeSink(script=[RuntimeError(f"cannot reach {secret_url} with Bearer tok123 and tok999")], )
    s._secrets = lambda: ["tok999"]
    d, _ = mk(tmp_path, [s])
    d.submit("cam", ev()); d.drain_once()
    err = rows(tmp_path)[0]["last_error"]
    for bad in ("user", "pw", "SECRETPATH", "abc", "tok123", "tok999", "10.0.0.5", "http://"):
        assert bad not in err
    assert len(err) <= 200 and len(redact_error("x" * 1000)) == 200
    d.close()


def test_raising_and_hanging_sink_do_not_block_submit(tmp_path):
    class Hang:
        name, kinds = "hang", A.ALERT_KINDS
        gate = threading.Event()

        def send(self, alert):
            self.gate.wait(10)

    h = Hang()
    d, _ = mk(tmp_path, [h, FakeSink("boom", script=[RuntimeError("x")])], start=True)
    t = time.perf_counter()
    for i in range(20):
        d.submit("cam", ev(str(["fall", "inactivity", "camera_offline"][i % 3])))
    assert time.perf_counter() - t < 0.05
    d.submit("cam", None)                  # garbage must not raise
    d.submit("cam", {"kind": ["unhashable"]})
    Hang.gate.set()
    d.close()


def test_submit_never_raises_when_db_unwritable(tmp_path, caplog):
    d = AlertDispatcher(tmp_path / "no" / "such" / "dir" / "x.db", cfg(), [FakeSink()], {},
                        start=False)
    d.submit("cam", ev())
    d.drain_once()      # recording fails: logged, not raised
    d.close()


def test_pending_row_survives_restart_and_sends(tmp_path):
    clock = Clock()
    s = FakeSink(script=[RuntimeError("down")])
    d, _ = mk(tmp_path, [s], clock=clock)
    d.submit("cam", ev()); d.drain_once()
    d.close()                               # "process exits" with a pending row
    s2 = FakeSink()
    d2, _ = mk(tmp_path, [s2], clock=clock)
    clock.t += 100
    d2.drain_once()                         # nothing re-submitted; the stored row is resumed
    assert len(s2.sent) == 1 and s2.sent[0]["camera_id"] == "cam"
    assert rows(tmp_path)[0]["status"] == "sent"
    d2.close()


def test_unknown_sink_row_marked_failed(tmp_path):
    d, clock = mk(tmp_path, [FakeSink("old", script=[RuntimeError("x")])])
    d.submit("cam", ev()); d.drain_once(); d.close()
    clock.t += 100
    d2, _ = mk(tmp_path, [FakeSink("new")], clock=clock)
    d2.drain_once()
    assert rows(tmp_path)[0]["status"] == "failed"
    d2.close()


def test_real_thread_smoke(tmp_path):
    s = FakeSink()
    d, _ = mk(tmp_path, [s], clock=time.time, start=True)
    d.submit("cam", ev())
    deadline = time.time() + 5
    while not s.sent and time.time() < deadline:
        time.sleep(0.02)
    d.close()
    assert len(s.sent) == 1 and not d._thread.is_alive()


# ---- payload, titles, wording -----------------------------------------------------------
def test_payload_shape_and_titles():
    row = {"id": 7, "camera_id": "cam", "kind": "fall", "created_ts": T0,
           "event_json": json.dumps({"clip": "c.mp4"})}
    p = build_payload(row, "kitchen")
    assert set(p) == {"id", "camera_id", "room", "kind", "title", "message", "ts", "clip_name",
                      "disclaimer", "data_label"}
    assert p["title"] == "Possible fall — check-in suggested" and p["clip_name"] == "c.mp4"
    assert p["disclaimer"] == A.DISCLAIMER and "kitchen" in p["message"]
    assert datetime.fromisoformat(p["ts"]).timestamp() == T0
    assert A.TITLES["inactivity"] == "Long period without movement — check-in suggested"
    assert A.TITLES["camera_offline"] == "Camera offline — check the camera"


FORBIDDEN = ["diagnos", "emergency detected", "fall confirmed", "confirmed fall"]


def _scan(text):
    low = text.lower()
    for f in FORBIDDEN:
        assert f not in low, f
    stripped = low.replace("not a medical device", "").replace("a wellness aid, not a medical device", "")
    assert "medical device" not in stripped


def test_wording_in_sources_and_payloads():
    root = Path(__file__).resolve().parent.parent
    _scan((root / "carelens" / "alerts.py").read_text())
    _scan((root / "dashboard" / "static" / "index.html").read_text().replace(
        "not a diagnosis", ""))   # pre-existing "Wellness trends only, not a diagnosis."
    for kind in A.ALERT_KINDS:
        _scan(json.dumps(build_payload({"id": 1, "camera_id": "c", "kind": kind, "created_ts": T0,
                                        "event_json": None}, "room")))


# ---- webhook against a local server -----------------------------------------------------
class Hook:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append((self.path, dict(self.headers), json.loads(body)))
                code = outer.responses.pop(0) if outer.responses else 200
                self.send_response(code)
                if code in (301, 302, 307):
                    self.send_header("Location", f"{outer.url}/elsewhere")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown(); self.srv.server_close()


@pytest.fixture
def hook():
    made = []
    def mk_(r=()):
        h = Hook(r); made.append(h); return h
    yield mk_
    for h in made:
        h.stop()


PAYLOAD = {"id": 1, "title": "t", "disclaimer": A.DISCLAIMER}


def test_webhook_payload_and_bearer(hook):
    h = hook()
    WebhookSink(h.url + "/hook", token="s3cret").send(PAYLOAD)
    path, headers, body = h.requests[0]
    assert path == "/hook" and body == PAYLOAD and headers["Authorization"] == "Bearer s3cret"
    assert headers["Content-Type"] == "application/json"
    h2 = hook()
    WebhookSink(h2.url).send(PAYLOAD)
    assert "Authorization" not in h2.requests[0][1]


def test_webhook_500_retried_via_dispatcher(hook, tmp_path):
    h = hook([500, 200])
    d, clock = mk(tmp_path, [WebhookSink(h.url + "/secretpath", token="tok")])
    d.submit("cam", ev()); d.drain_once()
    (r,) = rows(tmp_path)
    assert r["status"] == "pending" and r["attempts"] == 1 and r["last_error"].endswith("HTTP 500")
    assert "secretpath" not in r["last_error"] and "tok" not in r["last_error"]
    clock.t += 61
    d.drain_once()
    assert rows(tmp_path)[0]["status"] == "sent" and len(h.requests) == 2
    assert h.requests[1][2]["room"] == "kitchen"
    d.close()


def test_webhook_connection_refused_retried(tmp_path):
    h = Hook([]); url = h.url; h.stop()      # nothing listens any more
    d, _ = mk(tmp_path, [WebhookSink(url)])
    d.submit("cam", ev()); d.drain_once()
    (r,) = rows(tmp_path)
    assert r["status"] == "pending" and r["attempts"] == 1 and "127.0.0.1" not in r["last_error"]
    d.close()


def test_webhook_redirect_not_followed(hook):
    h = hook([302, 200])
    with pytest.raises(PermanentSinkError):
        WebhookSink(h.url + "/a").send(PAYLOAD)
    assert len(h.requests) == 1


def test_webhook_url_validation():
    with pytest.raises(ValueError) as e:
        WebhookSink("https://hooks.example.com/abc/SECRET")
    assert "SECRET" not in str(e.value) and "allow_public_hosts" in str(e.value)
    WebhookSink("https://hooks.example.com/x", allow_hosts=["hooks.example.com"])
    WebhookSink("http://192.168.1.20:8080/x")
    WebhookSink("http://ha.local/x")
    with pytest.raises(ValueError) as e:
        WebhookSink("http://admin:hunter2@192.168.1.20/x")
    assert "hunter2" not in str(e.value) and "admin" not in str(e.value)
    with pytest.raises(ValueError):
        WebhookSink("ftp://192.168.1.20/x")


def test_build_sinks_from_env(monkeypatch):
    c = cfg(sinks=[AlertSinkConfig(type="webhook", url_env="U", token_env="T", kinds=["fall"]),
                   AlertSinkConfig(type="desktop")])
    with pytest.raises(ValueError, match="U is not set"):
        build_sinks(c, {})
    sinks = build_sinks(c, {"U": "http://192.168.1.2/h", "T": "tok"})
    assert [s.name for s in sinks] == ["webhook", "desktop"] and sinks[0].kinds == ("fall",)
    with pytest.raises(ValueError) as e:
        build_sinks(c, {"U": "https://evil.example.com/SECRET"})
    assert "SECRET" not in str(e.value)


def test_desktop_sink_missing_notify_send_fails_once(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(A.shutil, "which", lambda n: None)
    d, clock = mk(tmp_path, [DesktopSink()], dedupe_window_s=0)
    with caplog.at_level(logging.WARNING, logger="carelens.alerts"):
        d.submit("cam", ev()); d.submit("cam", ev()); d.drain_once()
        clock.t += 10_000; d.drain_once()
    rs = rows(tmp_path)
    assert [r["status"] for r in rs] == ["failed", "failed"] and all(r["attempts"] == 1 for r in rs)
    assert sum("desktop failed" in m for m in caplog.messages) == 1    # logged once
    d.close()


# ---- config ------------------------------------------------------------------------------
def test_alert_config_validation():
    assert AppConfig().alerts.enabled is False
    with pytest.raises(Exception):
        AlertSinkConfig(type="webhook")
    with pytest.raises(Exception):
        QuietHours(start="25:00", end="07:00")
    with pytest.raises(Exception):
        QuietHours(start="22:00", end="22:00")
    with pytest.raises(Exception):
        AppConfig(alerts={"quiet_hours": {"nope": {"start": "22:00", "end": "07:00"}}})
    with pytest.raises(Exception):
        AppConfig.model_validate({"alerts": {"sinks": [{"type": "webhook", "url_env": "U",
                                                         "url": "http://x"}]}})
    ok = AppConfig.model_validate({
        "cameras": [{"id": "demo", "room": "r", "type": "file", "source": "a.mp4"}],
        "alerts": {"enabled": True, "quiet_hours": {"demo": {"start": "22:00", "end": "07:00"}},
                   "sinks": [{"type": "desktop"}]}})
    assert ok.alerts.sinks[0].kinds == ["fall", "inactivity", "camera_offline"]


# ---- pipeline integration ----------------------------------------------------------------
def test_run_survives_raising_on_event(tmp_path):
    from carelens.pipeline import Config, run
    from carelens.synth import Scenario, Segment
    from test_live_source import GapSource

    def boom(e):
        raise RuntimeError("hook exploded")
    res = run(GapSource(), Scenario([Segment("still", 60, posture="sit")]).provider(), tmp_path,
              Config(data_label="simulated"), start_time=datetime(2025, 1, 1, 8), clip_name="g",
              on_event=boom)
    assert [e["kind"] for e in res["events"]].count("camera_offline") == 1


def test_camera_gap_yields_exactly_one_offline_alert(tmp_path):
    from carelens.pipeline import Config, run
    from carelens.synth import Scenario, Segment
    from test_live_source import GapSource

    s = FakeSink()
    d, _ = mk(tmp_path, [s])
    run(GapSource(), Scenario([Segment("still", 60, posture="sit")]).provider(), tmp_path / "o",
        Config(data_label="simulated"), start_time=datetime(2025, 1, 1, 8), clip_name="g",
        on_event=lambda e: d.submit("cam", e))
    d.drain_once()
    assert [a["kind"] for a in s.sent] == ["camera_offline"]
    assert s.sent[0]["title"] == "Camera offline — check the camera" and s.sent[0]["room"] == "kitchen"
    d.close()


# ---- review fixes ------------------------------------------------------------------------
def test_suppressed_row_does_not_dedupe_a_real_alert(tmp_path):
    s = FakeSink()
    d, clock = mk(tmp_path, [s], clock=Clock(datetime(2026, 1, 15, 6, 55, tzinfo=timezone.utc).timestamp()),
                  quiet_hours=QH, dedupe_window_s=900)
    d.submit("cam", ev("inactivity")); d.drain_once()
    assert s.sent == [] and rows(tmp_path)[0]["status"] == "suppressed"
    clock.t += 600                                    # 07:05, quiet hours over
    d.submit("cam", ev("inactivity")); d.drain_once()
    assert len(s.sent) == 1
    d.close()


def test_token_env_missing_is_an_error():
    c = cfg(sinks=[AlertSinkConfig(type="webhook", url_env="U", token_env="T")])
    with pytest.raises(ValueError, match="T is not set"):
        build_sinks(c, {"U": "http://192.168.1.2/h"})
    with pytest.raises(ValueError, match="T is not set"):
        build_sinks(c, {"U": "http://192.168.1.2/h", "T": ""})


def test_non_live_payload_is_labelled_test(tmp_path):
    s = FakeSink()
    d = AlertDispatcher(tmp_path / "a.db", cfg(), [s], {"cam": "kitchen"}, clock=Clock(),
                        start=False, data_label="simulated")
    d.submit("cam", ev()); d.drain_once()
    assert s.sent[0]["title"] == "[TEST] " + A.TITLES["fall"] and s.sent[0]["data_label"] == "simulated"
    d.close()
    d2, _ = mk(tmp_path / "x" if False else tmp_path, [FakeSink()])   # live default
    assert d2.data_label == "live"


class Flaky:
    def __init__(self, name, fail):
        self.name, self.kinds, self.fail, self.calls, self.sent = name, A.ALERT_KINDS, fail, 0, []

    def send(self, alert):
        self.calls += 1
        if self.fail:
            raise TimeoutError("timed out")
        self.sent.append(alert)


def test_fresh_alert_not_starved_and_failing_sink_tried_once_per_pass(tmp_path):
    bad, good = Flaky("bad", True), Flaky("good", False)
    d, clock = mk(tmp_path, [bad, good], dedupe_window_s=0, backoff_base_s=30)
    for _ in range(30):
        d.submit("cam", ev("inactivity"))
    d.drain_once()
    assert bad.calls == 1 and len(good.sent) == 30       # one try per pass, other sink unaffected
    clock.t += 100
    bad.calls = 0
    d.submit("cam", ev("fall"))
    d.drain_once()
    assert bad.calls == 1
    assert [a["kind"] for a in good.sent].count("fall") == 1
    d.close()


def test_main_alerts_only_for_live_unless_include_non_live(tmp_path, monkeypatch):
    from carelens import pipeline as P
    import carelens.pose
    h = Hook([])
    try:
        for flag, expect in ((False, 0), (True, 1)):
            (tmp_path / "out").mkdir(exist_ok=True)
            (tmp_path / "config.yaml").write_text(f"""
cameras: [{{id: demo, room: kitchen, type: file, source: x.mp4}}]
storage: {{out_dir: out}}
alerts:
  enabled: true
  include_non_live: {str(flag).lower()}
  dedupe_window_s: 0
  sinks: [{{type: webhook, url_env: HOOK_URL}}]
""")
            monkeypatch.setenv("HOOK_URL", h.url + "/h")

            class Pose:
                def __init__(self, *a, **k): pass
                def close(self): pass

            class Src:
                fps = 10.0
                def __init__(self, *a, **k): pass

            def fake_run(source, pose, out, cfg, **kw):
                if kw["on_event"]:
                    kw["on_event"](ev("fall"))
                return {"events": [], "heart_rate": {"summary": {}}, "breathing_rate": {"summary": {}},
                        "gait_segments": []}
            monkeypatch.setattr(carelens.pose, "MediaPipePose", Pose)
            monkeypatch.setattr(P, "VideoSource", Src)
            monkeypatch.setattr(P, "run", fake_run)
            before = len(h.requests)
            P.main(["--camera", "demo", "--config", str(tmp_path / "config.yaml")])
            assert len(h.requests) - before == expect
        assert h.requests[-1][2]["title"].startswith("[TEST] ")
        assert h.requests[-1][2]["data_label"] == "test"
    finally:
        h.stop()
