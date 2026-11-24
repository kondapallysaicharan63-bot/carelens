import pytest

from carelens.store import Store
from dashboard.app import create_app
from tests.test_store import make_results


@pytest.fixture
def env(tmp_path):
    out = tmp_path / "out"
    (out / "clips").mkdir(parents=True)
    (out / "clips" / "a.mp4").write_bytes(b"vid")
    (tmp_path / "secret.txt").write_text("secret")
    with Store(out / "carelens.db") as st:
        rid = st.save_run(make_results(), name="a")
    return out, rid, create_app(str(out)).test_client()


def test_runs_and_run(env):
    _, rid, c = env
    runs = c.get("/api/runs").get_json()
    assert runs[0]["id"] == rid and runs[0]["name"] == "a"
    r = c.get(f"/api/run/{rid}").get_json()
    assert r["id"] == rid and r["events"][0]["kind"] == "fall" and "banner" in r
    assert c.get("/api/run/9999").status_code == 404


def test_history_and_rollups(env):
    _, _, c = env
    h = c.get("/api/history").get_json()
    assert h[0]["clip"] == "a" and h[0]["n_falls"] == 1
    assert c.get("/api/rollups").get_json()[0]["n_runs"] == 1


def test_empty_when_no_db(tmp_path):
    c = create_app(str(tmp_path)).test_client()
    for url in ("/api/runs", "/api/history", "/api/rollups"):
        assert c.get(url).get_json() == []
    assert c.get("/api/run/1").status_code == 404
    assert not (tmp_path / "carelens.db").exists()  # reading never creates the DB


def test_clips_and_traversal(env):
    _, _, c = env
    assert c.get("/clips/a.mp4").data == b"vid"
    for url in ("/clips/../../secret.txt", "/clips/%2e%2e/%2e%2e/secret.txt", "/clips/..%2fcarelens.db"):
        assert c.get(url).status_code in (400, 404)


def test_relative_db_resolves_against_out(tmp_path):
    out = tmp_path / "out"
    with Store(out / "other.db") as st:
        st.save_run(make_results(), name="a")
    c = create_app(str(out), "other.db").test_client()
    assert len(c.get("/api/runs").get_json()) == 1


# ---- alerts API -----------------------------------------------------------------------------
def _alert_env(tmp_path, monkeypatch, token=None):
    if token is None:
        monkeypatch.delenv("CARELENS_DASH_TOKEN", raising=False)
    else:
        monkeypatch.setenv("CARELENS_DASH_TOKEN", token)
    out = tmp_path / "out"
    with Store(out / "carelens.db") as st:
        a = st.add_alert("cam", "fall", {"clip": "x.mp4"}, 1000.0, "webhook", "sent")
        i = st.add_alert("cam", "inactivity", {}, 2000.0, "webhook", "failed", last_error="HTTP 500")
    return out, a, i, create_app(str(out)).test_client()


def _db_alert(out, aid):
    with Store(out / "carelens.db", readonly=True) as st:
        return next(g for g in st.list_alerts() if g["id"] == aid)


def test_alerts_get_and_filters(tmp_path, monkeypatch):
    _, a, i, c = _alert_env(tmp_path, monkeypatch)
    d = c.get("/api/alerts").get_json()
    assert d["unacked"] == 2 and [x["kind"] for x in d["alerts"]] == ["inactivity", "fall"]
    assert [x["id"] for x in c.get("/api/alerts?status=failed").get_json()["alerts"]] == [i]
    assert [x["id"] for x in c.get("/api/alerts?since=1500").get_json()["alerts"]] == [i]
    assert c.get("/api/alerts?status=bogus").status_code == 400
    assert c.get("/api/alerts?since=abc").status_code == 400
    assert create_app(str(tmp_path / "none")).test_client().get("/api/alerts").get_json()["alerts"] == []


def test_alert_writes_loopback_no_token(tmp_path, monkeypatch):
    out, a, i, c = _alert_env(tmp_path, monkeypatch)
    assert c.post(f"/api/alerts/{a}/ack").status_code == 200      # test client is 127.0.0.1
    assert _db_alert(out, a)["acknowledged"]
    assert c.post(f"/api/alerts/{a}/false-alarm").status_code == 200
    d = c.get("/api/alerts").get_json()
    assert d["unacked"] == 1 and sum(d["false_alarms_per_day"].values()) == 1
    assert c.post("/api/alerts/999/ack").status_code == 404
    assert c.get(f"/api/alerts/{a}/ack").status_code == 405


def test_alert_writes_403_from_non_loopback_without_token(tmp_path, monkeypatch):
    out, a, _, c = _alert_env(tmp_path, monkeypatch)
    env = {"REMOTE_ADDR": "192.168.1.9"}
    for act in ("ack", "false-alarm"):
        assert c.post(f"/api/alerts/{a}/{act}", environ_overrides=env).status_code == 403
    assert not _db_alert(out, a)["acknowledged"]


def test_alert_writes_with_token(tmp_path, monkeypatch):
    out, a, _, c = _alert_env(tmp_path, monkeypatch, token="s3cret")
    env = {"REMOTE_ADDR": "192.168.1.9"}
    assert c.post(f"/api/alerts/{a}/ack", environ_overrides=env).status_code == 403
    assert c.post(f"/api/alerts/{a}/ack", environ_overrides=env,
                  headers={"X-CareLens-Token": "wrong"}).status_code == 403
    assert c.post(f"/api/alerts/{a}/ack").status_code == 403       # token set: loopback is not enough
    assert c.post(f"/api/alerts/{a}/ack", environ_overrides=env,
                  headers={"X-CareLens-Token": "s3cret"}).status_code == 200
    assert _db_alert(out, a)["acknowledged"]
    assert c.post(f"/api/alerts/{a}/ack", headers={"X-CareLens-Token": "s3cret",
                                                  "Origin": "http://evil.example"}).status_code == 403


def test_alert_write_csrf_origin_rejected(tmp_path, monkeypatch):
    out, a, _, c = _alert_env(tmp_path, monkeypatch)
    assert c.post(f"/api/alerts/{a}/ack", headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.post(f"/api/alerts/{a}/ack", headers={"Origin": "null"}).status_code == 403
    assert not _db_alert(out, a)["acknowledged"]
    assert c.post(f"/api/alerts/{a}/ack", headers={"Origin": "http://localhost:5000"}).status_code == 200


def test_bind_warning():
    from dashboard.app import bind_warning
    assert bind_warning("127.0.0.1", False) is None and bind_warning("0.0.0.0", True) is None
    assert "CARELENS_DASH_TOKEN" in bind_warning("0.0.0.0", False)


def test_index_has_banner_and_footer():
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "dashboard" / "static" / "index.html").read_text()
    for s in ("Acknowledge", "Not a fall", "Wellness aid &mdash; not for emergencies"):
        assert s in html


def test_token_read_from_dotenv_and_env_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("CARELENS_DASH_TOKEN", raising=False)
    out = tmp_path / "out"
    with Store(out / "carelens.db") as st:
        a = st.add_alert("cam", "fall", {}, 1000.0, "w", "sent")
    (tmp_path / ".env").write_text("CARELENS_DASH_TOKEN=fromfile\n")
    c = create_app(str(out), env_dir=tmp_path).test_client()
    assert c.post(f"/api/alerts/{a}/ack").status_code == 403          # loopback no longer enough
    assert c.post(f"/api/alerts/{a}/ack", headers={"X-CareLens-Token": "fromfile"}).status_code == 200
    from dashboard.app import dash_token
    assert dash_token(tmp_path) == "fromfile"
    monkeypatch.setenv("CARELENS_DASH_TOKEN", "fromenv")
    assert dash_token(tmp_path) == "fromenv"


def test_proxy_origin_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("CARELENS_DASH_TOKEN", "t")
    monkeypatch.delenv("CARELENS_DASH_ORIGINS", raising=False)
    out = tmp_path / "out"
    with Store(out / "carelens.db") as st:
        a = st.add_alert("cam", "fall", {}, 1000.0, "w", "sent")
    h = {"X-CareLens-Token": "t", "Origin": "https://carelens.example.org",
         "X-Forwarded-Host": "carelens.example.org"}
    assert create_app(str(out)).test_client().post(f"/api/alerts/{a}/ack", headers=h).status_code == 403
    c = create_app(str(out), allowed_origins=["https://carelens.example.org/"]).test_client()
    assert c.post(f"/api/alerts/{a}/ack", headers=h).status_code == 200
    monkeypatch.setenv("CARELENS_DASH_ORIGINS", "https://other.example.org, https://x.example")
    h["Origin"] = "https://other.example.org"
    assert create_app(str(out)).test_client().post(f"/api/alerts/{a}/ack", headers=h).status_code == 200


def test_alerts_api_on_unmigrated_db_and_unacked_beyond_limit(tmp_path, monkeypatch):
    import sqlite3
    from carelens import store as S
    out = tmp_path / "out"; out.mkdir()
    db = sqlite3.connect(out / "carelens.db", isolation_level=None)
    for s in S.MIGRATIONS[0]:
        db.execute(s)
    db.execute("PRAGMA user_version=1"); db.close()
    c = create_app(str(out)).test_client()
    r = c.get("/api/alerts")
    assert r.status_code == 200 and r.get_json() == {"alerts": [], "unacked": 0, "false_alarms_per_day": {}}
    out2 = tmp_path / "o2"
    with Store(out2 / "carelens.db") as st:
        for i in range(205):
            st.add_alert("cam", "fall", {}, 1000.0 + i, "w", "sent")
    d = create_app(str(out2)).test_client().get("/api/alerts").get_json()
    assert len(d["alerts"]) == 200 and d["unacked"] == 205


# ---- daily trends (F14) ---------------------------------------------------------------------
def _seed_rollups(out):
    with Store(out / "carelens.db") as st:
        for cam, day in (("cam1", "2026-11-01"), ("cam1", "2026-11-05"), ("cam2", "2026-11-05")):
            st.db.execute("INSERT INTO daily_rollups VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                          (day, cam, 1, 70.0, 15.0, 0.9, 0, 1, 1.0, 1.0, 1.0, 0, 0.0))
        st.db.commit()


def test_rollups_filter_by_camera_and_since_day(tmp_path):
    out = tmp_path / "out"
    _seed_rollups(out)
    c = create_app(str(out)).test_client()
    assert len(c.get("/api/rollups").get_json()) == 3
    assert {r["camera_id"] for r in c.get("/api/rollups?camera_id=cam1").get_json()} == {"cam1"}
    assert [r["day"] for r in c.get("/api/rollups?since_day=2026-11-05").get_json()] == ["2026-11-05"] * 2
    both = c.get("/api/rollups?camera_id=cam2&since_day=2026-11-02").get_json()
    assert [(r["camera_id"], r["day"]) for r in both] == [("cam2", "2026-11-05")]
    assert c.get("/api/rollups?since_day=notaday").status_code == 400
    assert c.get("/api/rollups?since_day=20261101").status_code == 400


def test_index_is_local_only_and_has_trend_section():
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "dashboard" / "static" / "index.html").read_text()
    assert not re.search(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", html, re.I)
    assert not re.search(r"@import|url\(\s*[\"']?https?:", html, re.I)
    assert 'id="trends"' in html and 'id="trtable"' in html
    assert "Not enough data for trends yet \u2014 trends need several days of recordings." in html
    assert "Wellness trends &mdash; not a diagnosis. A sustained change may be worth a check-in." in html
    for bad in ("diagnos", "emergency detected", "fall confirmed", "confirmed fall"):
        # the only permitted use of the stem is the negated disclaimer
        assert all("not a diagnosis" in m for m in re.findall(r"[^.\n]*" + bad + r"[^.\n]*", html, re.I)), bad
