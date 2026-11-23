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
