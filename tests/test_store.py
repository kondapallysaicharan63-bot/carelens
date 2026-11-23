import copy
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from carelens import store as S
from carelens.store import Store, open_db

DAY = 86400.0


def make_results(start="2026-03-10T12:00:00", label="test", n_falls=1, hr=72.0, gait=1.1):
    ev = [{"kind": "fall", "t_onset": 10.0, "t_confirmed": 30.0, "message": "Possible fall - check-in suggested",
           "details": {"drop": 0.4}, "clip": "r_fall_10s.mp4"}] * n_falls
    ev = [dict(e) for e in ev] + [{"kind": "inactivity", "t_onset": 40.0, "t_confirmed": 70.0,
                                   "message": "Inactivity - check-in suggested", "details": {}, "clip": None}]
    win = lambda v: [{"t": 5.0, "value": v, "quality": 0.8, "reported": True, "snr": 3.2},
                     {"t": 10.0, "value": v + 50, "quality": 0.1, "reported": False}]
    return {
        "banner": "Simulated or test data - not a medical device", "data_label": label,
        "source": "rtsp://cam/stream", "start_time": start, "duration_s": 90.0,
        "config": {"fall_still_seconds": 20.0}, "events": ev,
        "timeline_by_hour": {"2026-03-10 12:00": {"sitting": 10.0, "standing": 5.0, "walking": 20.0,
                                                   "lying": 30.0, "absent": 25.0}},
        "posture_track": [{"t": 0.0, "posture": "sitting", "motion": 0.1},
                          {"t": 1.0, "posture": "lying", "motion": 0.0}],
        "gait_segments": [{"t_start": 3.0, "duration_s": 4.0, "speed_m_s_approx": gait,
                           "speed_cv": 0.2, "trunk_sway_deg": 5.0}],
        "heart_rate": {"unit": "bpm", "summary": {"median": hr, "n_reported": 1, "n_windows": 2,
                                                    "mean_quality": 0.8}, "windows": win(hr)},
        "breathing_rate": {"unit": "breaths/min", "summary": {"median": 14.0, "n_reported": 1,
                           "n_windows": 2, "mean_quality": 0.8}, "windows": win(14.0)},
    }


@pytest.fixture
def st(tmp_path):
    with Store(tmp_path / "t.db") as s:
        yield s


def ts_of(iso):
    return datetime.fromisoformat(iso).astimezone().timestamp()


# ---- migrations ---------------------------------------------------------------------------
def test_migrate_from_zero_and_reopen(tmp_path):
    p = tmp_path / "m.db"
    db = open_db(p)
    assert db.execute("PRAGMA user_version").fetchone()[0] == len(S.MIGRATIONS)
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    db.execute("INSERT INTO runs (name,data_label,start_ts,created_ts) VALUES ('a','test',1,1)")
    db.close()
    db = open_db(p)  # idempotent, data kept
    assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    db.close()


def test_newer_database_raises(tmp_path):
    p = tmp_path / "n.db"
    c = sqlite3.connect(p)
    c.execute(f"PRAGMA user_version={len(S.MIGRATIONS) + 1}")
    c.close()
    with pytest.raises(RuntimeError, match="newer than this app"):
        open_db(p)


def test_failed_migration_rolls_back(tmp_path, monkeypatch):
    p = tmp_path / "f.db"
    open_db(p).close()
    bad = [["CREATE TABLE ok_table (x)", "THIS IS NOT SQL"]]
    monkeypatch.setattr(S, "MIGRATIONS", S.MIGRATIONS + bad)
    with pytest.raises(sqlite3.Error):
        open_db(p)
    c = sqlite3.connect(p)
    assert c.execute("PRAGMA user_version").fetchone()[0] == len(S.MIGRATIONS) - 1
    assert c.execute("SELECT name FROM sqlite_master WHERE name='ok_table'").fetchone() is None
    c.close()


def test_data_label_check_and_cascade(st):
    rid = st.save_run(make_results())
    with pytest.raises(sqlite3.IntegrityError):
        st.db.execute("INSERT INTO runs (name,data_label,start_ts,created_ts) VALUES ('x','bogus',1,1)")
    st.db.execute("DELETE FROM runs WHERE id=?", (rid,))
    for t in ("events", "vitals_windows", "posture_track", "timeline_hours", "gait_segments"):
        assert st.db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] == 0, t


# ---- save / get ---------------------------------------------------------------------------
def test_round_trip(st):
    r = make_results()
    out = st.get_run(st.save_run(r, camera_id="cam1", source_kind="rtsp", name="r"))
    assert out.pop("id")
    assert out == r  # same shape as the old results JSON, incl. extra window keys
    row = st.db.execute("SELECT * FROM events WHERE kind='fall'").fetchone()
    assert row["ts"] == pytest.approx(ts_of(r["start_time"]) + 10.0)


def test_resave_replaces(st):
    r = make_results()
    st.save_run(r, name="r")
    st.save_run(r, name="r")
    assert st.db.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert st.db.execute("SELECT count(*) FROM events").fetchone()[0] == 2
    assert st.daily_rollups()[0]["n_runs"] == 1
    assert st.get_run(999) is None


def test_list_and_history(st):
    a = st.save_run(make_results("2026-03-10T08:00:00", hr=70), name="a")
    b = st.save_run(make_results("2026-03-10T09:00:00", hr=80, n_falls=0), name="b", camera_id="c2")
    assert [r["id"] for r in st.list_runs()] == [b, a]  # newest first
    assert set(st.list_runs()[0]) == {"id", "name", "camera_id", "start_time", "data_label"}
    h = st.history()
    assert [x["clip"] for x in h] == ["a", "b"]
    assert {"start_time", "clip", "data_label", "hr_median", "br_median", "gait_speed",
            "n_falls", "n_inactivity", "id", "camera_id"} == set(h[0])
    assert h[0]["start_time"] == "2026-03-10T08:00:00" and h[0]["n_falls"] == 1
    assert h[1]["n_falls"] == 0 and h[1]["gait_speed"] == 1.1
    assert [x["clip"] for x in st.history(camera_id="c2")] == ["b"]
    assert [x["clip"] for x in st.history(since_ts=ts_of("2026-03-10T08:30:00"))] == ["b"]


# ---- rollups ------------------------------------------------------------------------------
def test_rollup_aggregates_and_idempotent(st):
    st.save_run(make_results("2026-03-10T08:00:00", hr=70, gait=1.0, label="live"), name="a")
    st.save_run(make_results("2026-03-10T20:00:00", hr=80, gait=1.4), name="b")
    (d,) = st.daily_rollups()
    assert d["day"] == "2026-03-10" and d["n_runs"] == 2
    assert d["hr_median"] == 75.0 and d["gait_speed"] == 1.2
    assert d["n_falls"] == 2 and d["n_inactivity"] == 2
    assert d["walk_s"] == 40 and d["lying_s"] == 60 and d["absent_s"] == 50
    assert d["any_simulated"] == 1
    st.save_run(make_results("2026-03-10T20:00:00", hr=80, gait=1.4), name="b")
    assert len(st.daily_rollups()) == 1 and st.daily_rollups()[0]["n_runs"] == 2
    assert st.daily_rollups(since_day="2026-03-11") == []
    assert st.daily_rollups(camera_id="nope") == []


def test_rollup_local_day_uses_tz_offset(st):
    # 23:30 at UTC-05:00 is already the next day in UTC; the local day must win.
    r = make_results("2026-03-10T23:30:00-05:00")
    rid = st.save_run(r, name="tz")
    assert st.db.execute("SELECT tz_offset_min FROM runs").fetchone()[0] == -300
    assert st.daily_rollups()[0]["day"] == "2026-03-10"
    assert st.get_run(rid)["start_time"] == "2026-03-10T23:30:00"


# ---- purge --------------------------------------------------------------------------------
def _setup_files(tmp_path, name, clip="r_fall_10s.mp4"):
    out = tmp_path / "out"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "clips" / clip).write_bytes(b"x")
    (out / f"{name}_results.json").write_text("{}")
    return out


def test_purge_old_keeps_fresh_and_rollups(tmp_path):
    out = _setup_files(tmp_path, "old")
    (out / "fresh_results.json").write_text("{}")
    now = ts_of("2026-03-20T00:00:00")
    with Store(tmp_path / "p.db", clock=lambda: now) as st:
        st.save_run(make_results("2026-03-10T12:00:00"), name="old")
        fresh = make_results("2026-03-19T12:00:00")
        fresh["events"][0]["clip"] = "fresh_fall.mp4"
        st.save_run(fresh, name="fresh")
        res = st.purge(retention_days=7, clips_dir=out / "clips")
        assert res["runs"] == 1 and res["files"] == 2
        assert not (out / "clips" / "r_fall_10s.mp4").exists()
        assert not (out / "old_results.json").exists() and (out / "fresh_results.json").exists()
        assert [r["name"] for r in st.list_runs()] == ["fresh"]
        assert st.db.execute("SELECT count(*) FROM events").fetchone()[0] == 2  # only fresh's
        assert {d["day"] for d in st.daily_rollups()} == {"2026-03-10", "2026-03-19"}


def test_purge_missing_clip_and_outside_path_guard(tmp_path):
    out = _setup_files(tmp_path, "old")
    secret = tmp_path / "secret.txt"
    secret.write_text("keep")
    (out / "clips" / "r_fall_10s.mp4").unlink()  # missing clip is tolerated
    r = make_results("2026-03-10T12:00:00")
    r["events"][0]["clip"] = "../../secret.txt"
    now = ts_of("2026-03-30T00:00:00")
    with Store(tmp_path / "p.db", clock=lambda: now) as st:
        st.save_run(r, name="old")
        res = st.purge(clips_dir=out / "clips")
    assert res["runs"] == 1 and secret.exists()
    assert not (out / "old_results.json").exists()


def test_purge_boundary_is_whole_local_days(tmp_path):
    out = _setup_files(tmp_path, "edge")
    start = ts_of("2026-03-10T12:00:00+00:00")
    with Store(tmp_path / "p.db") as st:
        st.save_run(make_results("2026-03-10T12:00:00+00:00"), name="edge")
        # cutoff still on the run's local day (03-10) -> kept, even 11h later
        assert st.purge(now=start + 7 * DAY + 11 * 3600, clips_dir=out / "clips")["runs"] == 0
        # cutoff reaches 03-11 -> the whole day 03-10 goes
        assert st.purge(now=start + 7 * DAY + 13 * 3600, clips_dir=out / "clips")["runs"] == 1


def test_rollup_frozen_after_purge_and_backfill(st):
    st.save_run(make_results("2026-03-10T08:00:00+00:00", n_falls=1), name="a")
    st.save_run(make_results("2026-03-10T14:00:00+00:00", n_falls=0), name="b")
    now = ts_of("2026-03-30T00:00:00+00:00")
    assert st.purge(now=now)["runs"] == 2
    (d,) = st.daily_rollups()
    assert (d["n_runs"], d["n_falls"], d["frozen"]) == (2, 1, 1)
    st.purge(now=now + DAY)
    st.save_run(make_results("2026-03-10T16:00:00+00:00", n_falls=0), name="c")  # backfill
    (d2,) = st.daily_rollups()
    assert (d2["n_runs"], d2["n_falls"]) == (2, 1)


def test_purge_keeps_files_shared_with_surviving_run(tmp_path):
    out = _setup_files(tmp_path, "cam")
    shared = str(out / "cam_results.json")
    now = ts_of("2026-03-20T00:00:00+00:00")
    with Store(tmp_path / "p.db", clock=lambda: now) as st:
        st.save_run(make_results("2026-03-01T12:00:00+00:00"), name="cam", export_path=shared)
        st.save_run(make_results("2026-03-19T12:00:00+00:00"), name="cam", export_path=shared)
        res = st.purge(clips_dir=out / "clips")  # both runs reference the same clip + export
        assert res["runs"] == 1 and res["files"] == 0
        assert (out / "cam_results.json").exists() and (out / "clips" / "r_fall_10s.mp4").exists()


def test_purge_exclude_run_ids(tmp_path):
    out = _setup_files(tmp_path, "old")
    now = ts_of("2026-03-30T00:00:00+00:00")
    with Store(tmp_path / "p.db", clock=lambda: now) as st:
        rid = st.save_run(make_results("2026-03-10T12:00:00+00:00"), name="old")
        res = st.purge(clips_dir=out / "clips", exclude_run_ids=[rid])
        assert res["runs"] == 0 and res["kept_excluded"] == 1 and (out / "old_results.json").exists()
        assert st.purge(clips_dir=out / "clips")["runs"] == 1


def test_import_legacy_idempotent(tmp_path):
    import json
    out = tmp_path / "out"
    out.mkdir()
    (out / "demo_results.json").write_text(json.dumps(make_results("2026-03-10T12:00:00")))
    (out / "history.jsonl").write_text(json.dumps({"clip": "demo"}) + "\n" + json.dumps({"clip": "gone"}) + "\n")
    with Store(out / "carelens.db") as st:
        assert st.import_legacy(out) == {"imported": 1, "failed": 0, "history_rows_skipped": 1}
        st.import_legacy(out)
        assert [r["name"] for r in st.list_runs()] == ["demo"]
    S.main(["import-legacy", "--out", str(out)])


def test_rollup_retention(tmp_path):
    with Store(tmp_path / "p.db") as st:
        st.save_run(make_results("2026-03-10T12:00:00"), name="a")
        base = ts_of("2026-03-10T12:00:00")
        res = st.purge(now=base + 30 * DAY, retention_days=7, rollup_retention_days=365)
        assert res["runs"] == 1 and res["rollups"] == 0 and len(st.daily_rollups()) == 1
        res = st.purge(now=base + 400 * DAY, retention_days=7, rollup_retention_days=365)
        assert res["rollups"] == 1 and st.daily_rollups() == []


def test_cli_purge(tmp_path, capsys):
    out = tmp_path / "o"
    with Store(out / "carelens.db") as st:
        st.save_run(make_results("2020-01-01T00:00:00"), name="old")
    S.main(["purge", "--out", str(out), "--days", "7"])
    assert '"runs": 1' in capsys.readouterr().out
