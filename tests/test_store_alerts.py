import sqlite3
from datetime import datetime, timezone

import pytest

from carelens import store as S
from carelens.store import Store, open_db

DAY = 86400.0


@pytest.fixture
def st(tmp_path):
    with Store(tmp_path / "a.db") as s:
        yield s


def test_migration_1_to_2_preserves_data(tmp_path):
    p = tmp_path / "old.db"
    db = sqlite3.connect(p, isolation_level=None)
    for stmt in S.MIGRATIONS[0]:
        db.execute(stmt)
    db.execute("PRAGMA user_version=1")
    db.execute("INSERT INTO runs (name,data_label,start_ts,created_ts) VALUES ('keep','test',1,1)")
    db.close()
    db = open_db(p)
    assert db.execute("PRAGMA user_version").fetchone()[0] == len(S.MIGRATIONS) >= 2
    assert db.execute("SELECT name FROM runs").fetchone()[0] == "keep"
    assert db.execute("SELECT count(*) FROM alerts").fetchone()[0] == 0
    db.close()


def test_status_check_and_run_fk_set_null(st):
    with pytest.raises(sqlite3.IntegrityError):
        st.add_alert("c", "fall", {}, 1.0, "w", "bogus")
    st.db.execute("INSERT INTO runs (id,name,data_label,start_ts,created_ts) VALUES (5,'r','test',1,1)")
    aid = st.add_alert("c", "fall", {}, 1.0, "w", "pending", run_id=5)
    st.db.execute("DELETE FROM runs WHERE id=5")
    assert st.db.execute("SELECT run_id FROM alerts WHERE id=?", (aid,)).fetchone()[0] is None


def test_due_mark_and_claim(st):
    a = st.add_alert("c", "fall", {"x": 1}, 100.0, "w", "pending", next_attempt_ts=100.0)
    b = st.add_alert("c", "fall", {}, 100.0, "w2", "pending", next_attempt_ts=500.0)
    assert [r["id"] for r in st.due_alerts(200.0)] == [a]
    assert st.claim_alert(a, 200.0, 60) and not st.claim_alert(a, 200.0, 60)   # leased
    assert st.due_alerts(200.0) == [] and [r["id"] for r in st.due_alerts(300.0)] == [a]
    st.mark_alert(a, "sent", 1)
    assert st.due_alerts(1e9)[0]["id"] == b


def test_list_groups_across_sinks_and_ack(st):
    a1 = st.add_alert("c", "fall", {"clip": "x"}, 100.0, "webhook", "sent")
    st.add_alert("c", "fall", {"clip": "x"}, 100.0, "desktop", "failed", last_error="boom")
    st.add_alert("c", "inactivity", {}, 200.0, "", "suppressed")
    g = st.list_alerts()
    assert [x["kind"] for x in g] == ["inactivity", "fall"]           # newest first
    fall = g[1]
    assert fall["status"] == "sent" and len(fall["sinks"]) == 2 and not fall["acknowledged"]
    assert [x["kind"] for x in st.list_alerts(status="suppressed")] == ["inactivity"]
    assert [x["kind"] for x in st.list_alerts(since_ts=150.0)] == ["inactivity"]
    assert st.ack_alert(a1, now=300.0) and not st.ack_alert(999)
    assert st.list_alerts()[1]["acknowledged"] is True
    assert st.list_alerts()[0]["acknowledged"] is False                # other alert untouched


def test_false_alarms_per_day_local_days(st, monkeypatch):
    monkeypatch.setattr(S, "_local_offset_min", lambda ts: -300)       # UTC-5
    t = lambda h, d=10: datetime(2026, 3, d, h, tzinfo=timezone.utc).timestamp()
    ids = [st.add_alert("c", "fall", {}, t(2), "w", "sent"),          # 21:00 local on the 9th
           st.add_alert("c", "fall", {}, t(2), "w2", "sent"),         # same alert, 2nd sink
           st.add_alert("c", "fall", {}, t(12), "w", "sent"),         # 10th
           st.add_alert("c", "fall", {}, t(14), "w", "sent")]         # 10th
    st.add_alert("c", "fall", {}, t(15), "w", "sent")                 # no feedback
    for i in (ids[0], ids[2], ids[3]):
        assert st.mark_false_alarm(i, now=t(20))
    assert st.false_alarms_per_day() == {"2026-03-09": 1, "2026-03-10": 2}
    assert st.false_alarms_per_day(since_ts=t(10)) == {"2026-03-10": 2}
    assert all(g["acknowledged"] for g in st.list_alerts() if g["false_alarm"])


def test_purge_drops_old_alerts(st):
    now = 1000 * DAY
    st.add_alert("c", "fall", {}, now - 400 * DAY, "w", "sent")
    st.add_alert("c", "fall", {}, now - 10 * DAY, "w", "sent")
    st.purge(now=now, rollup_retention_days=365)
    assert [g["created_ts"] for g in st.list_alerts()] == [now - 10 * DAY]


def test_unacked_count_ignores_list_limit(st):
    for i in range(7):
        st.add_alert("c", "fall", {}, 100.0 + i, "w", "sent")
    assert len(st.list_alerts(limit=3)) == 3 and st.unacked_count() == 7
    st.ack_alert(st.list_alerts()[0]["id"])
    assert st.unacked_count() == 6 and st.unacked_count(since_ts=105.0) == 1


def test_due_order_fewest_attempts_then_newest(st):
    a = st.add_alert("c", "fall", {}, 100.0, "w", "pending", next_attempt_ts=0)
    st.mark_alert(a, "pending", attempts=3, next_attempt_ts=0)
    old = st.add_alert("c", "fall", {}, 101.0, "w", "pending", next_attempt_ts=0)
    new = st.add_alert("c", "fall", {}, 102.0, "w", "pending", next_attempt_ts=0)
    assert [r["id"] for r in st.due_alerts(10)] == [new, old, a]
    assert [r["id"] for r in st.due_alerts(10, exclude_sinks=["w"])] == []
