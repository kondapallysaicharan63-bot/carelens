"""SQLite store (stdlib only): runs, events, vitals, posture/timeline/gait, daily rollups.

Local-first: one file under the output dir. Instants are epoch seconds (UTC); each run keeps
its local UTC offset so daily rollups are bucketed by the *local* calendar day. Daily rollups
have no foreign key to runs, so they survive the raw-data purge (the privacy-preserving trend
record), and are pruned on their own, longer retention.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

log = logging.getLogger(__name__)

# Each migration is a list of statements (no executescript: it would auto-commit and
# defeat the transaction that makes a failed migration roll back).
MIGRATIONS: list[list[str]] = [
    [
        """CREATE TABLE runs (
            id INTEGER PRIMARY KEY,
            camera_id TEXT NOT NULL DEFAULT 'default',
            name TEXT NOT NULL,
            source TEXT,
            source_kind TEXT,
            export_path TEXT,
            data_label TEXT NOT NULL CHECK (data_label IN ('test','simulated','live')),
            simulated INTEGER NOT NULL DEFAULT 1,
            start_ts REAL NOT NULL,
            tz_offset_min INTEGER NOT NULL DEFAULT 0,
            duration_s REAL,
            config_json TEXT,
            summary_json TEXT,
            created_ts REAL NOT NULL,
            UNIQUE (camera_id, name, start_ts))""",
        """CREATE TABLE events (
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            kind TEXT NOT NULL, ts REAL NOT NULL, t_onset REAL, t_confirmed REAL,
            message TEXT, details_json TEXT, clip_path TEXT)""",
        """CREATE TABLE vitals_windows (
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('hr','br')),
            t REAL NOT NULL, ts REAL NOT NULL, value REAL, quality REAL,
            reported INTEGER NOT NULL DEFAULT 0, extra_json TEXT)""",
        """CREATE TABLE posture_track (
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            t REAL NOT NULL, posture TEXT, motion REAL, PRIMARY KEY (run_id, t))""",
        """CREATE TABLE timeline_hours (
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            hour_label TEXT NOT NULL,
            sitting REAL, standing REAL, walking REAL, lying REAL, absent REAL,
            PRIMARY KEY (run_id, hour_label))""",
        """CREATE TABLE gait_segments (
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            t_start REAL NOT NULL, ts REAL NOT NULL, duration_s REAL,
            speed_m_s_approx REAL, speed_cv REAL, trunk_sway_deg REAL)""",
        """CREATE TABLE daily_rollups (
            day TEXT NOT NULL, camera_id TEXT NOT NULL,
            n_runs INTEGER, hr_median REAL, br_median REAL, gait_speed REAL,
            n_falls INTEGER, n_inactivity INTEGER,
            walk_s REAL, lying_s REAL, absent_s REAL,
            any_simulated INTEGER, updated_ts REAL,
            frozen INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (day, camera_id))""",
        "CREATE INDEX idx_runs_start ON runs(start_ts)",
        "CREATE INDEX idx_events_run ON events(run_id)",
        "CREATE INDEX idx_events_ts ON events(ts)",
        "CREATE INDEX idx_vitals_run ON vitals_windows(run_id)",
        "CREATE INDEX idx_vitals_ts ON vitals_windows(ts)",
        "CREATE INDEX idx_gait_run ON gait_segments(run_id)",
        "CREATE INDEX idx_gait_ts ON gait_segments(ts)",
    ],
    [
        """CREATE TABLE alerts (
            id INTEGER PRIMARY KEY,
            camera_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL,
            event_json TEXT,
            created_ts REAL NOT NULL,
            sink TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK (status IN ('pending','sent','failed','suppressed')),
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_ts REAL,
            last_error TEXT,
            acknowledged_ts REAL,
            false_alarm INTEGER NOT NULL DEFAULT 0,
            feedback_ts REAL)""",
        "CREATE INDEX idx_alerts_due ON alerts(status, next_attempt_ts)",
        "CREATE INDEX idx_alerts_dedupe ON alerts(camera_id, kind, created_ts)",
    ],
]

_POSTURES = ("sitting", "standing", "walking", "lying", "absent")
_VITALS = (("hr", "heart_rate", "bpm"), ("br", "breathing_rate", "breaths/min"))
_WINDOW_KEYS = {"t", "value", "quality", "reported"}


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the database and apply pending migrations."""
    db = sqlite3.connect(str(path), isolation_level=None)
    try:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = sqlite3.Row
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version > len(MIGRATIONS):
            raise RuntimeError(f"database is newer than this app (schema v{version}, "
                               f"app supports v{len(MIGRATIONS)})")
        for i in range(version, len(MIGRATIONS)):
            db.execute("BEGIN IMMEDIATE")
            try:
                for stmt in MIGRATIONS[i]:
                    db.execute(stmt)
                db.execute(f"PRAGMA user_version={i + 1}")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
    except BaseException:
        db.close()
        raise
    return db


def _open_readonly(path: str | Path) -> sqlite3.Connection:
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True,
                         isolation_level=None)
    db.row_factory = sqlite3.Row
    return db


def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(float(statistics.median(vals)), 2) if vals else None


def _local_iso(ts: float, tz_offset_min: int) -> str:
    tz = timezone(timedelta(minutes=tz_offset_min))
    return datetime.fromtimestamp(ts, tz).replace(tzinfo=None).isoformat()


def _day_of(ts: float, tz_offset_min: int) -> str:
    """Local calendar day (YYYY-MM-DD) of an instant, given a local UTC offset in minutes."""
    return datetime.fromtimestamp(ts + tz_offset_min * 60, timezone.utc).strftime("%Y-%m-%d")


def _local_offset_min(ts: float) -> int:
    """This machine's local UTC offset (minutes) at instant ts."""
    return int(datetime.fromtimestamp(ts).astimezone().utcoffset().total_seconds() // 60)


class Store:
    def __init__(self, path: str | Path, clock: Callable[[], float] = time.time,
                 readonly: bool = False):
        self.path, self.clock = Path(path), clock
        if readonly:
            self.db = _open_readonly(self.path)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = open_db(self.path)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def _tx(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    # ---- writes -----------------------------------------------------------------------
    def save_run(self, results: dict, *, camera_id: str = "default",
                 source_kind: str = "file", name: Optional[str] = None,
                 export_path: Optional[str] = None) -> int:
        """Persist one run in a single transaction. `name` defaults to the source file stem."""
        start = datetime.fromisoformat(results["start_time"])
        aware = start if start.tzinfo else start.astimezone()  # naive = local wall clock
        start_ts = round(aware.timestamp(), 6)
        off = int(aware.utcoffset().total_seconds() // 60)
        name = name or self._name_of(results)
        label = results["data_label"]
        events, gait = results["events"], results["gait_segments"]
        summary = {
            "banner": results.get("banner"),
            "heart_rate": {"unit": results["heart_rate"].get("unit", "bpm"),
                           "summary": results["heart_rate"]["summary"]},
            "breathing_rate": {"unit": results["breathing_rate"].get("unit", "breaths/min"),
                               "summary": results["breathing_rate"]["summary"]},
            "hr_median": results["heart_rate"]["summary"].get("median"),
            "br_median": results["breathing_rate"]["summary"].get("median"),
            "gait_speed": _median([g["speed_m_s_approx"] for g in gait]),
            "n_falls": sum(e["kind"] == "fall" for e in events),
            "n_inactivity": sum(e["kind"] == "inactivity" for e in events),
        }
        with self._tx() as db:
            db.execute("DELETE FROM runs WHERE camera_id=? AND name=? AND start_ts=?",
                       (camera_id, name, start_ts))
            cur = db.execute(
                "INSERT INTO runs (camera_id,name,source,source_kind,export_path,data_label,simulated,"
                "start_ts,tz_offset_min,duration_s,config_json,summary_json,created_ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (camera_id, name, results.get("source"), source_kind, export_path, label,
                 int(label != "live"), start_ts, off, results.get("duration_s"),
                 json.dumps(results.get("config", {})), json.dumps(summary), self.clock()))
            rid = cur.lastrowid
            db.executemany(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
                [(rid, e["kind"], start_ts + e["t_onset"], e["t_onset"], e.get("t_confirmed"),
                  e.get("message"), json.dumps(e.get("details", {})), e.get("clip"))
                 for e in events])
            for kind, key, _unit in _VITALS:
                db.executemany(
                    "INSERT INTO vitals_windows VALUES (?,?,?,?,?,?,?,?)",
                    [(rid, kind, w["t"], start_ts + w["t"], w["value"], w["quality"],
                      int(bool(w["reported"])),
                      json.dumps({k: v for k, v in w.items() if k not in _WINDOW_KEYS}))
                     for w in results[key]["windows"]])
            db.executemany(
                "INSERT OR REPLACE INTO posture_track VALUES (?,?,?,?)",
                [(rid, p["t"], p["posture"], p["motion"]) for p in results["posture_track"]])
            db.executemany(
                "INSERT INTO timeline_hours VALUES (?,?,?,?,?,?,?)",
                [(rid, h, *[v.get(p, 0.0) for p in _POSTURES])
                 for h, v in results["timeline_by_hour"].items()])
            db.executemany(
                "INSERT INTO gait_segments VALUES (?,?,?,?,?,?,?)",
                [(rid, g["t_start"], start_ts + g["t_start"], g["duration_s"],
                  g["speed_m_s_approx"], g["speed_cv"], g["trunk_sway_deg"]) for g in gait])
            self._rebuild_rollup(db, _day_of(start_ts, off), camera_id)
        return rid

    @staticmethod
    def _name_of(results: dict) -> str:
        return Path(str(results.get("source", "run"))).stem or "run"

    def _rebuild_rollup(self, db, day: str, camera_id: str):
        runs = db.execute(
            "SELECT id, simulated, summary_json FROM runs WHERE camera_id=? AND "
            "date(start_ts + tz_offset_min*60, 'unixepoch')=?", (camera_id, day)).fetchall()
        if not runs:
            return
        frozen = db.execute("SELECT frozen FROM daily_rollups WHERE day=? AND camera_id=?",
                            (day, camera_id)).fetchone()
        if frozen and frozen["frozen"]:
            log.warning("rollup %s/%s is frozen (its raw runs were purged); not updated",
                        day, camera_id)
            return
        ids = [r["id"] for r in runs]
        q = ",".join("?" * len(ids))
        sums = [json.loads(r["summary_json"]) for r in runs]
        posture = {p: db.execute(
            f"SELECT COALESCE(SUM({p}),0) FROM timeline_hours WHERE run_id IN ({q})",
            ids).fetchone()[0] for p in ("walking", "lying", "absent")}
        db.execute(
            "INSERT OR REPLACE INTO daily_rollups VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
            (day, camera_id, len(runs), _median(s["hr_median"] for s in sums),
             _median(s["br_median"] for s in sums), _median(s["gait_speed"] for s in sums),
             sum(s["n_falls"] for s in sums), sum(s["n_inactivity"] for s in sums),
             posture["walking"], posture["lying"], posture["absent"],
             int(any(r["simulated"] for r in runs)), self.clock()))

    def latest_run_id(self, camera_id: str, name: str) -> Optional[int]:
        r = self.db.execute("SELECT id FROM runs WHERE camera_id=? AND name=? "
                            "ORDER BY created_ts DESC, id DESC LIMIT 1", (camera_id, name)).fetchone()
        return r["id"] if r else None

    def purge(self, now: Optional[float] = None, retention_days: int = 7,
              clips_dir: str | Path | None = None, rollup_retention_days: int = 365,
              exclude_run_ids: Iterable[int] = ()) -> dict:
        """Delete raw runs older than the retention window, in whole local days: a run goes when
        its local day is before the local day of (now - retention_days). Its clip files and
        results export go too (unless a surviving run still references the same file); its
        day's rollup is rebuilt first and then frozen so it keeps the aggregate. Rollups older
        than rollup_retention_days (local days) are deleted separately. Runs in
        `exclude_run_ids` are never purged; `kept_excluded` counts those that otherwise would be."""
        now = self.clock() if now is None else now
        cutoff = now - retention_days * 86400
        excluded = set(exclude_run_ids)
        due = [r for r in self.db.execute("SELECT id, camera_id, name, start_ts, tz_offset_min, "
                                          "export_path FROM runs").fetchall()
               if _day_of(r["start_ts"], r["tz_offset_min"]) < _day_of(cutoff, r["tz_offset_min"])]
        old = [r for r in due if r["id"] not in excluded]
        files = 0
        if old:
            old_ids = {r["id"] for r in old}
            # Rollups first, so the aggregate is kept once the raw rows are gone.
            with self._tx() as db:
                for day, cam in {(_day_of(r["start_ts"], r["tz_offset_min"]), r["camera_id"])
                                 for r in old}:
                    self._rebuild_rollup(db, day, cam)
                    db.execute("UPDATE daily_rollups SET frozen=1 WHERE day=? AND camera_id=?",
                               (day, cam))
            if clips_dir is not None:
                base = Path(clips_dir).resolve()
                out_dir = base.parent
                keep_clips, keep_exports = set(), set()
                for r in self.db.execute("SELECT id, name, export_path FROM runs"):
                    if r["id"] not in old_ids:
                        keep_exports.add(self._export_of(out_dir, r))
                for e in self.db.execute("SELECT run_id, clip_path FROM events "
                                         "WHERE clip_path IS NOT NULL"):
                    if e["run_id"] not in old_ids:
                        keep_clips.add((base / e["clip_path"]).resolve())
                for r in old:
                    for ev in self.db.execute("SELECT clip_path FROM events WHERE run_id=?",
                                              (r["id"],)).fetchall():
                        if ev["clip_path"] and (base / ev["clip_path"]).resolve() not in keep_clips:
                            files += self._unlink_inside(base, base / ev["clip_path"])
                    exp = self._export_of(out_dir, r)
                    if exp not in keep_exports:
                        files += self._unlink_inside(out_dir, exp)
            with self._tx() as db:
                db.executemany("DELETE FROM runs WHERE id=?", [(r["id"],) for r in old])
        cutoff_day = _day_of(now - rollup_retention_days * 86400,
                             _local_offset_min(now - rollup_retention_days * 86400))
        with self._tx() as db:
            n_roll = db.execute("DELETE FROM daily_rollups WHERE day < ?", (cutoff_day,)).rowcount
            db.execute("DELETE FROM alerts WHERE created_ts < ?",
                       (now - rollup_retention_days * 86400,))
        return {"runs": len(old), "files": files, "rollups": n_roll,
                "kept_excluded": len(due) - len(old)}

    @staticmethod
    def _export_of(out_dir: Path, run) -> Path:
        p = run["export_path"] or f"{run['name']}_results.json"
        return (out_dir / p).resolve()  # absolute export_path wins over out_dir

    def import_legacy(self, out_dir: str | Path) -> dict:
        """Import Phase-0 `*_results.json` files (idempotent). Returns counts."""
        out_dir = Path(out_dir)
        imported = failed = 0
        names = set()
        for p in sorted(out_dir.glob("*_results.json")):
            name = p.name[: -len("_results.json")]
            names.add(name)
            try:
                self.save_run(json.loads(p.read_text()), name=name, export_path=str(p.resolve()))
                imported += 1
            except (ValueError, KeyError, sqlite3.Error, OSError) as e:
                log.warning("could not import %s: %s", p.name, e)
                failed += 1
        skipped = 0
        hist = out_dir / "history.jsonl"
        if hist.is_file():
            for line in hist.read_text().splitlines():
                try:
                    if line.strip() and json.loads(line).get("clip") not in names:
                        skipped += 1
                except ValueError:
                    skipped += 1
        return {"imported": imported, "failed": failed, "history_rows_skipped": skipped}

    @staticmethod
    def _unlink_inside(root: Path, path: Path) -> int:
        """Unlink `path` only if it resolves inside `root` (path-traversal guard)."""
        p = path.resolve()
        if p == root or not p.is_relative_to(root) or not p.is_file():
            return 0
        p.unlink(missing_ok=True)
        return 1

    # ---- reads ------------------------------------------------------------------------
    def list_runs(self, limit: int = 200) -> list[dict]:
        """Newest first."""
        rows = self.db.execute(
            "SELECT id,name,camera_id,start_ts,tz_offset_min,data_label FROM runs "
            "ORDER BY start_ts DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r["id"], "name": r["name"], "camera_id": r["camera_id"],
                 "start_time": _local_iso(r["start_ts"], r["tz_offset_min"]),
                 "data_label": r["data_label"]} for r in rows]

    def get_run(self, run_id: int) -> Optional[dict]:
        db = self.db
        r = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if r is None:
            return None
        s = json.loads(r["summary_json"])
        out = {
            "id": r["id"], "banner": s.get("banner"), "data_label": r["data_label"],
            "source": r["source"], "start_time": _local_iso(r["start_ts"], r["tz_offset_min"]),
            "duration_s": r["duration_s"], "config": json.loads(r["config_json"] or "{}"),
            "events": [{"kind": e["kind"], "t_onset": e["t_onset"],
                        "t_confirmed": e["t_confirmed"], "message": e["message"],
                        "details": json.loads(e["details_json"] or "{}"), "clip": e["clip_path"]}
                       for e in db.execute("SELECT * FROM events WHERE run_id=? ORDER BY rowid",
                                           (run_id,))],
            "timeline_by_hour": {h["hour_label"]: {p: h[p] for p in _POSTURES}
                                 for h in db.execute("SELECT * FROM timeline_hours WHERE "
                                                     "run_id=? ORDER BY hour_label", (run_id,))},
            "posture_track": [{"t": p["t"], "posture": p["posture"], "motion": p["motion"]}
                              for p in db.execute("SELECT * FROM posture_track WHERE run_id=? "
                                                  "ORDER BY t", (run_id,))],
            "gait_segments": [{k: g[k] for k in ("t_start", "duration_s", "speed_m_s_approx",
                                                 "speed_cv", "trunk_sway_deg")}
                              for g in db.execute("SELECT * FROM gait_segments WHERE run_id=? "
                                                  "ORDER BY rowid", (run_id,))],
        }
        for kind, key, unit in _VITALS:
            out[key] = {"unit": s[key]["unit"], "summary": s[key]["summary"], "windows": [
                {"t": w["t"], "value": w["value"], "quality": w["quality"],
                 "reported": bool(w["reported"]), **json.loads(w["extra_json"] or "{}")}
                for w in db.execute("SELECT * FROM vitals_windows WHERE run_id=? AND kind=? "
                                    "ORDER BY rowid", (run_id, kind))]}
        return out

    def history(self, camera_id: Optional[str] = None, since_ts: Optional[float] = None) -> list[dict]:
        sql, args = "SELECT * FROM runs WHERE 1=1", []
        if camera_id is not None:
            sql, args = sql + " AND camera_id=?", args + [camera_id]
        if since_ts is not None:
            sql, args = sql + " AND start_ts>=?", args + [since_ts]
        rows = []
        for r in self.db.execute(sql + " ORDER BY start_ts, id", args):
            s = json.loads(r["summary_json"])
            rows.append({"id": r["id"], "camera_id": r["camera_id"],
                         "start_time": _local_iso(r["start_ts"], r["tz_offset_min"]),
                         "clip": r["name"], "data_label": r["data_label"],
                         **{k: s[k] for k in ("hr_median", "br_median", "gait_speed",
                                              "n_falls", "n_inactivity")}})
        return rows

    def daily_rollups(self, camera_id: Optional[str] = None,
                      since_day: Optional[str] = None) -> list[dict]:
        sql, args = "SELECT * FROM daily_rollups WHERE 1=1", []
        if camera_id is not None:
            sql, args = sql + " AND camera_id=?", args + [camera_id]
        if since_day is not None:
            sql, args = sql + " AND day>=?", args + [since_day]
        return [dict(r) for r in self.db.execute(sql + " ORDER BY day, camera_id", args)]

    # ---- alerts -----------------------------------------------------------------------
    def add_alert(self, camera_id: str, kind: str, event: dict, created_ts: float, sink: str,
                  status: str, next_attempt_ts: Optional[float] = None,
                  last_error: Optional[str] = None, run_id: Optional[int] = None) -> int:
        cur = self.db.execute(
            "INSERT INTO alerts (camera_id,kind,run_id,event_json,created_ts,sink,status,"
            "next_attempt_ts,last_error) VALUES (?,?,?,?,?,?,?,?,?)",
            (camera_id, kind, run_id, json.dumps(event, default=str), created_ts, sink, status,
             next_attempt_ts, last_error))
        return cur.lastrowid

    def recent_alert_exists(self, camera_id: str, kind: str, since_ts: float) -> bool:
        """Only alerts that were (or will be) delivered count: a quiet-hours or no-sink
        'suppressed' row, or a failed one, must not hide a later real alert."""
        return self.db.execute("SELECT 1 FROM alerts WHERE camera_id=? AND kind=? AND created_ts>? "
                               "AND status IN ('pending','sent') LIMIT 1",
                               (camera_id, kind, since_ts)).fetchone() is not None

    def due_alerts(self, now: float, limit: int = 50, exclude_sinks: Iterable[str] = ()) -> list[dict]:
        """Due pending rows: fewest attempts first, then newest first (fresh alerts are never
        starved by old retries)."""
        ex = list(exclude_sinks)
        skip = f" AND sink NOT IN ({','.join('?' * len(ex))})" if ex else ""
        rows = self.db.execute("SELECT * FROM alerts WHERE status='pending' AND "
                               f"COALESCE(next_attempt_ts,0)<=?{skip} "
                               "ORDER BY attempts, created_ts DESC, id LIMIT ?",
                               (now, *ex, limit)).fetchall()
        return [dict(r) for r in rows]

    def mark_alert(self, alert_id: int, status: str, attempts: Optional[int] = None,
                   next_attempt_ts: Optional[float] = None, last_error: Optional[str] = None):
        self.db.execute("UPDATE alerts SET status=?, attempts=COALESCE(?,attempts), "
                        "next_attempt_ts=?, last_error=? WHERE id=?",
                        (status, attempts, next_attempt_ts, last_error, alert_id))

    def claim_alert(self, alert_id: int, now: float, lease_s: float) -> bool:
        """Atomically lease a due pending row (so two processes never both send it)."""
        return self.db.execute(
            "UPDATE alerts SET next_attempt_ts=? WHERE id=? AND status='pending' AND "
            "COALESCE(next_attempt_ts,0)<=?", (now + lease_s, alert_id, now)).rowcount == 1

    def _group_ids(self, alert_id: int) -> list[int]:
        r = self.db.execute("SELECT camera_id,kind,created_ts FROM alerts WHERE id=?",
                            (alert_id,)).fetchone()
        if not r:
            return []
        return [x["id"] for x in self.db.execute(
            "SELECT id FROM alerts WHERE camera_id=? AND kind=? AND created_ts=?",
            (r["camera_id"], r["kind"], r["created_ts"]))]

    def ack_alert(self, alert_id: int, now: Optional[float] = None) -> bool:
        """Acknowledge an alert (all sink rows of the same alert). False if it does not exist."""
        ids, now = self._group_ids(alert_id), self.clock() if now is None else now
        for i in ids:
            self.db.execute("UPDATE alerts SET acknowledged_ts=COALESCE(acknowledged_ts,?) "
                            "WHERE id=?", (now, i))
        return bool(ids)

    def mark_false_alarm(self, alert_id: int, now: Optional[float] = None) -> bool:
        """Record 'not a fall' feedback (also acknowledges). Recorded only; nothing is auto-tuned."""
        ids, now = self._group_ids(alert_id), self.clock() if now is None else now
        for i in ids:
            self.db.execute("UPDATE alerts SET false_alarm=1, feedback_ts=?, "
                            "acknowledged_ts=COALESCE(acknowledged_ts,?) WHERE id=?", (now, now, i))
        return bool(ids)

    def list_alerts(self, since_ts: Optional[float] = None, status: Optional[str] = None,
                    limit: int = 200) -> list[dict]:
        """One entry per alert (camera_id, kind, created_ts), merging its per-sink rows. Newest first.
        `status` filters on the merged status (pending > sent > failed > suppressed)."""
        sql, args = "SELECT * FROM alerts", []
        if since_ts is not None:
            sql, args = sql + " WHERE created_ts>=?", [since_ts]
        groups: dict = {}
        for r in self.db.execute(sql + " ORDER BY created_ts DESC, id", args):
            g = groups.setdefault((r["camera_id"], r["kind"], r["created_ts"]), {
                "id": r["id"], "camera_id": r["camera_id"], "kind": r["kind"],
                "run_id": r["run_id"], "created_ts": r["created_ts"],
                "created": _local_iso(r["created_ts"], _local_offset_min(r["created_ts"])),
                "event": json.loads(r["event_json"]) if r["event_json"] else None,
                "sinks": [], "acknowledged_ts": None, "false_alarm": False})
            g["sinks"].append({"sink": r["sink"], "status": r["status"], "attempts": r["attempts"],
                               "last_error": r["last_error"]})
            g["acknowledged_ts"] = g["acknowledged_ts"] or r["acknowledged_ts"]
            g["false_alarm"] = g["false_alarm"] or bool(r["false_alarm"])
        out = []
        for g in groups.values():
            sts = {x["status"] for x in g["sinks"]}
            g["status"] = next(x for x in ("pending", "sent", "failed", "suppressed") if x in sts)
            g["acknowledged"] = g["acknowledged_ts"] is not None
            if status is None or g["status"] == status:
                out.append(g)
        return out[:limit]

    def unacked_count(self, since_ts: Optional[float] = None) -> int:
        """Alerts (not sink rows) with no acknowledgement; independent of any list limit."""
        w, args = ("WHERE created_ts>=? ", [since_ts]) if since_ts is not None else ("", [])
        return self.db.execute(
            f"SELECT COUNT(*) FROM (SELECT 1 FROM alerts {w}GROUP BY camera_id,kind,created_ts "
            "HAVING MAX(acknowledged_ts) IS NULL)", args).fetchone()[0]

    def false_alarms_per_day(self, since_ts: Optional[float] = None) -> dict[str, int]:
        """Alerts marked 'not a fall', counted per local calendar day of the alert."""
        sql, args = "SELECT camera_id,kind,created_ts FROM alerts WHERE false_alarm=1", []
        if since_ts is not None:
            sql, args = sql + " AND created_ts>=?", [since_ts]
        days: dict[str, int] = {}
        for r in self.db.execute(sql + " GROUP BY camera_id,kind,created_ts", args):
            d = _day_of(r["created_ts"], _local_offset_min(r["created_ts"]))
            days[d] = days.get(d, 0) + 1
        return dict(sorted(days.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m carelens.store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("purge", help="delete raw runs/clips past retention")
    p.add_argument("--out", default="output")
    p.add_argument("--db", default="carelens.db", help="relative to --out")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--rollup-days", type=int, default=365)
    p = sub.add_parser("import-legacy", help="import Phase-0 *_results.json files (not automatic)")
    p.add_argument("--out", default="output")
    p.add_argument("--db", default="carelens.db", help="relative to --out")
    a = ap.parse_args(argv)
    out = Path(a.out)
    with Store(out / a.db) as st:
        if a.cmd == "purge":
            print(json.dumps(st.purge(retention_days=a.days, clips_dir=out / "clips",
                                      rollup_retention_days=a.rollup_days)))
        else:
            print(json.dumps(st.import_legacy(out)))


if __name__ == "__main__":
    main()
