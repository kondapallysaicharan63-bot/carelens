"""Local dashboard. Run:  python dashboard/app.py --out output   then open http://127.0.0.1:5000

Binds to localhost only. Nothing is sent anywhere."""
import argparse
import hmac
import ipaddress
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_from_directory, abort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `python dashboard/app.py` finds the carelens package
from carelens.config import runtime_env  # noqa: E402
from carelens.store import Store  # noqa: E402

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_ALERT_STATUSES = {"pending", "sent", "failed", "suppressed"}


def _is_loopback(addr) -> bool:
    try:
        ip = ipaddress.ip_address(addr or "")
        return ip.is_loopback or bool(getattr(ip, "ipv4_mapped", None) and ip.ipv4_mapped.is_loopback)
    except ValueError:
        return False


def bind_warning(host: str, token_set: bool):
    """Message if the dashboard would be exposed beyond this machine without a write token."""
    if _is_loopback(host) or host == "localhost" or token_set:
        return None
    return (f"warning: dashboard binds to {host} but CARELENS_DASH_TOKEN is not set; alert "
            "acknowledge/feedback writes will be refused from other machines")


def dash_token(env_dir=None) -> str:
    """CARELENS_DASH_TOKEN from the process env, else <env_dir or cwd>/.env."""
    return runtime_env(Path(env_dir or ".") / ".env").get("CARELENS_DASH_TOKEN", "")


def create_app(out_dir: str, db_path: str | None = None, env_dir=None,
               allowed_origins=()) -> Flask:
    out = Path(out_dir).resolve()
    db = out / (db_path or "carelens.db")  # a relative path is relative to out_dir
    app = Flask(__name__, static_folder=str(HERE / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 4096   # the write endpoints take no body
    token = dash_token(env_dir)
    extra = {o.strip().rstrip("/") for o in (*allowed_origins, *os.environ.get(
        "CARELENS_DASH_ORIGINS", "").split(",")) if o.strip()}

    def write_allowed() -> bool:
        """Gate for state-changing requests. With CARELENS_DASH_TOKEN set: the X-CareLens-Token
        header must match (and any Origin must be same-origin). Unset: loopback clients only, and
        any Origin must be a localhost one (drive-by CSRF / DNS rebinding)."""
        origin = request.headers.get("Origin")
        if token:
            sent = request.headers.get("X-CareLens-Token", "")
            if not hmac.compare_digest(sent.encode(), token.encode()):
                return False
            # X-Forwarded-* is not trusted: behind a proxy, list the public origin explicitly
            return origin is None or origin == request.host_url.rstrip("/") or origin in extra
        if not _is_loopback(request.remote_addr):
            return False
        if origin is not None:
            try:
                return (urlsplit(origin).hostname or "") in _LOCAL_HOSTS
            except ValueError:
                return False
        return True

    def read(fn, empty):
        """Run fn(store) on a fresh read-only connection; `empty` if there is no database yet."""
        if not db.is_file():
            return empty
        with Store(db, readonly=True) as st:
            return fn(st)

    @app.get("/")
    def index():
        return send_from_directory(HERE / "static", "index.html")

    @app.get("/api/runs")
    def runs():
        return jsonify(read(lambda s: s.list_runs(), []))

    @app.get("/api/run/<int:run_id>")
    def run(run_id):
        r = read(lambda s: s.get_run(run_id), None)
        if r is None:
            abort(404)
        return jsonify(r)

    @app.get("/api/history")
    def history():
        return jsonify(read(lambda s: s.history(), []))

    @app.get("/api/rollups")
    def rollups():
        return jsonify(read(lambda s: s.daily_rollups(), []))

    @app.get("/api/alerts")
    def alerts():
        try:
            since = float(request.args["since"]) if request.args.get("since") else None
        except ValueError:
            abort(400)
        status = request.args.get("status") or None
        if status is not None and status not in _ALERT_STATUSES:
            abort(400)

        empty = {"alerts": [], "unacked": 0, "false_alarms_per_day": {}}

        def fn(st):
            try:
                groups = st.list_alerts(since_ts=since)
                return {"alerts": [g for g in groups if status in (None, g["status"])],
                        "unacked": st.unacked_count(since),
                        "false_alarms_per_day": st.false_alarms_per_day(since)}
            except sqlite3.OperationalError:   # database not migrated to the alerts schema yet
                return empty
        return jsonify(read(fn, empty))

    def write_alert(method: str, alert_id: int):
        if not write_allowed():
            abort(403)
        if not db.is_file():
            abort(404)
        with Store(db) as st:   # writable connection, per request
            if not getattr(st, method)(alert_id):
                abort(404)
        return jsonify({"ok": True})

    @app.post("/api/alerts/<int:alert_id>/ack")
    def alert_ack(alert_id):
        return write_alert("ack_alert", alert_id)

    @app.post("/api/alerts/<int:alert_id>/false-alarm")
    def alert_false_alarm(alert_id):
        return write_alert("mark_false_alarm", alert_id)

    @app.get("/clips/<path:name>")
    def clip(name):
        return send_from_directory(out / "clips", name)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir (default: output, or storage.out_dir)")
    ap.add_argument("--config", default=None, help="use storage.out_dir / db_path from this config")
    ap.add_argument("--db", default=None, help="SQLite file (default: <out>/carelens.db)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    out, db = a.out, a.db
    if a.config:
        from carelens.config import load_config
        st = load_config(a.config).storage
        out = out or str(Path(a.config).resolve().parent / st.out_dir)
        db = db or st.db_path
    env_dir = Path(a.config).resolve().parent if a.config else None
    origins = load_config(a.config).dashboard.allowed_origins if a.config else []
    warn = bind_warning(a.host, bool(dash_token(env_dir)))
    if warn:
        print(warn, file=sys.stderr)
    create_app(out or "output", db, env_dir, origins).run(host=a.host, port=a.port)
