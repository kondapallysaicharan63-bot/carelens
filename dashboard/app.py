"""Local dashboard. Run:  python dashboard/app.py --out output   then open http://127.0.0.1:5000

Binds to localhost only. Nothing is sent anywhere."""
import argparse
import sys
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, abort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `python dashboard/app.py` finds the carelens package
from carelens.store import Store  # noqa: E402


def create_app(out_dir: str, db_path: str | None = None) -> Flask:
    out = Path(out_dir).resolve()
    db = out / (db_path or "carelens.db")  # a relative path is relative to out_dir
    app = Flask(__name__, static_folder=str(HERE / "static"))

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
    create_app(out or "output", db).run(host=a.host, port=a.port)
