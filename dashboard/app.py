"""Local dashboard. Run:  python dashboard/app.py --out output   then open http://127.0.0.1:5000

Binds to localhost only. Nothing is sent anywhere."""
import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, abort

HERE = Path(__file__).resolve().parent


def create_app(out_dir: str) -> Flask:
    out = Path(out_dir).resolve()
    app = Flask(__name__, static_folder=str(HERE / "static"))

    @app.get("/")
    def index():
        return send_from_directory(HERE / "static", "index.html")

    @app.get("/api/runs")
    def runs():
        return jsonify(sorted(p.name.replace("_results.json", "")
                              for p in out.glob("*_results.json")))

    @app.get("/api/run/<name>")
    def run(name):
        p = out / f"{name}_results.json"
        if not p.is_file() or p.parent != out:
            abort(404)
        return jsonify(json.loads(p.read_text()))

    @app.get("/api/history")
    def history():
        p = out / "history.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        return jsonify(rows)

    @app.get("/clips/<path:name>")
    def clip(name):
        return send_from_directory(out / "clips", name)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    create_app(a.out).run(host=a.host, port=a.port)
