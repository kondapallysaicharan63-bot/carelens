"""Phase 1 definition of done: a configured camera -> pipeline -> SQLite -> dashboard."""
import json

import yaml

from carelens.pipeline import main
from carelens.synth import Scenario, Segment, render
from dashboard.app import create_app


def test_configured_camera_to_sqlite_to_dashboard(tmp_path):
    scn = Scenario([
        Segment("walk", 3, x0=0.2, x1=0.6),
        Segment("transition", 0.5, posture="stand", to="lie", x0=0.6),  # sudden fall
        Segment("still", 8, posture="lie", x0=0.6),
    ])
    clip, truth = tmp_path / "clip.mp4", tmp_path / "clip.truth.json"
    render(scn, clip)
    truth.write_text(json.dumps(scn.truth))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "cameras": [{"id": "hall", "room": "hallway", "type": "file", "source": "clip.mp4"}],
        "thresholds": {"fall_still_s": 3, "inactivity_min": 10},
        "storage": {"out_dir": "out"},
    }))

    main(["--camera", "hall", "--config", str(cfg), "--truth-pose", str(truth)])

    client = create_app(tmp_path / "out").test_client()
    runs = client.get("/api/runs").get_json()
    assert len(runs) == 1 and runs[0]["camera_id"] == "hall"
    assert runs[0]["data_label"] == "simulated"  # ground-truth poses are never real footage
    run = client.get(f"/api/run/{runs[0]['id']}").get_json()
    assert [e["kind"] for e in run["events"]] == ["fall"]
    assert run["banner"]  # persistent simulated/test banner
    clip_name = run["events"][0]["clip"]
    assert clip_name and client.get(f"/clips/{clip_name}").status_code == 200
    roll = client.get("/api/rollups").get_json()
    assert roll[0]["camera_id"] == "hall" and roll[0]["n_falls"] == 1
