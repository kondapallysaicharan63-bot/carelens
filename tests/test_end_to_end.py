"""Render each synthetic sample clip, run the full pipeline, check expected events and vitals."""
import pytest

from carelens.pipeline import Config, run
from carelens.synth import Scenario, Segment, demo_scenario, render


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    scn = demo_scenario(inactivity_s=60, hr_bpm=78, br_bpm=14)
    render(scn, d / "demo.mp4")
    res = run(d / "demo.mp4", scn.provider(), d, Config(inactivity_seconds=60, data_label="simulated"),
              clip_name="demo")
    return d, res


def test_demo_flags_fall_and_inactivity(demo):
    d, r = demo
    kinds = [e["kind"] for e in r["events"]]
    assert kinds.count("fall") == 1 and kinds.count("inactivity") == 1
    fall = next(e for e in r["events"] if e["kind"] == "fall")
    assert 52 < fall["t_onset"] < 55
    assert all((d / "clips" / e["clip"]).exists() for e in r["events"])


def test_demo_vitals_within_tolerance(demo):
    _, r = demo
    assert abs(r["heart_rate"]["summary"]["median"] - 78) <= 3
    assert abs(r["breathing_rate"]["summary"]["median"] - 14) <= 1.5
    assert r["heart_rate"]["summary"]["n_reported"] > 0


def test_banner_and_wellness_wording(demo):
    _, r = demo
    assert "Simulated or test data" in r["banner"]
    for e in r["events"]:
        assert "diagnos" not in e["message"].lower() and "check-in suggested" in e["message"].lower()


def test_dim_light_reports_no_heart_rate(tmp_path):
    scn = Scenario([Segment("still", 25, posture="sit")], hr_bpm=70)
    render(scn, tmp_path / "dark.mp4", light=0.15)
    r = run(tmp_path / "dark.mp4", scn.provider(), tmp_path, Config(), clip_name="dark")
    assert r["heart_rate"]["summary"]["median"] is None


def test_person_leaving_room_gives_no_events(tmp_path):
    scn = Scenario([Segment("walk", 4, x0=0.3, x1=1.4), Segment("absent", 30)])
    render(scn, tmp_path / "leave.mp4")
    r = run(tmp_path / "leave.mp4", scn.provider(), tmp_path, Config(inactivity_seconds=20),
            clip_name="leave")
    assert r["events"] == []
