"""Event logic driven directly by scripted landmarks (no video)."""
from datetime import datetime

from carelens.events import ActivityTimeline, FallDetector, GaitAnalyzer, InactivityDetector
from carelens.features import FeatureExtractor
from carelens.synth import ASPECT, Scenario, Segment


def run_states(scn, fps=15.0):
    fx = FeatureExtractor(aspect=ASPECT)
    n = int(scn.duration * fps)
    return [fx.update(i / fps, scn.pose_landmarks(i / fps)) for i in range(n)]


def detect(states, det):
    return [e for s in states if (e := det.update(s))]


def test_postures_are_classified():
    scn = Scenario([Segment("still", 3, posture="stand"), Segment("still", 3, posture="sit"),
                    Segment("still", 3, posture="lie"), Segment("walk", 4, x0=0.2, x1=1.1),
                    Segment("absent", 2)])
    st = run_states(scn)
    at = lambda t: st[int(t * 15)].posture
    assert at(1.5) == "standing" and at(4.5) == "sitting" and at(7.5) == "lying"
    assert at(11) == "walking" and at(14) == "absent"


def test_sudden_fall_then_stillness_is_flagged():
    scn = Scenario([Segment("still", 5, posture="stand"),
                    Segment("transition", 0.6, posture="stand", to="lie"),
                    Segment("still", 25, posture="lie")])
    ev = detect(run_states(scn), FallDetector(still_seconds=20))
    assert len(ev) == 1 and 4.5 < ev[0].t_onset < 6.5 and ev[0].t_confirmed >= 25


def test_slow_lying_down_is_not_a_fall():
    scn = Scenario([Segment("still", 3, posture="stand"),
                    Segment("transition", 2.5, posture="stand", to="sit"),
                    Segment("transition", 4, posture="sit", to="lie"),
                    Segment("still", 30, posture="lie")])
    assert detect(run_states(scn), FallDetector(still_seconds=20)) == []


def test_fall_with_quick_recovery_is_not_flagged():
    scn = Scenario([Segment("still", 3, posture="stand"),
                    Segment("transition", 0.6, posture="stand", to="lie"),
                    Segment("still", 5, posture="lie"),
                    Segment("transition", 2, posture="lie", to="stand"),
                    Segment("still", 25, posture="stand")])
    assert detect(run_states(scn), FallDetector(still_seconds=20)) == []


def test_inactivity_fires_once_and_not_when_absent():
    scn = Scenario([Segment("still", 70, posture="sit"), Segment("absent", 80)])
    ev = detect(run_states(scn), InactivityDetector(seconds=60))
    assert len(ev) == 1 and 59 < ev[0].t_confirmed - ev[0].t_onset < 62


def test_timeline_and_gait():
    scn = Scenario([Segment("walk", 5, x0=0.2, x1=1.2), Segment("still", 10, posture="sit")])
    states = run_states(scn)
    tl, g = ActivityTimeline(datetime(2026, 1, 1, 9, 30)), GaitAnalyzer(person_height_m=1.65)
    for s in states:
        tl.update(s); g.update(s)
    hours = tl.to_dict()["2026-01-01 09:00"]
    assert hours["sitting"] > 8 and hours["walking"] > 3
    segs = g.finish()
    assert len(segs) == 1
    # true speed: 1.0 height-unit / 5 s, body extent 0.65 -> ~0.3 body/s -> ~0.5 m/s
    assert 0.35 < segs[0]["speed_m_s_approx"] < 0.7
