"""W9 blink rate / eyes-closed time (Python reference) and activity summary. Synthetic faces and poses only; not a medical device."""
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from carelens import websession as ws
from carelens.blink import (BlinkTracker, FIXTURE_INDICES, LEFT_EYE, RIGHT_EYE, ear_trace, eye_ear, eye_points, frame_ear, head_turned,
                            percentile, run_blink, synth_face)
from carelens.events import ActivitySummary, ActivityTimeline
from carelens.features import BodyState
from carelens.params import PARAMS, Blink, blink_errors

BP = PARAMS.blink
ROOT = Path(__file__).resolve().parent.parent
END = 82.0   # 20 s calibration + about 1 s to verify the frame rate + a full 60 s window


def _truth(spans, end=END - 0.04, window=60.0, slack=0.0):   # the window ends at the last frame (80 s at 25 Hz: 79.96 s)
    lo = end - window + slack
    return (sum(1 for a, b in spans if a >= lo and b - a <= 0.5), sum(1 for a, b in spans if a >= lo and b - a > 0.5))


def _bracket(spans):
    """Blinks whose start is within +-0.1 s of the window edge may fall either side (the detector times a closure at its first
    closed frame, the truth at the half-way crossing): (certainly inside, possibly inside)."""
    return _truth(spans, slack=0.1)[0], _truth(spans, slack=-0.1)[0]


def _final(inp):
    return run_blink(inp)["final"]


def test_params_validate_and_reject_inconsistent():
    assert blink_errors(PARAMS.blink.model_dump()) == []
    base = PARAMS.blink.model_dump()
    for bad in ({"perclos_frac": 0.6}, {"closed_frac": 0.8}, {"reopen_frac": 1.0}, {"blink_min_ms": 600.0}, {"min_window_s": 90.0},
                {"baseline_pct": 0.0}, {"min_rate_hz": 10.0}, {"max_dt_s": 3.0}, {"baseline_s": 0.0},
                {"eye_landmarks": {"left": [33, 160, 158, 133, 153, 33], "right": base["eye_landmarks"]["right"]}},
                {"pose_landmarks": {"nose": 1, "forehead": 10, "mouth": 13}}):
        assert blink_errors({**base, **bad}) != [], bad
        with pytest.raises(ValueError):
            Blink.model_validate({**base, **bad})


def test_eye_landmarks_follow_the_p1_to_p6_pattern():
    # MediaPipe FaceLandmarker: 33/133 corners, 160/158 upper lid, 144/153 lower lid (left); 362/263 corners, 385/387 upper, 380/373 lower (right)
    assert LEFT_EYE == [33, 160, 158, 133, 153, 144] and RIGHT_EYE == [362, 385, 387, 263, 373, 380]
    for idx in (LEFT_EYE, RIGHT_EYE):
        pts = eye_points(0.4, 0.5, 0.3)
        # p2 sits above p6, p3 above p5 (same x), corners p1/p4 on the eye line
        assert pts[1][0] == pytest.approx(pts[5][0]) and pts[2][0] == pytest.approx(pts[4][0])
        assert pts[1][1] < pts[5][1] and pts[2][1] < pts[4][1] and pts[0][1] == pts[3][1]
    lm = {i: p for i, p in zip(LEFT_EYE, eye_points(0.4, 0.5, 0.31))}
    assert eye_ear(lm, LEFT_EYE, 1.0) == pytest.approx(0.31)
    lm.update({i: p for i, p in zip(RIGHT_EYE, eye_points(0.6, 0.5, 0.27))})
    assert frame_ear(lm, 1.0) == pytest.approx(0.29)            # mean of the two eyes
    assert frame_ear({**lm, RIGHT_EYE[0]: lm[RIGHT_EYE[3]]}, 1.0) is None   # a degenerate eye: no value


def test_percentile_matches_numpy():
    rng = np.random.default_rng(3)
    v = rng.normal(0.3, 0.02, 237).tolist()
    for q in (0, 10, 50, 90, 100):
        assert percentile(v, q) == pytest.approx(float(np.percentile(v, q)))


@pytest.mark.parametrize("bpm", [8, 15, 25])
def test_blink_counts_exact_on_clean_traces(bpm):
    for seed in (1, 2, 3):
        inp, spans = synth_face(END, 25.0, blinks_per_min=bpm, blink_ms_range=(80, 400), seed=seed, open_ear=0.25 + 0.05 * seed)
        f = _final(inp)
        lo, hi = _bracket(spans)
        assert f["reason"] == "ok" and f["reported"]
        assert lo <= round(f["blinks_per_min"]) <= hi and f["long_closures"] == 0
        # one-to-one with the truth: every counted blink starts within 0.1 s of a true one
        out = run_blink(inp)
        assert all(any(abs(b["t_start"] - a) < 0.1 for a, _ in spans) for b in out["blinks"])
        assert sum(1 for a, _ in spans if a >= 21.4) <= len(out["blinks"]) <= sum(1 for a, _ in spans if a >= 20.8)
        assert abs(f["blinks_per_min"] - bpm) <= 3          # the generator's own interval jitter, not the detector


@pytest.mark.parametrize("bpm", [8, 15, 25])
def test_blink_rate_with_landmark_jitter_within_one_per_minute(bpm):
    for seed in (4, 5, 6):
        inp, spans = synth_face(END, 25.0, blinks_per_min=bpm, blink_ms_range=(80, 400), jitter_px=0.4, seed=seed, open_ear=0.25 + 0.04 * (seed - 4))
        f = _final(inp)
        assert f["reason"] == "ok"
        lo, hi = _bracket(spans)
        assert lo - 1.0 <= f["blinks_per_min"] <= hi + 1.0, (bpm, seed)


def test_durations_and_long_closures_are_separated():
    # 80 ms, 200 ms and 400 ms blinks are blinks; 0.8 s and 1.5 s closures are long closures, not blinks
    inp, spans = synth_face(70.0, 30.0, blinks_per_min=0, open_ear=0.3, seed=7,
                            closures=[(30.0, 30.08), (34.0, 34.2), (38.0, 38.4), (45.0, 45.8), (52.0, 53.5)])
    out = run_blink(inp)
    assert len(out["blinks"]) == 3 and len(out["long_closures"]) == 2
    assert out["final"]["long_closures"] == 2
    assert [round(b["t_end"] - b["t_start"], 1) for b in out["blinks"]] == pytest.approx([0.1, 0.2, 0.4], abs=0.06)


def test_a_one_frame_flicker_is_not_a_blink():
    inp, _ = synth_face(60.0, 25.0, blinks_per_min=0, seed=8, closures=[(40.0, 40.03)])
    assert len(run_blink(inp)["blinks"]) == 0


@pytest.mark.parametrize("name,kw,pct", [
    ("none", dict(blinks_per_min=0), 0.0),
    ("blink_level", dict(blinks_per_min=15, blink_ms=200), 5.0),
    ("long", dict(blinks_per_min=0, closures=[(25.0 + 7 * i, 25.0 + 7 * i + 1.6) for i in range(7)]), 20.0),
])
def test_perclos_within_two_points_of_truth(name, kw, pct):
    inp, spans = synth_face(85.0, 25.0, jitter_px=0.4, seed=5, **kw)
    t, ear, _ = ear_trace(85.0, 25.0, seed=5, **kw)
    _, open_ear, _ = ear_trace(85.0, 25.0, seed=5)
    m = t >= 85.0 - 60.0
    truth = 100.0 * float(np.mean(ear[m] < BP.perclos_frac * open_ear[m]))
    f = _final(inp)
    assert f["reason"] == "ok"
    assert abs(100 * f["perclos"] - truth) <= 2.0
    assert abs(truth - pct) <= 3.0                         # the scenario really is the advertised level


def test_baseline_survives_long_closures():
    # closed 25 % of the time: the 90th percentile of the EAR is still the open level
    inp, spans = synth_face(80.0, 25.0, blinks_per_min=0, closures=[(22.0 + 8 * i, 22.0 + 8 * i + 2.0) for i in range(7)], seed=9, open_ear=0.3)
    f = _final(inp)
    assert 0.27 < f["baseline"] < 0.34 and f["long_closures"] >= 5


def test_calibration_gate_and_reason_precedence():
    inp, _ = synth_face(60.0, 25.0, blinks_per_min=15, seed=2)
    runs = run_blink(inp)["reason_runs"]
    assert runs[0] == {"t": 0.0, "reason": "calibrating"}
    assert runs[1]["reason"] == "ok" and runs[1]["t"] == pytest.approx(BP.baseline_s + BP.min_window_s + 1.0, abs=0.2)   # +1 s: the frame rate is verified first
    # a face-lost span: no_face, and it neither dilutes the rate nor counts as time
    inp, _ = synth_face(90.0, 25.0, blinks_per_min=15, blink_ms=150, seed=2, lost=[(50.0, 60.0)])
    out = run_blink(inp)
    assert {"t": 50.0, "reason": "no_face"} in out["reason_runs"] and out["final"]["reason"] == "ok"
    assert out["final"]["window_s"] == pytest.approx(50.0, abs=1.0)
    assert 11.0 <= out["final"]["blinks_per_min"] <= 19.0


def test_head_turn_withholds_and_blinks_resume_after():
    inp, spans = synth_face(80.0, 25.0, blinks_per_min=15, blink_ms=150, seed=2, jitter_px=0.4, yaw_segments=[(50.0, 65.0, 0.5)])
    out = run_blink(inp)
    reasons = [(r["t"], r["reason"]) for r in out["reason_runs"]]
    assert (50.0, "head_turned") in reasons and (65.0, "ok") in reasons
    assert not any(50.0 < b["t_start"] < 65.0 for b in out["blinks"]), "blinks while turned must not be counted"
    assert out["final"]["window_s"] == pytest.approx(45.0, abs=1.0)
    # a small turn (yaw ratio 0.1) is fine
    inp, _ = synth_face(60.0, 25.0, blinks_per_min=15, seed=2, yaw_segments=[(30.0, 50.0, 0.1)])
    assert all(r["reason"] != "head_turned" for r in run_blink(inp)["reason_runs"])
    # the estimate itself: frontal face is not turned, a pitched one is
    lm = {i: p for i, p in zip(LEFT_EYE + RIGHT_EYE, eye_points(0.5, 0.5, 0.3) * 2)}
    inp, _ = synth_face(1.0, 25.0)
    lm = {k: r for k, r in zip(inp["landmark_indices"], inp["landmarks"][0])}
    assert not head_turned(lm, inp["w"] / inp["h"])
    lm[1] = [lm[1][0], lm[10][1] + 0.95 * (lm[152][1] - lm[10][1])]
    assert head_turned(lm, inp["w"] / inp["h"])


def test_low_fps_is_withheld_never_guessed():
    inp, _ = synth_face(60.0, 15.0, blinks_per_min=15, seed=2)
    out = run_blink(inp)
    assert out["final"]["reason"] == "low_fps" and not out["final"]["reported"]
    assert out["blinks"] == [] or out["reason_runs"][-1]["reason"] == "low_fps"
    # 30 Hz with frame-time jitter and a few dropped frames stays ok
    inp, spans = synth_face(80.0, 30.0, blinks_per_min=15, seed=3, t_jitter_s=0.004)
    assert _final(inp)["reason"] == "ok"
    # frame gap: the outage is not counted as time and ends any open closure
    tr = BlinkTracker()
    inp, _ = synth_face(40.0, 25.0, seed=1)
    aspect = inp["w"] / inp["h"]
    for i, fr in enumerate(inp["landmarks"]):
        tr.update(i / 25.0, {k: r for k, r in zip(inp["landmark_indices"], fr)}, aspect)
    before = tr.valid_total
    tr.update(40.0 + 5.0, {k: r for k, r in zip(inp["landmark_indices"], inp["landmarks"][0])}, aspect)
    assert tr.valid_total == pytest.approx(before)


# ------------------------------------------------------------------ fixtures + export

def test_blink_fixtures_cover_the_cases():
    def load(n):
        return json.loads((ROOT / "shared" / "fixtures" / f"{n}.json").read_text())["expected"]
    for n, bpm in (("blink_8pm", 8), ("blink_15pm", 15), ("blink_25pm", 25)):
        f = load(n)["final"]
        assert f["reason"] == "ok" and abs(f["blinks_per_min"] - bpm) <= 3
    assert load("blink_long_closures")["final"]["long_closures"] == 4
    assert any(r["reason"] == "head_turned" for r in load("blink_head_turned")["reason_runs"])
    assert load("blink_low_fps")["final"]["reason"] == "low_fps"
    assert set(FIXTURE_INDICES) == set(LEFT_EYE + RIGHT_EYE + [1, 10, 152])


def _row(**kw):
    return {"t": 5.0, "wall": "2026-11-20T10:00:05.000-08:00", **kw}


def test_blink_and_activity_summaries_validated():
    s = json.loads((ROOT / "shared" / "session.example.json").read_text())
    good = [_row(kind="blink", reported=True, reason="ok", blinks_per_min=14.5, eyes_closed_pct=4.2, long_closures=0, duration_s=58.0, quality=0.9),
            _row(kind="blink", reported=False, reason="calibrating"),
            _row(kind="activity", reported=True, reason="ok", active_min=1.5, still_min=10.0, absent_min=0.5, longest_still_min=4.0, duration_s=720.0)]
    s["summaries"] = good
    assert ws.validate(s) == []
    for row, frag in ((_row(kind="blink", reported=True, reason="ok", blinks_per_min=14.0), "reported rows need"),
                      (_row(kind="blink", reported=True, reason="ok", blinks_per_min=-1, eyes_closed_pct=3), "blinks_per_min"),
                      (_row(kind="blink", reported=True, reason="ok", blinks_per_min=1, eyes_closed_pct=140), "eyes_closed_pct"),
                      (_row(kind="blink", reported=False, reason="ok", long_closures=1.5), "long_closures"),
                      (_row(kind="activity", reported=True, reason="ok", active_min=1, still_min=2), "reported rows need"),
                      (_row(kind="activity", reported=False, reason="too_short", still_min=-1), "still_min")):
        bad = copy.deepcopy(s)
        bad["summaries"] = [row]
        assert any(frag in e for e in ws.validate(bad)), (row, ws.validate(bad))
    # backward compatible: an older export without any W9 rows is still valid
    s["summaries"] = []
    del s["summaries"]
    assert ws.validate(s) == []


# ------------------------------------------------------------------ activity

def _state(t, present=True, motion=0.0, posture="sitting"):
    return BodyState(t=t, present=present, motion=motion, posture=posture if present else "absent")


def test_activity_summary_splits_active_still_absent_and_tracks_longest_still():
    st = PARAMS.common.still_motion_bodysizes_s
    a = ActivitySummary()
    tl = ActivityTimeline(datetime(2026, 1, 5, 8, 59, 50))
    plan = [(0, 10, "still"), (10, 15, "active"), (15, 40, "still"), (40, 46, "absent"), (46, 56, "still")]
    for t10 in range(0, 560):
        t = t10 / 10.0
        kind = next(k for lo, hi, k in plan if lo <= t < hi)
        s = _state(t, present=kind != "absent", motion=0.5 if kind == "active" else st / 2)
        a.update(s)
        tl.update(s)
    d = a.to_dict()
    assert d["active_s"] == pytest.approx(5.0, abs=0.2) and d["absent_s"] == pytest.approx(6.0, abs=0.2)
    assert d["still_s"] == pytest.approx(10 + 25 + 10, abs=0.3) and d["longest_still_s"] == pytest.approx(25.0, abs=0.3)
    assert d["total_s"] == pytest.approx(a.active_s + a.still_s + a.absent_s) and d["reported"]
    assert d["active_min"] == pytest.approx(d["active_s"] / 60)
    # the timeline buckets split at the hour boundary and add up to the same total
    buckets = tl.to_dict()
    assert sorted(buckets) == ["2026-01-05 08:00", "2026-01-05 09:00"]
    assert sum(sum(v.values()) for v in buckets.values()) == pytest.approx(d["total_s"], abs=0.3)


def test_activity_outage_is_not_credited_and_breaks_a_still_stretch():
    a = ActivitySummary()
    for i in range(0, 50):
        a.update(_state(i / 5.0))
    a.update(_state(10.0 + 30.0))                      # a 30 s outage: capped like the timeline, and the stretch ends
    for i in range(1, 50):
        a.update(_state(40.0 + i / 5.0))
    assert a.total_s == pytest.approx(19.6, abs=0.1)   # the 30 s outage credits nothing
    assert 9.0 < a.longest_still_s < 11.5
    assert ActivitySummary().to_dict()["reason"] == "too_short"


def test_activity_fixtures_present():
    for n in ("activity_session", "activity_gap_dropout"):
        d = json.loads((ROOT / "shared" / "fixtures" / f"{n}.json").read_text())
        assert d["input_level"] == "activity" and d["expected"]["summary"]["reported"]
    e = json.loads((ROOT / "shared" / "fixtures" / "activity_session.json").read_text())["expected"]
    assert e["summary"]["absent_s"] > 5 and e["summary"]["still_s"] > 50 and e["summary"]["longest_still_s"] > 30
    assert len(e["timeline"]) == 1


def test_stalls_cannot_bypass_the_low_fps_gate():
    # a 14 fps source with a 0.3 s stall every 2.5 s: after each stall the rate is unverified for a while; nothing may be counted then
    inp, _ = synth_face(90.0, 14.0, blinks_per_min=20, blink_ms=200, seed=3)
    t = [i / 14.0 for i in range(len(inp["landmarks"]))]
    t = [x + 0.3 * int(x / 2.5) for x in t]
    out = run_blink({**inp, "t": t})
    assert out["blinks"] == [] and out["long_closures"] == []
    assert all(r["reason"] != "ok" for r in out["reason_runs"])
    assert all(u["reason"] != "ok" for u in out["updates_1hz"])


def test_long_closure_end_time_is_the_reopening():
    inp, _ = synth_face(60.0, 25.0, blinks_per_min=0, closures=[(35.0, 36.4)], seed=4)
    ev = run_blink(inp)["long_closures"]
    assert len(ev) == 1 and ev[0]["t_end"] == pytest.approx(36.4, abs=0.1) and ev[0]["t_start"] == pytest.approx(35.0, abs=0.1)
    # cut off by a lost face: the end is when the run was cut
    inp, _ = synth_face(60.0, 25.0, blinks_per_min=0, closures=[(35.0, 38.0)], seed=4, lost=[(36.5, 40.0)])
    ev = run_blink(inp)["long_closures"]
    assert len(ev) == 1 and 36.4 <= ev[0]["t_end"] <= 36.7
