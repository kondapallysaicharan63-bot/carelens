"""W7 guided mobility tests (carelens/mobility.py) on synthetic side-on figures. Synthetic data only; not a medical device."""
import numpy as np
import pytest

from carelens import mobility as M
from carelens import synth
from carelens.params import PARAMS, Mobility, mobility_errors

MP = PARAMS.mobility


def samples(script, fps=15.0, dropouts=()):
    ts, fr = script.frames(fps, dropouts)
    return M.track_samples(ts, fr, synth.ASPECT)


def cross_t(knots, deg, up=True, after=0.0):
    """First time after ``after`` the scripted knee angle crosses ``deg`` (noise-free, 1 kHz)."""
    prev = None
    for t in np.arange(after, knots[-1][0], 0.001):
        k = synth.side_knee_deg(synth._interp_knots(t, knots))
        if prev is not None and ((up and prev < deg <= k) or (not up and prev >= deg > k)):
            return float(t)
        prev = k
    raise AssertionError("no crossing")


# ---------------------------------------------------------------- params

def test_mobility_params_validated():
    d = PARAMS.mobility.model_dump()
    assert mobility_errors(d) == []
    for key, val in (("rise_deg", 100.0), ("stand_deg", 90.0), ("stand_hip_frac", 1.5), ("sts_reps", 1), ("chair_stand_s", 0)):
        bad = {**d, key: val}
        assert mobility_errors(bad), key
        with pytest.raises(ValueError):
            Mobility.model_validate(bad)
    assert mobility_errors({**d, "walk": {**d["walk"], "height_min_m": 3.0}})
    assert mobility_errors({**d, "tug": {**d["tug"], "turn_speed_frac": 1.0}})


# ---------------------------------------------------------------- 30 s chair stand

@pytest.mark.parametrize("n,cycle", [(8, 2.9), (12, 2.1), (15, 1.7)])
def test_chair_stand_counts_full_reps_only(n, cycle):
    scr, _ = synth.chair_stand_script(n, partial_after=(2, 5), cycle=cycle, seed=n)
    r = M.chair_stand_30s(samples(scr), 2.0)
    assert r["reported"] and r["reason"] == "ok"
    assert r["reps"] == n                      # exact; the two partial rises are not counted
    assert r["partial_rises"] == 2
    assert not r["rep_in_progress"] and r["quality"] == 1.0
    assert len(r["rep_times_s"]) == n and r["rep_times_s"] == sorted(r["rep_times_s"])
    assert all(0 < t <= MP.chair_stand_s for t in r["rep_times_s"])


def test_partial_rises_alone_count_nothing():
    knots = [(0, 0), (2, 0), (2.5, 0.9), (3.0, 0.9), (3.6, 0), (6, 0), (6.5, 0.9), (7.0, 0.9), (7.6, 0), (12, 0)]
    scr = synth.SideScript(12, lambda t: synth._interp_knots(t, knots), lambda t: 0.55)
    r = M.chair_stand_30s(samples(scr), 1.0)
    assert (r["reps"], r["partial_rises"]) == (0, 2)


def test_chair_stand_noise_and_dropouts_through_tracker():
    scr, _ = synth.chair_stand_script(12, partial_after=(3, 7), cycle=2.1, jitter=0.004, seed=5)
    dropouts = [(6.0, 6.3), (14.1, 14.4), (21.5, 21.8)]       # shorter than the tracker's hold: bridged as held frames
    s = samples(scr, dropouts=dropouts)
    assert sum(x.held for x in s) > 6
    r = M.chair_stand_30s(s, 2.0)
    assert r["reps"] == 12 and r["partial_rises"] == 2
    assert 0.85 < r["quality"] < 1.0


def test_chair_stand_withheld_reasons():
    stand = synth.SideScript(12, lambda t: 1.0, lambda t: 0.55)
    assert M.chair_stand_30s(samples(stand), 2.0)["reason"] == "not_seated_at_start"
    gone = synth.SideScript(12, lambda t: 0.0, lambda t: 0.55)
    ts, _ = gone.frames()
    r = M.chair_stand_30s(M.track_samples(ts, [None] * len(ts), synth.ASPECT), 2.0)
    assert (r["reported"], r["reason"], r["quality"], r["reps"]) == (False, "out_of_view", 0.0, None)


def test_rep_in_progress_not_counted():
    scr, k = synth.chair_stand_script(3, cycle=2.4, duration=12)
    go = 2.0
    s = samples(scr)
    full = M.chair_stand_30s(s, go)
    assert full["reps"] == 3 and not full["rep_in_progress"]
    # cut the window (via a shorter chair_stand_s) while the third rep is still standing: not counted, flagged in progress
    cut = Mobility.model_validate({**MP.model_dump(), "chair_stand_s": 5.5})
    r = M.chair_stand_30s(s, go, cut)
    assert r["reps"] == 2 and r["rep_in_progress"] is True


# ---------------------------------------------------------------- 5x sit-to-stand

def test_sts_known_time_and_per_rep():
    scr, k = synth.chair_stand_script(5, cycle=2.4, seed=1)
    r = M.sit_to_stand_5x(samples(scr), 2.0)
    assert r["reported"] and r["stands"] == 5
    # truth: stand time of rep i = scripted knee crossing of stand_deg on the way up
    truth = [cross_t(k, MP.stand_deg, True, 2.0 + 2.4 * i - 0.2) for i in range(5)]
    assert r["time_s"] == pytest.approx(truth[4] - 2.0, abs=0.25)
    assert sum(r["rep_times_s"]) == pytest.approx(r["time_s"], abs=1e-9)
    assert all(t == pytest.approx(2.4, abs=0.25) for t in r["rep_times_s"][1:])


def test_sts_clock_start_convention():
    # rises 0.5 s BEFORE Go: the clock starts at the rise onset, not at Go
    scr, k = synth.chair_stand_script(5, cycle=2.4, t0=1.5)
    r = M.sit_to_stand_5x(samples(scr), 2.0)
    stand5 = cross_t(k, MP.stand_deg, True, 1.5 + 2.4 * 4 - 0.2)
    assert r["reported"] and 1.5 < stand5 - r["time_s"] < 1.9 and stand5 - r["time_s"] < 2.0
    # started more than sts_early_s before Go: already standing up when the clock window opens
    scr, _ = synth.chair_stand_script(5, cycle=2.4, t0=0.6)
    assert M.sit_to_stand_5x(samples(scr), 2.0)["reason"] == "not_seated_at_start"


def test_sts_incomplete():
    scr, _ = synth.chair_stand_script(4, cycle=2.4)
    r = M.sit_to_stand_5x(samples(scr), 2.0)
    assert (r["reported"], r["reason"], r["stands"], r["time_s"]) == (False, "incomplete", 4, None)


# ---------------------------------------------------------------- Timed Up and Go

def test_tug_known_total_and_phases():
    scr, ev, k = synth.tug_script(seed=2)
    r = M.tug(samples(scr), ev["go"])
    assert r["reported"] and r["reason"] == "ok"
    seated = cross_t(k, MP.sit_deg, False, ev["sit_start"])
    assert r["time_s"] == pytest.approx(seated - ev["go"], abs=0.3)
    ph = r["phases_s"]
    assert list(ph) == list(M.PHASES) and sum(ph.values()) == pytest.approx(r["time_s"], abs=1e-9)
    assert all(v >= 0 for v in ph.values())
    assert ph["rise"] == pytest.approx(ev["rise_end"] - ev["go"], abs=0.3)
    for name, key in (("turn_1", "turn1"), ("turn_2", "turn2")):   # slow-down stretch = pivot plus the ramps either side
        piv = ev[key][1] - ev[key][0]
        assert piv - 0.2 <= ph[name] <= piv + 0.7
    assert ph["walk_out"] == pytest.approx(ev["turn1"][0] - ev["rise_end"], abs=0.7)
    assert ph["walk_back"] == pytest.approx(ev["turn2"][0] - ev["turn1"][1], abs=0.7)


def test_tug_withheld_reasons():
    scr, ev, _ = synth.tug_script(return_trip=False)
    r = M.tug(samples(scr), ev["go"])
    assert (r["reported"], r["reason"], r["time_s"], r["phases_s"]) == (False, "no_turn", None, None)
    scr, ev, _ = synth.tug_script(sit_at_end=False)
    assert M.tug(samples(scr), ev["go"])["reason"] == "not_seated_at_end"
    start_up = synth.SideScript(8, lambda t: 1.0, lambda t: 0.3)
    assert M.tug(samples(start_up), 2.0)["reason"] == "not_seated_at_start"
    still = synth.SideScript(8, lambda t: 0.0, lambda t: 0.3)
    assert M.tug(samples(still), 2.0)["reason"] == "no_rise"
    ts, _ = still.frames()
    assert M.tug(M.track_samples(ts, [None] * len(ts), synth.ASPECT), 2.0)["reason"] == "out_of_view"


def test_tug_with_noise_and_dropouts():
    scr, ev, k = synth.tug_script(jitter=0.004, seed=9)
    r = M.tug(samples(scr, dropouts=[(5.0, 5.3), (9.6, 9.9)]), ev["go"])
    assert r["reported"]
    assert r["time_s"] == pytest.approx(cross_t(k, MP.sit_deg, False, ev["sit_start"]) - ev["go"], abs=0.4)


# ---------------------------------------------------------------- walking speed

@pytest.mark.parametrize("v,secs", [(0.6, 6.0), (1.0, 4.0), (1.4, 3.2)])
def test_walking_speed_side_on(v, secs):
    r = M.walking_speed(samples(synth.walk_script(v, secs, lead=1.0, jitter=0.002, seed=3)))
    assert r["reported"] and r["reason"] == "ok"
    assert r["speed_m_s"] == pytest.approx(v, rel=0.10)
    assert r["cadence_spm"] == pytest.approx(108.0, abs=10.0)    # scripted 0.9 stride/s = 108 steps/min
    assert r["n_not_side_on"] == 0 and r["quality"] > 0.9


def test_walking_speed_scales_with_entered_height():
    s = samples(synth.walk_script(1.0, 4.0, lead=1.0))
    base = M.walking_speed(s)["speed_m_s"]
    assert M.walking_speed(s, 1.80)["speed_m_s"] == pytest.approx(base * 1.80 / 1.65, rel=1e-6)


@pytest.mark.parametrize("drift", [0.0, 0.6])
def test_walking_toward_camera_is_not_side_on(drift):
    r = M.walking_speed(samples(synth.approach_script(drift_u=drift)))
    assert (r["reported"], r["reason"], r["speed_m_s"]) == (False, "not_side_on", None)


def test_walking_speed_other_reasons():
    s = samples(synth.walk_script(1.0, 4.0, lead=1.0))
    assert M.walking_speed(s, 0.5)["reason"] == "bad_height"
    assert M.walking_speed(samples(synth.walk_script(1.0, 1.2)))["reason"] == "too_short"
    still = synth.SideScript(6, lambda t: 1.0, lambda t: 0.5)
    assert M.walking_speed(samples(still))["reason"] == "no_walk"
    ts, _ = still.frames()
    assert M.walking_speed(M.track_samples(ts, [None] * len(ts), synth.ASPECT))["reason"] == "out_of_view"


def test_walking_with_dropout_and_noise():
    r = M.walking_speed(samples(synth.walk_script(1.0, 4.0, lead=1.0, jitter=0.004, seed=4), dropouts=[(3.0, 3.3)]))
    assert r["reported"] and r["speed_m_s"] == pytest.approx(1.0, rel=0.10)


# ---------------------------------------------------------------- review fixes

def test_walking_single_crossing_quality_over_in_view_span():
    scr = synth.walk_script(1.0, 3.0, lead=2.0, tail=2.0, jitter=0.002, seed=8)
    s = samples(scr, dropouts=[(0.0, 2.0), (5.0, 7.0)])
    assert sum(1 for x in s if x.ok) < 0.6 * len(s)       # the old whole-recording quality would have withheld this
    r = M.walking_speed(s)
    assert r["reported"] and r["speed_m_s"] == pytest.approx(1.0, rel=0.10) and r["quality"] > 0.9


def test_tug_stand_wobble_sit_then_walk():
    scr, ev, k = synth.tug_script(reaction=5.0, seed=11)
    wob = [(0, 0), (2.3, 0), (3.1, 1), (3.4, 1), (4.2, 0), (7, 0)]
    base = scr.prog
    scr.prog = lambda t: max(base(t), synth._interp_knots(t, wob))
    s = samples(scr)
    m = M.RepMachine()
    for x in s:
        m.update(x)
    assert len(m.stands) == 2 and len(m.seats) == 2           # wobble: stood, sat; then the real walk
    r = M.tug(s, ev["go"])
    assert r["reported"], r
    ph = r["phases_s"]
    assert sum(ph.values()) == pytest.approx(r["time_s"], abs=1e-9)
    assert ph["rise"] == pytest.approx(ev["rise_end"] - ev["go"], abs=0.3)   # rise ends at the stand that starts the walk
    assert r["time_s"] == pytest.approx(cross_t(k, MP.sit_deg, False, ev["sit_start"]) - ev["go"], abs=0.4)


def test_sts_reps_bounded():
    d = PARAMS.mobility.model_dump()
    assert mobility_errors({**d, "sts_reps": 51}) and mobility_errors({**d, "sts_reps": 50}) == []
