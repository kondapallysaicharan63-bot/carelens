"""W8 rhythmic hand movement (3-12 Hz): params, accuracy on synthetic hands, the noise floor, gating. Synthetic data only."""
from __future__ import annotations

import json

import pytest

from carelens.params import PARAMS, Tremor, tremor_errors
from carelens.tremor import run_tremor, synth_hand

TP = PARAMS.tremor


def run(**kw) -> dict:
    return run_tremor(synth_hand(**kw))


def test_params_validate_and_reject_inconsistent_sets():
    d = TP.model_dump()
    assert tremor_errors(d) == []
    for bad in ({**d, "taper_stop_hi_hz": 15.5}, {**d, "taper_stop_lo_hz": 3.5}, {**d, "motion_band_hi_hz": 2.5},
                {**d, "min_rate_hz": 24.0}, {**d, "fs_hz": 20.0}, {**d, "min_window_s": 20.0}, {**d, "edge_trim_s": 3.0},
                {**d, "min_prominence": 0.0}, {**d, "noise_floor_pct": 0.0}, {**d, "zero_pad_factor": 0}):
        assert tremor_errors(bad)
        with pytest.raises(ValueError):
            Tremor.model_validate(bad)
    assert (TP.band_lo_hz, TP.band_hi_hz, TP.taper_stop_lo_hz, TP.taper_stop_hi_hz) == (3.0, 12.0, 2.0, 13.5)


@pytest.mark.parametrize("hz", [4.0, 6.0, 9.0, 11.0])
@pytest.mark.parametrize("amp", [1.5, 3.0, 6.0])
@pytest.mark.parametrize("jitter", [0.003, 0.005])
@pytest.mark.parametrize("drift", [0.0, 3.0])
def test_frequency_and_amplitude_accuracy(hz, amp, jitter, drift):
    r = run(hz=hz, amp_rms_pct=amp, jitter=jitter, drift_pct=drift, seed=int(hz * 10 + amp * 7 + drift), angle_deg=15.0 + 10.0 * hz)
    assert r["reason"] == "ok" and r["reported"]
    assert abs(r["freq_hz"] - hz) <= 0.3
    assert abs(r["amp_pct"] / amp - 1.0) <= 0.15
    assert r["p2p_pct"] > 2.0 * r["amp_pct"]          # a sinusoid: peak-to-peak = 2.83 x RMS
    assert r["prominence"] >= TP.min_prominence


def test_amplitude_is_dimensionless_and_independent_of_frame_shape():
    """Hand size and frame aspect cancel: the same movement in hand-size units gives the same answer."""
    base = run(hz=6.0, amp_rms_pct=3.0, jitter=0.0015, drift_pct=1.0, seed=11)
    far = run(hz=6.0, amp_rms_pct=3.0, jitter=0.0015 * 0.5 / 1.0, drift_pct=1.0, seed=11, hand_frac=0.12)
    wide = run(hz=6.0, amp_rms_pct=3.0, jitter=0.0015, drift_pct=1.0, seed=11, w=1280, h=720)
    for r in (base, far, wide):
        assert r["reason"] == "ok" and abs(r["freq_hz"] - 6.0) <= 0.1 and abs(r["amp_pct"] / 3.0 - 1.0) <= 0.1


def test_timing_jitter_keeps_frequency_and_stays_within_amplitude_tolerance():
    r = run(hz=9.0, amp_rms_pct=3.0, seed=3, t_jitter_s=0.002)
    assert r["reason"] == "ok" and abs(r["freq_hz"] - 9.0) <= 0.3
    assert 0.80 <= r["amp_pct"] / 3.0 <= 1.05      # linear resampling reads fast movements slightly low (documented)


# ------------------------------------------------------------------ noise floor

def test_steady_hands_are_no_rhythm_at_least_95_percent():
    """MediaPipe-like jitter (0.3-0.5 % of frame height per point) + slow drift, hands from far to near. The expected result."""
    n = hits = 0
    rms_max = prom_max = 0.0
    for hf in (0.15, 0.24, 0.35):
        for jit in (0.003, 0.004, 0.005):
            for s in range(20):
                r = run(jitter=jit, hand_frac=hf, drift_pct=2.0, seed=2000 + s)
                n += 1
                hits += r["reason"] == "no_rhythm"
                rms_max, prom_max = max(rms_max, r["rms_pct"]), max(prom_max, r["prominence"])
    assert hits / n >= 0.95
    assert hits == n, f"{n - hits} of {n} steady hands were not no_rhythm"
    assert rms_max < TP.noise_floor_pct and prom_max < TP.min_prominence   # both gates independently hold here


def test_noise_floor_is_about_two_to_three_times_the_jitter_floor():
    rms = [run(jitter=0.004, seed=3000 + s)["rms_pct"] for s in range(20)]
    assert 0.25 < sum(rms) / len(rms) < 0.6          # ~0.4 % of hand size for a 0.24-frame-height hand
    assert TP.noise_floor_pct >= 2.0 * max(rms)


def test_tremor_below_the_floor_is_no_rhythm_neutral():
    r = run(hz=7.0, amp_rms_pct=0.5, jitter=0.004, seed=5)
    assert r["reason"] == "no_rhythm" and not r["reported"] and r["freq_hz"] is None and r["amp_pct"] is None


# ------------------------------------------------------------------ gating

@pytest.mark.parametrize("fps", [20.0, 24.0, 24.9])
def test_low_fps_never_reports_a_frequency(fps):
    r = run(hz=11.0, amp_rms_pct=3.0, fps=fps, seed=4)
    assert r["reason"] == "low_fps" and not r["reported"] and r["freq_hz"] is None and r["amp_pct"] is None
    assert r["rate_hz"] < TP.min_rate_hz


def test_hand_lost_at_the_end_shortens_the_window_but_still_reports():
    r = run(hz=6.0, amp_rms_pct=3.0, seed=6, lost=[(11.0, 15.0)])
    assert r["reason"] == "ok" and 10.5 < r["duration_s"] < 11.1 and abs(r["freq_hz"] - 6.0) <= 0.3


def test_hand_lost_mid_test_too_short():
    r = run(hz=6.0, amp_rms_pct=3.0, seed=6, lost=[(6.0, 9.0)])
    assert r["reason"] == "too_short" and r["duration_s"] < TP.min_window_s and r["freq_hz"] is None


def test_short_clip_too_short_and_no_hand():
    assert run(hz=6.0, amp_rms_pct=3.0, duration=8.0, seed=7)["reason"] == "too_short"
    assert run(hz=6.0, amp_rms_pct=3.0, seed=7, lost=[(0.0, 99.0)])["reason"] == "no_hand"


def test_low_score_frames_do_not_count():
    inp = synth_hand(hz=6.0, amp_rms_pct=3.0, seed=8)
    inp["score"] = [0.2] * len(inp["score"])
    assert run_tremor(inp)["reason"] == "no_hand"


def test_big_arm_movement_is_moving_not_a_frequency():
    r = run(hz=6.0, amp_rms_pct=2.5, arm_pct=20.0, seed=9)
    assert r["reason"] == "moving" and r["freq_hz"] is None and r["motion_pct"] > TP.moving_max_pct


def test_voluntary_drift_below_the_limit_is_removed():
    r = run(hz=5.0, amp_rms_pct=2.5, drift_pct=4.0, seed=10)
    assert r["reason"] == "ok" and abs(r["freq_hz"] - 5.0) <= 0.3 and abs(r["amp_pct"] / 2.5 - 1) <= 0.15


def test_only_the_last_test_s_seconds_are_used():
    r = run(hz=6.0, amp_rms_pct=3.0, duration=25.0, seed=12)
    assert r["reason"] == "ok" and r["duration_s"] <= TP.test_s + 1e-6


def test_results_never_use_condition_words():
    src = json.dumps(run(hz=6.0, amp_rms_pct=3.0, seed=13)).lower()
    for w in ("parkinson", "diagnos", "screening", "tremor disorder"):
        assert w not in src
