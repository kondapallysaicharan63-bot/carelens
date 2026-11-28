"""W6 HRV (RMSSD / SDNN from a fingertip segment): estimation error against known RR series, artifact handling, gates.

Synthetic data only. The measured bias table is printed with ``pytest -s tests/test_hrv.py -k bias``.
Timing noise of a 30 fps camera inflates RMSSD (a peak is located to ~ms, successive differences square that); the
bias is real and is reported, not hidden: it grows with noise and shrinks as the true RMSSD grows.
"""
import numpy as np
import pytest

from carelens.hrv import (analyze_hrv, hrv_result, keep_mask, hrv_metrics, synth_rr, truth_hrv, with_ectopics,
                          without_beats)
from carelens.params import PARAMS
from carelens.ppg import ContactPPG, synth_trace

LEVELS = {"clean": dict(noise=0.05), "moderate": dict(noise=0.08, jitter_s=0.008, resp_mod=0.05),
          "w1_default": dict(noise=0.2)}   # W1's synth_trace default noise: reported, not asserted tight
SEEDS = (1, 2, 3)


def _run(rmssd, seed, dur=75.0, ect=(), mis=(), fs=30.0, mean_rr=850.0, **kw):
    bt0 = synth_rr(dur + 10, rmssd, seed=seed, mean_rr_ms=mean_rr)
    bt = without_beats(with_ectopics(bt0, ect), mis) if (ect or mis) else bt0
    tr = synth_trace(dur, beat_times=bt, seed=seed, fs=fs, **kw)
    cp = ContactPPG(hrv=PARAMS.hrv)
    for i, t in enumerate(tr["t"]):
        cp.update(t, tr["r"][i], tr["g"][i], tr["b"][i], tr["clip_frac"][i])
    m = (bt0 > tr["t"][0]) & (bt0 < tr["t"][-1])
    return hrv_result(cp), truth_hrv(np.diff(bt0[m]) * 1000.0)


def test_synth_rr_has_the_requested_rmssd():
    for r in (15, 30, 50, 80):
        assert truth_hrv(np.diff(synth_rr(70, r, seed=4)) * 1000)[0] == pytest.approx(r, rel=1e-9)


def test_keep_mask_and_metrics_rules():
    hp = PARAMS.hrv
    ib = [800.0] * 12
    ib[5] = 3000.0   # out of range
    ib[8] = 1100.0   # local-median deviation 37 %
    k = keep_mask(ib, hp)
    assert [i for i, x in enumerate(k) if not x] == [5, 8]
    rmssd, sdnn, hr = hrv_metrics([800, 810, 1500, 820, 830], [True, True, False, True, True])
    assert rmssd == pytest.approx(np.sqrt((10 ** 2 + 10 ** 2) / 2))   # 810->1500->820 is NOT bridged
    assert sdnn == pytest.approx(np.std([800, 810, 820, 830], ddof=1))
    assert hr == pytest.approx(60000 / 815)


@pytest.mark.parametrize("level", ["clean", "moderate"])
@pytest.mark.parametrize("rmssd", [15, 30, 50, 80])
def test_error_bounds(level, rmssd):
    for seed in SEEDS:
        res, (tr, ts) = _run(rmssd, seed, **LEVELS[level])
        assert res["reported"], (res["reason"], res["kept_frac"])
        assert abs(res["rmssd_ms"] - tr) <= max(6.0, 0.20 * tr), (seed, res["rmssd_ms"], tr)
        assert abs(res["sdnn_ms"] - ts) <= max(6.0, 0.15 * ts), (seed, res["sdnn_ms"], ts)
        assert res["mean_hr_bpm"] == pytest.approx(60000 / 850, rel=0.03)


def test_bias_table():
    rows = []
    for name, kw in LEVELS.items():
        for rmssd in (15, 30, 50, 80):
            er, es = [], []
            for seed in SEEDS:
                res, (tr, ts) = _run(rmssd, seed, **kw)
                assert res["reported"]
                er.append(res["rmssd_ms"] - tr)
                es.append(res["sdnn_ms"] - ts)
            rows.append((name, rmssd, float(np.mean(er)), float(np.max(np.abs(er))), float(np.mean(es)), float(np.max(np.abs(es)))))
    print("\nlevel        true_RMSSD  RMSSD bias  max|err|  SDNN bias  max|err|   (ms, 30 fps, 3 seeds)")
    for r in rows:
        print(f"{r[0]:<12} {r[1]:>9}  {r[2]:>+10.1f} {r[3]:>9.1f} {r[4]:>+10.1f} {r[5]:>9.1f}")
    assert all(r[2] > 0 for r in rows), "timing noise should inflate RMSSD, never deflate it"
    assert max(r[3] for r in rows if r[0] == "w1_default") <= 20.0


def test_ectopic_and_missed_beats_are_rejected():
    for seed in SEEDS:
        res, (tr, ts) = _run(30, seed, ect=(12, 45), mis=(25, 62), noise=0.05)
        assert res["reported"], res
        assert res["n_kept"] < res["n_beats"] - 1   # the damaged intervals were dropped
        assert abs(res["rmssd_ms"] - tr) <= max(6.0, 0.20 * tr) + 1.0
        assert abs(res["sdnn_ms"] - ts) <= max(6.0, 0.15 * ts)


def test_too_many_artifacts_is_withheld():
    res, _ = _run(30, 1, ect=(8, 17, 26, 35, 44, 53, 62, 71, 78), noise=0.05)
    assert (res["reported"], res["reason"], res["rmssd_ms"]) == (False, "too_many_artifacts", None)
    assert res["kept_frac"] < PARAMS.hrv.min_kept_frac


def test_too_short_low_fps_unsteady():
    res, _ = _run(30, 1, dur=50.0, noise=0.05)
    assert (res["reported"], res["reason"]) == (False, "too_short") and res["duration_s"] < 60
    res, _ = _run(30, 1, fs=20.0, noise=0.05)
    assert (res["reported"], res["reason"]) == (False, "low_fps")
    tr = synth_trace(70, beat_times=synth_rr(80, 30, seed=1), seed=1, noise=0.05)
    steady = analyze_hrv(tr["t"], tr["r"], tr["g"], 62, 50)       # 81 % of the 1 s updates steady
    assert steady["reported"]
    res = analyze_hrv(tr["t"], tr["r"], tr["g"], 62, 49)          # 79 %
    assert (res["reported"], res["reason"], res["rmssd_ms"]) == (False, "unsteady", None)


def test_contact_loss_restarts_the_segment():
    bt = synth_rr(90, 30, seed=2)
    tr = synth_trace(80, beat_times=bt, seed=2, noise=0.05, events=((50.0, 52.0, "lift"),))
    cp = ContactPPG(hrv=PARAMS.hrv)
    for i, t in enumerate(tr["t"]):
        cp.update(t, tr["r"][i], tr["g"][i], tr["b"][i], tr["clip_frac"][i])
    res = hrv_result(cp)
    assert res["reason"] == "too_short" and res["duration_s"] < 40   # only the ~28 s after the lift


def test_segment_is_frozen_at_max_duration():
    res, _ = _run(30, 1, dur=150.0, noise=0.05)
    assert res["reported"] and res["duration_s"] <= PARAMS.hrv.max_duration_s + 0.1


def test_analyze_hrv_degenerate_inputs():
    assert analyze_hrv([], [], [], 0, 0)["reason"] == "too_short"
    assert analyze_hrv([0.0], [1.0], [1.0], 0, 0)["duration_s"] == 0.0


def test_segment_starts_after_settling():
    res, _ = _run(30, 1, dur=66.0, noise=0.05)   # 66 s of contact = 58 s after the 8 s settling: still too short
    assert res["reason"] == "too_short" and res["duration_s"] == pytest.approx(66.0 - PARAMS.contact_ppg.settle_s, abs=0.2)
    res, _ = _run(30, 1, dur=70.0, noise=0.05)
    assert res["reported"] and res["duration_s"] < 62.1


def test_slow_heart_rate_collects_then_reports():
    res, _ = _run(30, 1, dur=69.0, mean_rr=1250.0, noise=0.05)   # 48 bpm: ~48 beats in 61 s, fewer than 50 kept
    assert (res["reported"], res["reason"]) == (False, "collecting") and res["n_kept"] < PARAMS.hrv.min_kept_beats
    assert res["kept_frac"] >= PARAMS.hrv.min_kept_frac
    res, _ = _run(30, 1, dur=140.0, mean_rr=1250.0, noise=0.05)  # segment frozen at max length: 96 beats
    assert res["reported"] and res["duration_s"] <= PARAMS.hrv.max_duration_s + 0.1
    # a full-length segment that still lacks beats is withheld
    ts = np.arange(0, 121, 1 / 30.0)
    steady = analyze_hrv(ts, np.full(ts.size, 200.0), np.full(ts.size, 60.0), 100, 100)
    assert steady["reason"] == "too_many_artifacts"
