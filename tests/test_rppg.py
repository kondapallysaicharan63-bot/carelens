"""Face rPPG (W2) reference tests on SYNTHETIC skin-colour traces. Not a medical device.

The generator is a documented approximation (carelens/rppg.py); passing these tests says the algorithm does
what it is designed to do on that model, not how it performs on people (see PRD section 14).
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from carelens import rppg
from carelens.params import PARAMS, Params, PARAMS_PATH, rppg_errors
from carelens.rppg import SKIN_TONES, FaceRPPG, analyze_window, green_estimate, synth_face_trace

P = PARAMS.rppg
CLEAN_NOISE = 0.03   # "clean": low sensor noise, steady light, still face


def run(tr):
    fr = FaceRPPG()
    ups = []
    for i, t in enumerate(tr["t"]):
        u = fr.update(t, tr["roi_rgb"][i], tr["roi_px"][i], tr["face_motion"][i], tr["brightness"][i])
        if u:
            ups.append(u)
    return ups


def reported(ups, lo=0.0, hi=1e9):
    return [u for u in ups if u["reported"] and lo <= u["t"] <= hi]


def reasons(ups, lo, hi):
    return {u["reason"] for u in ups if lo <= u["t"] <= hi}


# ------------------------------------------------------------------ accuracy per skin tone

@pytest.mark.parametrize("skin", list(SKIN_TONES))
@pytest.mark.parametrize("bpm", [50, 60, 72, 90, 110, 130, 150])
def test_clean_accuracy_within_3_bpm(skin, bpm):
    ups = run(synth_face_trace(40, bpm, skin=skin, seed=bpm, noise=CLEAN_NOISE))
    rep = reported(ups)
    for u in rep:   # whatever is reported is right
        assert abs(u["bpm"] - bpm) <= 3.0 and abs(u["bpm_chrom"] - bpm) <= 5.0
        assert 0.0 <= u["quality"] <= 1.0
    # darker profile = weaker pulse: lower coverage is expected, recorded in the table below
    assert len(rep) >= {"light": 25, "medium": 22, "dark": 5}[skin], (skin, bpm, len(rep))
    assert ups[0]["reason"] == "settling" and reasons(ups, 0, P.settle_s - 1e-6) == {"settling"}


def test_skin_tone_coverage_table_is_recorded_honestly():
    """Per-skin-tone, per-noise coverage and error on the synthetic profiles. The point is to RECORD that darker
    profiles (weaker pulse, same sensor noise) are reported less often - nothing is tuned to equalise them."""
    bpms = (50, 72, 100, 130, 150)
    table, lines = {}, ["| noise sd | skin | coverage | MAE bpm | max err bpm |", "|---|---|---|---|---|"]
    for noise in (0.02, 0.03, 0.05):
        for skin in SKIN_TONES:
            cov, errs = [], []
            for k, bpm in enumerate(bpms):
                ups = run(synth_face_trace(40, bpm, skin=skin, seed=100 + k, noise=noise))
                post = [u for u in ups if u["t"] >= P.settle_s]
                rep = reported(post)
                cov.append(len(rep) / len(post))
                errs += [abs(u["bpm"] - bpm) for u in rep]
            table[(noise, skin)] = (float(np.mean(cov)), float(np.mean(errs)) if errs else None, max(errs, default=None))
            c, mae, mx = table[(noise, skin)]
            lines.append(f"| {noise} | {skin} | {c:.2f} | {'-' if mae is None else f'{mae:.2f}'} | {'-' if mx is None else f'{mx:.2f}'} |")
            assert mx is None or mx <= 3.0, (noise, skin, mx)   # what is reported is accurate in every cell
    text = ("Synthetic skin-tone profiles (documented approximation in carelens/rppg.py), 40 s traces, POS + CHROM gate, "
            "coverage = share of post-settling windows reported. Darker profiles have a weaker pulse and a darker DC "
            "level at the same sensor noise, and are reported less often. This is NOT evidence about real people; "
            "validate per skin tone and lighting on consented recordings.\n\n" + "\n".join(lines) + "\n")
    out = Path(__file__).resolve().parent.parent / "output"
    out.mkdir(exist_ok=True)
    (out / "rppg_skin_tone_coverage.md").write_text(text)
    print("\n" + text)
    for noise in (0.02, 0.03, 0.05):   # honest ordering: nothing was tuned to make dark look equal
        assert table[(noise, "light")][0] >= table[(noise, "dark")][0]
    assert table[(0.05, "light")][0] >= 0.85 and table[(0.05, "dark")][0] < table[(0.05, "light")][0] - 0.3
    assert table[(0.02, "dark")][0] > 0.5   # but a clean dark-skin trace is still usable on this model


# ------------------------------------------------------------------ robustness

def _windows(tr, step=30):
    ts, rgb, mo = np.array(tr["t"]), np.array(tr["roi_rgb"]), np.array(tr["face_motion"])
    for k in range(12 * 30, len(ts), step):
        s = slice(max(0, k - 12 * 30), k + 1)
        yield ts[s], rgb[s], mo[s]


def test_pos_beats_green_under_common_mode_motion_noise():
    pos_err, green_err = [], []
    for skin in ("light", "medium"):
        for k, bpm in enumerate((55, 70, 85, 100, 120, 140)):
            tr = synth_face_trace(40, bpm, skin=skin, seed=300 + k, noise=CLEAN_NOISE, motion_noise=0.02)
            for ts, rgb, mo in _windows(tr):
                pos_err.append(abs(analyze_window(ts, rgb, mo)["bpm"] - bpm))
                green_err.append(abs(green_estimate(ts, rgb)["bpm"] - bpm))
    pos_ok, green_ok = np.mean(np.array(pos_err) <= 3), np.mean(np.array(green_err) <= 3)
    assert np.mean(pos_err) < 0.25 * np.mean(green_err)
    assert pos_ok > 0.95 and pos_ok > green_ok + 0.5


def test_pos_and_chrom_reject_common_mode_intensity_but_green_does_not():
    base = synth_face_trace(30, 72, skin="light", seed=5, noise=0.0, drift=0.0)
    rgb = np.array(base["roi_rgb"])
    t = np.array(base["t"])
    k = 1.0 + 0.05 * np.sin(2 * np.pi * 1.9 * t)   # strong multiplicative intensity flicker in band
    y0, y1 = rgb, rgb * k[:, None]
    n_fft = 4096
    for f in (lambda y: rppg.pos_signal(y, 30.0, 1.6),
              lambda y: rppg.chrom_signal(y, n_fft, 30.0, P)):
        a, b = f(y0), f(y1)
        assert np.std(a - b) < 0.12 * np.std(a)
    g0, g1 = y0[:, 1], y1[:, 1]
    assert np.std(g1 - g0) > 2 * np.std(g0 - g0.mean())


def test_motion_gate_withholds_then_recovers():
    tr = synth_face_trace(55, 78, skin="medium", seed=6, noise=CLEAN_NOISE, events=[(20.0, 23.0, "motion", 0.02)])
    ups = run(tr)
    assert reasons(ups, 21, 31) == {"motion"} and not reported(ups, 21, 31)
    assert all(u["bpm"] is None for u in ups if u["reason"] == "motion")
    assert len(reported(ups, 36)) >= 15
    assert all(abs(u["bpm"] - 78) <= 3 for u in reported(ups))


def test_lighting_step_then_recovery():
    tr = synth_face_trace(50, 84, skin="light", seed=7, noise=CLEAN_NOISE, events=[(20.0, 99.0, "lightstep", 0.5)])
    ups = run(tr)
    assert "lighting" in reasons(ups, 20, 23) and not reported(ups, 20.5, 24)
    assert reasons(ups, 24, 27) == {"settling"}
    late = reported(ups, 31)
    assert len(late) >= 15 and all(abs(u["bpm"] - 84) <= 3 for u in late)
    epochs = {u["track_epoch"] for u in ups}
    assert len(epochs) == 2 and max(u["track_epoch"] for u in ups if u["t"] > 31) == 2


def test_too_dark_is_lighting_even_without_a_step():
    tr = synth_face_trace(30, 72, skin="dark", seed=8, noise=CLEAN_NOISE, events=[(0.0, 99.0, "lightstep", 0.25)])
    ups = run(tr)
    assert all(u["reason"] == "lighting" for u in ups) and not reported(ups)


def test_slow_lighting_drift_is_tolerated():
    ups = run(synth_face_trace(40, 72, skin="light", seed=9, noise=CLEAN_NOISE, drift=0.15))
    assert "lighting" not in {u["reason"] for u in ups} and len(reported(ups)) >= 25


def test_face_lost_and_small_face():
    ups = run(synth_face_trace(45, 72, skin="medium", seed=10, noise=CLEAN_NOISE, events=[(18.0, 24.0, "noface")]))
    assert reasons(ups, 18, 23) == {"no_face"} and all(u["bpm"] is None for u in ups if u["reason"] == "no_face")
    assert reasons(ups, 24, 28) == {"settling"} and len(reported(ups, 31)) >= 10
    ups = run(synth_face_trace(38, 72, skin="light", seed=11, noise=CLEAN_NOISE, events=[(15.0, 20.0, "small")]))
    assert reasons(ups, 15, 19) == {"too_small"} and len(reported(ups, 27)) >= 8
    assert max(u["track_epoch"] for u in ups) == 2   # one restart


def test_disagree_when_pos_and_chrom_differ():
    ups = run(synth_face_trace(32, 70, skin="light", seed=12, noise=0.02, interference=(100.0, 0.006, (0.2, 0.2, 1.0))))
    dis = [u for u in ups if u["reason"] == "disagree"]
    assert len(dis) >= 20 and not reported(ups)
    for u in dis:
        assert abs(u["bpm"] - u["bpm_chrom"]) > P.agree_bpm and u["snr"] >= P.min_snr
        assert abs(u["bpm"] - 70) <= 3 and abs(u["bpm_chrom"] - 100) <= 3   # POS correct, CHROM pulled to the flicker


def test_pure_noise_is_never_reported():
    for seed in range(8):
        ups = run(synth_face_trace(40, 72, seed=seed, noise=0.1, pulse_vec=(0, 0, 0)))
        assert not reported(ups), seed


def test_gap_longer_than_limit_is_withheld():
    tr = synth_face_trace(40, 72, skin="light", seed=13, noise=CLEAN_NOISE)
    keep = [i for i, t in enumerate(tr["t"]) if not (20.0 <= t < 20.8)]
    tr = {k: ([v[i] for i in keep] if isinstance(v, list) else v) for k, v in tr.items()}
    ups = run(tr)
    assert "gaps" in reasons(ups, 20.8, 32)


def test_non_monotonic_timestamps_are_ignored():
    fr = FaceRPPG()
    assert fr.update(1.0, (180, 125, 95), 1000, 0.0, 130) is None
    assert fr.update(0.5, (180, 125, 95), 1000, 0.0, 130) is None
    assert fr.update(1.0, (180, 125, 95), 1000, 0.0, 130) is None


def test_harmonic_selection_regression_sweep():
    """>= 300 analysed windows over 50-150 bpm, light/medium skin, with and without common-mode motion noise:
    no reported window is off by more than 10 % (i.e. no half/double-rate pick gets through the gates)."""
    n_windows, bad = 0, []
    bpms = (50, 58, 66, 74, 82, 90, 98, 106, 114, 122, 130, 140, 150)
    for k, bpm in enumerate(bpms):
        for skin, mn in (("light", 0.0), ("medium", 0.02), ("light", 0.04)):
            ups = run(synth_face_trace(40, bpm, skin=skin, seed=500 + k, noise=0.04, motion_noise=mn))
            n_windows += sum(u["bpm"] is not None for u in ups)
            bad += [(bpm, u["t"], u["bpm"]) for u in reported(ups) if abs(u["bpm"] - bpm) > 0.10 * bpm]
    assert n_windows >= 300
    assert bad == []


# ------------------------------------------------------------------ quality, params, integration

def test_quality_is_monotone_and_bounded():
    q = rppg.quality_score
    assert q(4, 1, 0.1, P) < q(40, 1, 0.1, P)           # SNR
    assert q(10, 4, 0.1, P) < q(10, 0.5, 0.1, P)         # agreement
    assert q(10, 1, 0.4, P) < q(10, 1, 0.05, P)          # motion
    assert 0.0 <= q(1e9, 0, 0, P) <= 1.0 and q(0.0, 99, 9, P) >= 0.0


def test_not_analysed_windows_have_zero_quality_and_no_bpm():
    ups = run(synth_face_trace(30, 72, seed=1, events=[(10.0, 14.0, "noface")]))
    for u in ups:
        if u["reason"] in ("settling", "no_face"):
            assert u["quality"] == 0.0 and u["bpm"] is None and u["bpm_chrom"] is None and not u["reported"]


def test_rppg_params_validation():
    raw = json.loads(PARAMS_PATH.read_text())
    assert rppg_errors(raw["rppg"]) == []
    for key, val in (("q_w_snr", 0.9), ("min_brightness", 250.0), ("pos_window_s", 0.1), ("motion_frac_max", 1.5),
                     ("band_hi_hz", 5.0), ("skin_cr_min", 200)):
        bad = copy.deepcopy(raw)
        bad["rppg"][key] = val
        assert rppg_errors(bad["rppg"]), key
        with pytest.raises(ValueError):
            Params.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["rppg"]["roi_landmarks"]["forehead"] = [1, 2, 2]
    with pytest.raises(ValueError):
        Params.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["rppg"]["extra"] = 1
    with pytest.raises(ValueError):
        Params.model_validate(bad)


# FaceLandmarker (478-point) indices of the eyes, irises, brows, lips and nose outline; the ROI polygons avoid them.
_NON_SKIN = {0, 1, 2, 4, 5, 6, 7, 13, 14, 17, 19, 33, 37, 39, 40, 45, 46, 48, 52, 53, 55, 61, 63, 64, 65, 66, 70, 78,
             80, 81, 82, 84, 87, 88, 91, 94, 95, 97, 98, 105, 107, 115, 133, 144, 145, 146, 153, 154, 155, 157, 158,
             159, 160, 161, 163, 168, 173, 178, 181, 185, 191, 195, 197, 220, 246, 249, 263, 267, 269, 270, 275, 276,
             278, 282, 283, 285, 291, 293, 294, 295, 296, 300, 308, 310, 311, 312, 314, 317, 318, 321, 324, 326, 327,
             334, 336, 344, 362, 373, 374, 375, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 402, 405, 409, 415,
             440, 466, 469, 470, 471, 472, 474, 475, 476, 477}


def test_roi_polygons_avoid_eyes_brows_lips():
    lm = P.roi_landmarks
    allidx = lm.forehead + lm.left_cheek + lm.right_cheek
    assert len(set(allidx)) == len(allidx) and set(allidx).isdisjoint(_NON_SKIN)
    assert all(0 <= i <= 477 for i in allidx)


def test_vitals_pos_method_on_a_trace_and_green_still_available():
    from carelens.vitals import HeartRateEstimator
    tr = synth_face_trace(50, 80, skin="medium", seed=14, noise=CLEAN_NOISE, fs=15.0)
    est = HeartRateEstimator()
    assert est.method == "pos"
    for i, t in enumerate(tr["t"]):
        est.update_rgb(t, tr["roi_rgb"][i], tr["roi_px"][i], tr["brightness"][i], 0.0)
    rep = [e for e in est.estimates if e["reported"]]
    assert len(rep) >= 15 and all(abs(e["value"] - 80) <= 3 for e in rep)
    est.update_rgb(60.0, (180, 125, 95), 1000, 5.0, 0.5)   # motion above the pose gate resets
    assert est._face.valid_seconds(60.0) == 0.0
    with pytest.raises(ValueError):
        HeartRateEstimator(method="bogus")
    assert HeartRateEstimator(method="green").window_s == PARAMS.vitals.heart_rate.window_s


# ------------------------------------------------------------------ review fixes

def test_dead_channel_never_gives_nan_or_a_reported_reading():
    tr = synth_face_trace(30, 72, skin="light", seed=21, noise=CLEAN_NOISE)
    for i, t in enumerate(tr["t"]):
        if 14.0 <= t < 16.2:   # red channel reads 0 for 2.2 s (longer than a POS sub-window)
            tr["roi_rgb"][i][0] = 0.0
    y = np.array(tr["roi_rgb"])[: 12 * 30]
    y[100:200, 0] = 0.0
    with np.errstate(all="raise"):   # the unguarded 0/0 raised/NaN'd here
        assert np.all(np.isfinite(rppg.pos_signal(y, 30.0, P.pos_window_s)))
    ups = run(tr)
    for u in ups:
        for k in ("bpm", "bpm_chrom", "snr"):
            assert u[k] is None or np.isfinite(u[k]), (u["t"], k)
        assert np.isfinite(u["quality"])
        if u["reported"]:
            assert np.isfinite(u["bpm"])
    # non-finite window results are low_signal, never ok
    fr = FaceRPPG()
    fr._state = "ok"
    import carelens.rppg as R
    orig = R.analyze_window
    R.analyze_window = lambda *a, **k: {"gaps": False, "motion": False, "bpm": float("nan"), "bpm_chrom": 70.0,
                                        "snr": 9.0, "delta": float("nan"), "motion_mean": 0.0, "span_s": 12.0, "quality": 0.5}
    try:
        fr._valid_since = 0.0
        out = fr._emit(20.0)
    finally:
        R.analyze_window = orig
    assert out["reason"] == "low_signal" and not out["reported"] and out["bpm"] is None


def test_skin_filter_is_its_own_reason_not_too_small():
    ups = run(synth_face_trace(38, 72, skin="light", seed=22, noise=CLEAN_NOISE, events=[(15.0, 20.0, "skinless")]) | {})
    # run() ignores skin_frac, so replay with it
    tr = synth_face_trace(38, 72, skin="light", seed=22, noise=CLEAN_NOISE, events=[(15.0, 20.0, "skinless")])
    out = rppg.run_face_rppg(tr)["updates"]
    assert reasons(out, 15, 19) == {"skin_filter"} and all(u["bpm"] is None for u in out if u["reason"] == "skin_filter")
    assert reasons(out, 20, 24) == {"settling"} and len(reported(out, 27)) >= 8
    assert ups  # default skin_frac=1 leaves the trace usable


def test_estimates_record_their_analysed_span():
    ups = run(synth_face_trace(40, 72, skin="light", seed=23, noise=CLEAN_NOISE))
    an = [u for u in ups if u["bpm"] is not None]
    assert an[0]["span_s"] < P.window_s and an[-1]["span_s"] == P.window_s   # short windows right after settling
    assert all(u["span_s"] is None for u in ups if u["reason"] in ("settling", "no_face"))


def test_pipeline_threshold_is_separate_and_small_distant_faces_report():
    from carelens.vitals import HeartRateEstimator
    assert PARAMS.vitals.pipeline_min_roi_px < P.min_roi_px
    est = HeartRateEstimator()
    assert est._face.p.min_roi_px == PARAMS.vitals.pipeline_min_roi_px
    tr = synth_face_trace(45, 76, skin="medium", seed=24, noise=CLEAN_NOISE, fs=15.0)
    for i, t in enumerate(tr["t"]):   # a ~80-pixel patch (distant subject): too small for the web gate, fine here
        est.update_rgb(t, tr["roi_rgb"][i], 80, tr["brightness"][i], 0.0)
    rep = [e for e in est.estimates if e["reported"]]
    assert len(rep) >= 15 and all(abs(e["value"] - 76) <= 3 for e in rep) and all("span_s" in e for e in est.estimates)
    tiny = HeartRateEstimator()
    for i, t in enumerate(tr["t"]):
        tiny.update_rgb(t, tr["roi_rgb"][i], 20, tr["brightness"][i], 0.0)
    assert not tiny.estimates   # below the pipeline gate: nothing is analysed


def test_pipeline_passes_real_motion_into_quality():
    from carelens.vitals import HeartRateEstimator
    tr = synth_face_trace(40, 76, skin="light", seed=25, noise=CLEAN_NOISE, fs=15.0)
    q = {}
    for m in (0.0, 0.06):   # body sizes/s; the pose gate is 0.08
        est = HeartRateEstimator()
        for i, t in enumerate(tr["t"]):
            est.update_rgb(t, tr["roi_rgb"][i], 500, tr["brightness"][i], m)
        q[m] = np.mean([e["quality"] for e in est.estimates if e["reported"]])
    assert q[0.06] < q[0.0] - 0.03


def test_synth_defaults_and_accuracy_legacy_window():
    import inspect
    from carelens import synth
    assert inspect.signature(synth.render).parameters["noise_sd"].default == 4.0
