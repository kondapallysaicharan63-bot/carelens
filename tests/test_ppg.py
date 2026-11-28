"""Contact PPG (W1) reference tests: synthetic fingertip traces. Not a medical device."""
import math

import numpy as np
import pytest

from carelens import ppg
from carelens.params import PARAMS
from carelens.ppg import ContactPPG, synth_trace

P = PARAMS.contact_ppg


def run(tr):
    cp = ContactPPG()
    ups = []
    for i, t in enumerate(tr["t"]):
        u = cp.update(t, tr["r"][i], tr["g"][i], tr["b"][i], tr["clip_frac"][i])
        if u:
            ups.append(u)
    return ups, cp


def reported(ups, after=0.0):
    return [u for u in ups if u["reported"] and u["t"] >= after]


@pytest.mark.parametrize("bpm", [45, 60, 72, 90, 120, 150, 180])
def test_clean_bpm_within_1(bpm):
    ups, _ = run(synth_trace(40, bpm, seed=bpm))
    rep = reported(ups)
    assert len(rep) >= 25
    for u in rep:
        assert abs(u["bpm"] - bpm) <= 1.0 and abs(u["bpm_ibi"] - bpm) <= 2.0
        assert 0.0 <= u["quality"] <= 1.0 and u["channel"] == "R"
    assert ups[0]["reason"] == "settling"
    assert all(u["reason"] == "settling" for u in ups if u["t"] < P.settle_s)


def test_respiratory_modulation_and_baseline_wander():
    for kw in ({"resp_mod": 0.3}, {"wander": 6.0}, {"resp_mod": 0.3, "wander": 6.0}):
        ups, _ = run(synth_trace(40, 78, seed=3, **kw))
        rep = reported(ups)
        assert len(rep) >= 20, kw
        assert all(abs(u["bpm"] - 78) <= 1.0 for u in rep), kw


@pytest.mark.parametrize("noise", [0.3, 0.6, 1.0])
def test_white_noise_levels(noise):
    errs = []
    for seed in range(4):
        ups, _ = run(synth_trace(40, 84, seed=seed, noise=noise))
        errs += [abs(u["bpm"] - 84) for u in reported(ups)]
    assert errs, "moderate noise should still report"
    assert max(errs) <= 2.0   # whatever is reported is right


def test_pure_noise_is_never_reported():
    for seed in range(6):
        tr = synth_trace(40, 84, seed=seed, amp=0.0, noise=0.5)
        ups, _ = run(tr)
        assert not reported(ups), seed


def test_irregular_timing_jitter_drops_hiccup():
    tr = synth_trace(50, 72, seed=5, jitter_s=0.008, drop_frac=0.05, hiccup=(25.0, 0.3))
    ups, _ = run(tr)
    rep = reported(ups)
    assert len(rep) >= 30
    assert all(abs(u["bpm"] - 72) <= 1.0 for u in rep)


def test_gap_longer_than_limit_is_withheld():
    tr = synth_trace(50, 72, seed=5, hiccup=(25.0, 0.8))
    ups, _ = run(tr)
    assert any(u["reason"] == "gaps" for u in ups if 25 <= u["t"] <= 36)
    assert not reported(ups, 26.0) or all(u["t"] > 35.5 for u in reported(ups, 26.0))


def test_motion_burst_withheld_then_recovers():
    tr = synth_trace(70, 72, seed=6, events=[(30.0, 36.0, "burst")])
    ups, _ = run(tr)
    assert reported(ups, 0) and all(u["t"] < 31 for u in reported(ups) if u["t"] < 40)
    during = [u for u in ups if 31 <= u["t"] <= 36 + P.settle_s - 0.5]
    assert during and not any(u["reported"] for u in during)
    late = reported(ups, 36 + P.settle_s + 2)
    assert len(late) >= 15 and all(abs(u["bpm"] - 72) <= 1.0 for u in late)


def test_saturation_reason():
    tr = synth_trace(60, 72, seed=7, events=[(20.0, 40.0, "saturate")])
    ups, _ = run(tr)
    sat = [u for u in ups if 21 <= u["t"] <= 39]
    assert sat and all(u["reason"] == "saturated" and not u["reported"] and u["bpm"] is None for u in sat)
    assert reported(ups, 40 + P.settle_s + 1)


def test_finger_lifted_reason():
    tr = synth_trace(60, 72, seed=8, events=[(20.0, 30.0, "lift")])
    ups, _ = run(tr)
    lift = [u for u in ups if 21 <= u["t"] <= 29]
    assert lift and all(u["reason"] == "no_finger" and not u["reported"] for u in lift)
    assert reported(ups, 30 + P.settle_s + 1)


def test_low_perfusion_withheld():
    tr = synth_trace(40, 72, seed=9, events=[(0.0, 40.0, "lowperf")])
    ups, _ = run(tr)
    assert not reported(ups)
    assert any(u["reason"] == "low_signal" for u in ups)


def test_harmonic_trap_picks_fundamental():
    # 2nd harmonic comparable to the fundamental: reported and correct
    ups, _ = run(synth_trace(40, 52, seed=52, harmonic=0.5, amp=2.0))
    rep = reported(ups)
    assert len(rep) >= 20 and all(abs(u["bpm"] - 52) <= 1.0 for u in rep)
    # weaker harmonic at other rates: the spectral fundamental is still right in every analysed window
    for bpm in (45, 60, 66):
        ups, _ = run(synth_trace(40, bpm, seed=bpm, harmonic=0.3, amp=2.0))
        late = [u for u in ups if u["t"] > 12 and u["bpm"] is not None]
        assert late and all(abs(u["bpm"] - bpm) <= 2.0 for u in late), bpm
        assert all(abs(u["bpm"] - bpm) <= 2.0 for u in reported(ups)), bpm
    # (a 2nd harmonic that dominates by 2x or more is a genuinely 2x-rate waveform: not testable as a trap)


def test_no_rounding_and_contact_epoch():
    ups, _ = run(synth_trace(30, 72, seed=1, events=[(15.0, 17.0, "lift")]))
    rep = reported(ups)
    assert any(abs(u["bpm"] * 10 - round(u["bpm"] * 10)) > 1e-6 for u in rep)   # full precision
    ep = [u["contact_epoch"] for u in ups]
    assert ep[0] == 1 and ep == sorted(ep) and ep[-1] == 2 and ep.count(1) >= 14


def test_params_validation():
    from pydantic import ValidationError
    base = PARAMS.contact_ppg.model_dump()
    ppg_cls = type(PARAMS.contact_ppg)
    ppg_cls(**base)
    for k, v in (("band_lo_hz", 0.4), ("taper_stop_hi_hz", 2.0), ("q_w_pi", 0.5), ("min_snr", 0.0),
                 ("pi_min", 5.0), ("q_snr_full_db", 1.0), ("taper_stop_hi_hz", 20.0)):
        with pytest.raises(ValidationError):
            ppg_cls(**{**base, k: v})


def test_halving_doubling_regression_sweep():
    """No reported value off by > 10% across rates, noise, wander, RR and frame jitter (dicrotic waveform)."""
    n_win = n_rep = clean_win = clean_rep = 0
    cases = [(150, 3, 0.5, 4.0, 0.0, {"drop_frac": 0.05}), (170, 3, 0.5, 4.0, 0.0, {"drop_frac": 0.05}),
             (170, 3, 0.5, 4.0, 0.0, {"jitter_s": 0.008, "drop_frac": 0.05}),
             (170, 1, 0.5, 4.0, 0.0, {"jitter_s": 0.008, "drop_frac": 0.05})]
    for bpm in (45, 60, 90, 130, 150, 170, 180):
        for seed in (0, 1):
            for noise in (0.0, 0.25, 0.5):
                for wander in (0.0, 4.0):
                    for rrj in (0.0, 0.05):
                        cases.append((bpm, seed, noise, wander, rrj, {"drop_frac": 0.05, "jitter_s": 0.008} if rrj else {}))
    for bpm, seed, noise, wander, rrj, kw in cases:
        bt = None
        if rrj:
            rr = 60.0 / bpm * (1 + rrj * np.random.default_rng(seed + 7).standard_normal(80))
            bt = np.cumsum(rr) - 2.0
        ups, _ = run(synth_trace(40, bpm, seed=seed, noise=noise, wander=wander, amp=2.0, beat_times=bt, **kw))
        for u in (u for u in ups if u["t"] > 12):
            n_win += 1
            if u["reported"]:
                n_rep += 1
                assert abs(u["bpm"] - bpm) <= 0.1 * bpm, (bpm, seed, noise, wander, rrj, kw, u["bpm"])
            if noise <= 0.25:
                clean_win += 1
                clean_rep += u["reported"]
    assert n_win >= 300 and n_rep > 0
    assert clean_rep / clean_win >= 0.8


def test_ibi_accuracy_variable_rr():
    rng = np.random.default_rng(42)
    rr = np.clip(0.85 + 0.08 * np.sin(2 * np.pi * np.arange(80) / 9.0) + rng.normal(0, 0.03, 80), 0.6, 1.1)
    beats = np.cumsum(rr) - 2.0
    tr = synth_trace(60, 70, beat_times=beats, seed=1)
    cp = ContactPPG()
    last = None
    for i, t in enumerate(tr["t"]):
        u = cp.update(t, tr["r"][i], tr["g"][i], tr["b"][i], tr["clip_frac"][i])
        if u and u["reported"] and t > 50:
            last = (t, list(cp.last_ibis_ms))
    assert last is not None
    t_end, ibis = last
    truth = np.diff(beats[(beats > t_end - 10.0) & (beats <= t_end)]) * 1000.0
    # match each estimated IBI to the true IBI whose centre is nearest
    bt = beats[(beats > t_end - 10.0) & (beats <= t_end)]
    assert len(ibis) >= len(truth) - 2
    # window start = t_end - (n-1)/fs; compare in order after aligning on count
    k = min(len(ibis), len(truth))
    err = np.abs(np.array(ibis[-k:]) - truth[-k:])
    assert err.mean() < 15.0, err


def test_non_monotonic_timestamp_ignored():
    cp = ContactPPG()
    assert cp.update(1.0, 200, 60) is None
    assert cp.update(0.5, 200, 60) is None and cp.update(1.0, 200, 60) is None


def test_quality_monotone_and_bounded():
    q = lambda snr, d, pi: ppg.quality_score(snr, d, pi, P)
    assert q(1e6, 0.0, 1.0) <= 1.0 and q(0.01, None, 1e6) >= 0.0
    assert q(10, 1, 1) < q(30, 1, 1) < q(300, 1, 1)
    assert q(30, 4, 1) < q(30, 2, 1) < q(30, 0, 1)
    assert q(30, 1, 0.12) < q(30, 1, 0.3) < q(30, 1, 1.0) and q(30, 1, 8.0) > q(30, 1, 14.0)


def test_building_blocks():
    x = np.random.default_rng(1).normal(size=64)
    # FFT-domain taper
    f = np.array([0.0, 0.5, 0.6, 0.7, 1.5, 3.0, 3.25, 3.5, 4.0])
    g = ppg.taper(f, P)
    assert list(g[[0, 1, 3, 4, 5, 7, 8]]) == [0, 0, 1, 1, 1, 0, 0] and abs(g[2] - 0.5) < 1e-12 and abs(g[6] - 0.5) < 1e-12
    # parabolic interpolation recovers an exact parabola vertex
    assert abs(ppg.parabolic_lin(1.0, 3.0, 2.0) - 0.5 * (1 - 2) / (1 - 6 + 2)) < 1e-12
    assert ppg.fft_size(300, 4) == 2048 and ppg.fft_size(256, 4) == 1024
    rel, mean = ppg.detrend_relative(100 + 0.5 * np.arange(64) + x)
    assert abs(np.sum(rel * (np.arange(64) - 31.5))) < 1e-9 and mean > 100
    assert abs(ppg.percentile(np.arange(11.0), 95) - 9.5) < 1e-12
