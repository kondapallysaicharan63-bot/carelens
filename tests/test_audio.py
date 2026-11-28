"""Audio Tier 1 (W4) reference tests on SYNTHETIC signals only. Not a medical device."""
import json

import numpy as np
import pytest

from carelens import audio as A
from carelens.fixtures import FIXTURE_DIR as FX
from carelens.params import PARAMS, Params, PARAMS_PATH

P = PARAMS.audio


def run_env(env, clip=None, speech=()):
    e = A.BreathSoundRate()
    for t, s in speech:
        e.push_speech(t, s)
    return [u for i, v in enumerate(env) if (u := e.push(i / 50.0, v, 0.0 if clip is None else clip[i]))]


def reported(ups):
    return [u for u in ups if u["reported"]]


def test_params_validate_and_reject_bad():
    raw = json.loads(PARAMS_PATH.read_text())
    raw["audio"]["breathing"]["bogus"] = 1
    with pytest.raises(Exception):
        Params.model_validate(raw)
    raw = json.loads(PARAMS_PATH.read_text())
    raw["audio"]["cough"]["off_thr"] = 0.9
    with pytest.raises(Exception):
        Params.model_validate(raw)


def test_envelope_batch_equals_stream_and_frame_count():
    x = A.synth_shaped_noise(2.0, seed=1)
    env, clip = A.envelope(x)
    assert env.size == A.n_frames(x.size) == 1 + (x.size - 512) // 320
    st = A.EnvelopeStream()
    out = []
    for i in range(0, x.size, 1000):
        out += st.push(x[i:i + 1000])
    assert len(out) == env.size
    assert np.allclose([o[1] for o in out], env) and np.allclose([o[2] for o in out], clip)
    assert abs(out[0][0] - 512 / 16000) < 1e-12 and abs(out[1][0] - out[0][0] - 0.02) < 1e-12


def test_envelope_tracks_level_and_silence_is_flat():
    assert np.ptp(A.envelope(np.zeros(16000))[0]) == 0.0
    x = A.synth_shaped_noise(1.0, seed=2)
    assert np.mean(A.envelope(x * 10)[0]) - np.mean(A.envelope(x)[0]) == pytest.approx(2.0, abs=1e-6)  # +20 dB = +2 log10
    assert A.envelope(np.full(2000, 1.0))[1].min() == 1.0   # all samples clip


@pytest.mark.parametrize("bpm", [8, 12, 18, 22])
def test_breathing_rate_accuracy(bpm):
    for seed in range(3):
        ups = run_env(A.synth_breath_envelope(50, bpm, seed=seed))
        rep = reported(ups)
        assert len(rep) >= 10 and all(abs(u["bpm"] - bpm) < 0.6 for u in rep)
        assert ups[0]["reason"] == "settling" and all(0 <= u["quality"] <= 1 for u in ups)


@pytest.mark.parametrize("bpm", [10, 12, 15, 16])
@pytest.mark.parametrize("exhale", [0.5, 2.0])
def test_inhale_and_exhale_audible_reports_true_rate_never_double(bpm, exhale):
    ups = run_env(A.synth_breath_envelope(60, bpm, seed=7, double=True, exhale=exhale))
    rep = reported(ups)
    assert rep and all(abs(u["bpm"] - bpm) < 0.6 for u in rep)
    assert not any(u["bpm"] is not None and abs(u["bpm"] - 2 * bpm) < 1.5 for u in ups)


def test_equal_double_bumps_are_withheld_as_ambiguous():
    ups = run_env(A.synth_breath_envelope(50, 15, seed=3, double=True, exhale=1.0))
    late = [u for u in ups if u["t"] >= 30]
    assert late and all(not u["reported"] and u["bpm"] is None and u["reason"] == "ambiguous" for u in late)


def test_regression_sweep_no_double_no_half():
    """True rates 6-40/min x {single bump, equal inhale+exhale, exhale quieter, exhale louder} x noise x seeds:
    nothing may be reported off by more than 10 % (no 2x, no 1/2x). Coverage per condition is printed honestly."""
    conds = {"single": {}, "equal double": dict(double=True, exhale=1.0), "exhale half": dict(double=True, exhale=0.5),
             "exhale louder": dict(double=True, exhale=2.0)}
    table, total = {}, 0
    for name, kw in conds.items():
        win = rep = 0
        for jit in (0.03, 0.1):
            for bpm in (6, 8, 10, 12, 15, 18, 20, 24, 28, 32, 36, 40):
                for seed in range(3):
                    for u in run_env(A.synth_breath_envelope(50, bpm, seed=seed * 7 + bpm, jitter=jit, **kw)):
                        if u["t"] < 22:
                            continue
                        win += 1
                        if u["reported"]:
                            rep += 1
                            assert abs(u["bpm"] - bpm) <= 0.1 * bpm, (name, bpm, jit, seed, u["bpm"])
        table[name] = (win, rep)
        total += win
    print({k: f"{100 * r / w:.0f}% of {w}" for k, (w, r) in table.items()})
    assert total >= 300
    assert table["single"][1] / table["single"][0] > 0.9 and table["equal double"][1] == 0


def test_gates_quiet_clipping_talking_no_mic_settling():
    q = run_env(A.synth_breath_envelope(40, 15, amp=0.03, jitter=0.03, drift=0.02))
    assert {u["reason"] for u in q if u["t"] > 20} == {"too_quiet"}
    env = A.synth_breath_envelope(40, 15, seed=1)
    c = run_env(env, clip=np.full(env.size, 0.05))
    assert {u["reason"] for u in c if u["t"] > 20} == {"clipping"}
    sp = [(t, 0.9) for t in np.arange(1, 40, 0.48)]
    assert {u["reason"] for u in run_env(env, speech=sp) if u["t"] > 20} == {"talking"}
    sp = [(t, 0.9 if i % 20 == 0 else 0.01) for i, t in enumerate(np.arange(1, 40, 0.48))]   # ~5 % speech: not talking
    assert reported(run_env(env, speech=sp))
    e = A.BreathSoundRate()
    outs = [e.push(i / 50.0, None) for i in range(401)]
    assert [u["reason"] for u in outs if u] == ["no_mic"] * 4
    ups = run_env(env)
    assert all(u["reason"] == "settling" for u in ups if u["t"] < P.breathing.settle_s)


def test_pure_noise_never_reported():
    for seed in range(10):
        assert not reported(run_env(1.0 + np.random.default_rng(seed).normal(0, 0.3, 2500)))


def test_audio_to_rate_end_to_end_synthetic_breath_noise():
    for double in (False, True):
        env, clip = A.envelope(A.synth_breath_audio(60, 15, seed=4, double=double))
        rep = reported(run_env(env, clip))
        assert rep and all(abs(u["bpm"] - 15) < 0.7 for u in rep)


# ---- label events

def ev(scores, hop=0.48, **kw):
    p = PARAMS.audio.cough.model_copy(update=kw)
    return A.detect_label_events([[i * hop, s] for i, s in enumerate(scores)], p)


def test_hysteresis_merge_refractory_min_frames():
    assert ev([0, .6, .4, .35, .2, 0])["events"] == [{"t_start": .48, "t_end": 1.44, "peak_score": .6}]
    assert ev([0, .45, .499, .45, 0])["count"] == 0                       # never reaches on_thr
    assert ev([0, .6, .3, .299, .6])["count"] == 1                        # 0.30 holds, 0.299 ends; next onset merges (gap 0.96)
    assert ev([0, .6, 0, 0, 0, 0, .6, 0, 0, 0, 0, 0, .6])["count"] == 3      # starts >= 2.4 s apart
    assert ev([0, .6, 0, 0, 0, .6, 0])["count"] == 1                      # 1.44 s gap < refractory 2.0 s start-to-start
    assert ev([0, .6, 0], refractory_s=0.0)["count"] == 1
    assert ev([0, .6, 0, 0, 0, .6, 0], refractory_s=0.0)["count"] == 2
    assert ev([0, .6, 0, .6, 0, 0, 0, 0, 0, 0, 0, .6, .6, 0], min_frames=2)["count"] == 1


def test_count_is_monotone_as_frames_arrive():
    s = A.synth_scores(60, [(10, 4, .9), (11.5, 3, .8), (30, 3, .7), (50, 5, .95)], seed=2)
    counts = [A.detect_label_events(s[:i])["count"] for i in range(len(s) + 1)]
    assert counts == sorted(counts) and counts[-1] == 3


def test_label_generic_other_label_params():
    p = PARAMS.audio.cough.model_copy(update={"label": "Snoring", "on_thr": 0.3, "off_thr": 0.2})
    assert A.detect_label_events([[0, .1], [.48, .35], [.96, .1]], p)["count"] == 1


def test_fixture_semantics():
    def fx(n):
        return json.loads((FX / f"{n}.json").read_text())
    assert fx("audio_cough_bout_merge")["expected"]["count"] == 4
    assert fx("audio_cough_single")["expected"]["count"] == 1
    for n in ("audio_cough_sustained_noise", "audio_cough_speech_only"):
        assert fx(n)["expected"]["count"] == 0
    reasons = lambda n: [u["reason"] for u in fx(n)["expected"]["updates"]]
    assert "talking" in reasons("audio_breath_talking") and reasons("audio_breath_talking")[-1] == "ok"
    assert "clipping" in reasons("audio_breath_clipping") and reasons("audio_breath_clipping")[-1] == "ok"


# ---- real YAMNet: synthetic signals must not produce cough events

def _yamnet_or_skip():
    try:
        A.ensure_yamnet()
        from mediapipe.tasks.python import audio  # noqa: F401
    except Exception as e:   # pragma: no cover
        pytest.skip(f"mediapipe audio task / yamnet.tflite unavailable here: {e!r}")


def test_real_yamnet_no_cough_events_on_synthetic_signals(tmp_path):
    _yamnet_or_skip()
    fs, n = 16000, 16000 * 8
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    pink = np.fft.irfft(np.fft.rfft(rng.normal(size=n)) / np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / fs), 20)), n=n)
    sigs = {"silence": np.zeros(n), "pink": 0.1 * pink / np.std(pink), "tone440": 0.3 * np.sin(2 * np.pi * 440 * t),
            "chirp": 0.3 * np.sin(2 * np.pi * (200 * t + 0.5 * (3000 / 8) * t * t)),
            "breath_noise": A.synth_breath_audio(8, 15, seed=1, level=0.05)}
    import wave
    for name, x in sigs.items():
        w = tmp_path / f"{name}.wav"
        with wave.open(str(w), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(fs)
            f.writeframes(np.clip(x * 32767, -32768, 32767).astype("<i2").tobytes())
        r = A.classify_wav(w)
        assert len(r["t"]) >= 10 and abs(r["t"][1] - r["t"][0] - 0.48) < 1e-6
        res = A.detect_label_events(list(zip(r["t"], r["Cough"])))
        assert res["count"] == 0, (name, max(r["Cough"]))
