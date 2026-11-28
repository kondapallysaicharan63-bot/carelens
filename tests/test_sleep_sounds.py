"""W10 reference tests on SYNTHETIC signals only: snoring / distress / speech events, possible pauses in breathing sounds,
sound cues for falls. Not a medical device."""
import copy
import json
import random

import pytest

from carelens import audio as A
from carelens import sleep_sounds as S
from carelens import websession as ws
from carelens.fixtures import FIXTURE_DIR as FX
from carelens.params import PARAMS, Params, PARAMS_PATH

EV = PARAMS.audio_events
SL = PARAMS.sleep_sounds


def fx(n):
    return json.loads((FX / f"{n}.json").read_text())["expected"]


def env_run(starts, dur, **kw):
    env = S.synth_envelope_from_starts(starts, dur, **kw)
    return S.run_sleep_env({"fs_env": 50.0, "env": [round(float(v), 3) for v in env]})


# ---------------------------------------------------------------- params

def test_params_validators_reject_bad_values():
    raw = json.loads(PARAMS_PATH.read_text())
    for edit in (lambda r: r["audio_events"]["snoring"].update(off_thr=0.9),
                 lambda r: r["audio_events"].update(cue_labels=["Nope"]),
                 lambda r: r["audio_events"]["glass"].update(label="Snoring"),
                 lambda r: r["audio_events"].update(corroborate_s=0),
                 lambda r: r["sleep_sounds"].update(pause_max_s=5.0),
                 lambda r: r["sleep_sounds"].update(pre_min_bumps=2),
                 lambda r: r["sleep_sounds"]["bump"].update(floor_win_s=5.0),
                 lambda r: r["sleep_sounds"]["bump"].update(merge_s=2.5),
                 lambda r: r["sleep_sounds"].update(bogus=1)):
        bad = copy.deepcopy(raw)
        edit(bad)
        with pytest.raises(Exception):
            Params.model_validate(bad)


def test_yamnet_class_names():
    assert [s.label for s in EV.sections()] == ["Snoring", "Screaming", "Crying, sobbing", "Glass", "Thump, thud", "Speech"]
    assert EV.cue_labels == ["Thump, thud", "Screaming", "Glass"] and EV.corroborate_s == 10.0


# ---------------------------------------------------------------- label events: totals, per hour, tracker

def test_event_fixtures():
    assert fx("audio_snoring_episodes")["count"] == 2 and fx("audio_snoring_none")["count"] == 0
    assert fx("audio_snoring_per_hour")["per_hour"] == [1, 2, 1]
    assert fx("audio_distress_screaming")["count"] == 2 and fx("audio_distress_crying")["count"] == 1
    assert fx("audio_speech_spells")["count"] == 2 and fx("audio_speech_spells")["total_s"] > 10


def test_tracker_equals_batch_on_every_prefix():
    rng = random.Random(3)
    for label in ("Snoring", "Speech", "Screaming"):
        p = A.label_sections()[label]
        frames, t = [], 1.0
        for _ in range(900):
            frames.append((t, rng.choice([0.05, 0.1, 0.25, 0.35, 0.6, 0.9])))
            t += 0.48 if rng.random() < 0.97 else 600.0   # occasional long jumps cross session hours
        tr = A.LabelEventTracker(p)
        for i, (t, s) in enumerate(frames):
            tr.push(t, s)
            if i % 7 == 0 or i == len(frames) - 1:
                v = tr.view()
                b = A.detect_label_events(frames[:i + 1], p)
                sm = A.summarize_events(b["events"])
                assert v["count"] == b["count"] == sm["count"]
                assert v["per_hour"] == sm["per_hour"] and abs(v["total_s"] - sm["total_s"]) < 1e-6
                assert len(v["per_hour_s"]) == len(sm["per_hour_s"]) and all(abs(a - b) < 1e-6 for a, b in zip(v["per_hour_s"], sm["per_hour_s"]))
                assert (v["last"] is None) == (not b["events"])
                if b["events"]:
                    assert v["last"]["t_start"] == b["events"][-1]["t_start"] and v["last"]["t_end"] == b["events"][-1]["t_end"]


def test_tracker_bounded_memory_8h_stream():
    tr = A.LabelEventTracker(A.label_sections()["Snoring"])
    rng = random.Random(1)
    n = int(8 * 3600 / 0.48)
    for i in range(n):
        tr.push(1.0 + i * 0.48, 0.7 if (i // 40) % 3 == 0 and rng.random() < 0.9 else 0.05)
    v = tr.view()
    assert v["count"] > 500 and len(tr._starts) <= 500 and len(v["per_hour"]) == 8 and sum(v["per_hour"]) == v["count"]
    assert v["total_s"] > 3600


# ---------------------------------------------------------------- pauses (bump level fixtures)

def test_pause_fixtures():
    one = fx("sleep_pause_single")
    assert one["count"] == 1 and one["rate_per_h"] is None and abs(one["pauses"][0]["duration_s"] - 15.0) < 0.01
    s = fx("sleep_silence_90s")
    assert s["count"] == 0 and s["no_sound_gaps"] == 1
    for n in ("sleep_gap_during_speech", "sleep_gap_with_spike", "sleep_irregular_before", "sleep_short_regular_before"):
        assert fx(n)["count"] == 0 and fx(n)["no_sound_gaps"] == 0, n
    m = fx("sleep_multiple_pauses_short")
    assert m["count"] == 3 and m["rate_per_h"] is None and m["regular_s"] < 1800
    long = fx("sleep_multiple_pauses_30min")
    assert long["count"] == 4 and long["regular_s"] >= 1800
    assert long["rate_per_h"] == pytest.approx(4 / (long["regular_s"] / 3600), rel=1e-5)
    b = fx("sleep_gap_boundaries")
    assert [round(p["duration_s"], 1) for p in b["pauses"]] == [10.5, 59.5] and b["no_sound_gaps"] == 1


def test_pause_wording_never_claims_a_condition():
    src = open(S.__file__).read().lower()
    assert "apn" not in src and "emergency" not in src and "distress detected" not in src
    # view keys are neutral counters
    assert set(S.PauseTracker().view()) == {"count", "pauses", "regular_s", "rate_per_h", "no_sound_gaps"}


# ---------------------------------------------------------------- pauses (envelope level, the full chain)

def test_envelope_chain_finds_the_pause_and_rejects_the_rest():
    assert fx("sleep_env_pause")["count"] == 1
    assert fx("sleep_env_silence_90s")["count"] == 0 and fx("sleep_env_silence_90s")["no_sound_gaps"] == 1
    assert fx("sleep_env_startup_silence")["count"] == 1 and fx("sleep_env_inhale_exhale")["count"] == 1   # two bumps per breath (inhale + louder exhale) are one breath
    for n in ("sleep_env_gap_speech", "sleep_env_gap_burst", "sleep_env_irregular"):
        assert fx(n)["count"] == 0, n


def test_synthetic_accuracy_over_seeds():
    """Seeded synthetic sweep: 14/min breathing with a 3-breath gap must give exactly one pause; no gap, none."""
    hits = clean = 0
    for seed in range(20):
        st = S.synth_breath_starts(170, 14.0, [(80.0, 93.0)], seed=seed)
        hits += env_run(st, 170, seed=seed)["count"] == 1
        clean += env_run(S.synth_breath_starts(170, 14.0, [], seed=seed), 170, seed=seed)["count"] == 0
    assert hits >= 19 and clean == 20, (hits, clean)


def test_breaths_at_other_rates_and_no_signal():
    for bpm in (8.0, 12.0, 20.0):
        st = S.synth_breath_starts(200, bpm, [(100.0, 100.0 + 3.5 * 60 / bpm)], seed=2)
        assert env_run(st, 200, bpm=bpm)["count"] == 1, bpm
    tr = S.PauseTracker()
    for i in range(3000):
        tr.push_frame(i / 50, None)
    assert tr.view()["count"] == 0
    tr.push_frame(0.0, float("nan"))


def test_microphone_loss_breaks_regularity_but_keeps_totals():
    tr = S.PauseTracker()
    env = S.synth_envelope_from_starts(S.synth_breath_starts(160, 14.0, [(80.0, 95.0)], seed=3), 160, seed=11)
    for i, v in enumerate(env):
        tr.push_frame(i / 50, None if 3900 <= i < 3950 else float(v))   # a 1 s dropout just before the gap
    assert tr.view()["count"] in (0, 1) and tr.view()["no_sound_gaps"] == 0


def test_bounded_memory_8h_stream():
    tr = S.PauseTracker()
    per = 60 / 14.0
    t, n_b = 0.0, 0
    while t < 8 * 3600:
        tr.push_bump(t)
        n_b += 1
        if n_b % 400 == 0:
            t += 15.0
        t += per
        if n_b % 50 == 0:
            tr.push_speech(t, 0.02)
        if n_b % 997 == 0:
            tr.push_spike(t + 1, t + 1.5)
    v = tr.view()
    assert 12 <= v["count"] <= 17 and len(tr._recent) <= S.MAX_RECENT and len(tr._speech) < 100 and len(tr._spikes) < 20
    assert len(v["pauses"]) <= S.MAX_KEEP and v["rate_per_h"] is not None and 1.4 < v["rate_per_h"] < 2.3
    # the envelope path as well: 20 minutes of frames keep every buffer small
    st = S.BumpStream()
    for i in range(50 * 1200):
        st.push(i / 50, 1.0 + 0.5 * ((i % 214) < 25))
    assert len(st._mono) <= st.floor_n and len(st._rises) <= 16


# ---------------------------------------------------------------- sound cue for falls

def test_sound_cue_fixture_and_cases():
    assert fx("sound_cue_corroboration")["cues"] == [True, False, True, False, True, False, False]
    fall = {"t_onset": 100.0, "t_confirmed": 100.0}
    thud = lambda t: [{"label": "Thump, thud", "t_start": t, "t_end": t + 1}]
    assert S.sound_cue(fall, thud(104.0)) and not S.sound_cue(fall, thud(120.0))
    assert S.sound_cue(fall, [{"label": "Screaming", "t_start": 90.0, "t_end": 91.0}])
    assert not S.sound_cue(fall, [{"label": "Cough", "t_start": 101.0, "t_end": 102.0}]) and not S.sound_cue(fall, [])


# ---------------------------------------------------------------- export schema

def _session():
    d = json.loads((FX.parent / "session.example.json").read_text())
    d["events"] += [{"t": 5.0, "wall": d["started_at"], "kind": "distress_sound", "t_onset": 4.0, "t_end": 6.0, "peak_score": 0.8, "trigger": "Screaming", "merged_into_fall": True},
                    {"t": 9.0, "wall": d["started_at"], "kind": "possible_pause", "t_onset": 1.0, "t_end": 16.0},
                    {"t": 11.0, "wall": d["started_at"], "kind": "possible_fall", "t_onset": 0.5, "sound_cue": True}]
    d["summaries"] = d.get("summaries", []) + [
        {"t": 9.0, "wall": d["started_at"], "kind": "snoring", "reported": True, "reason": "ok", "count": 3, "minutes": 1.5, "per_hour": [3]},
        {"t": 9.0, "wall": d["started_at"], "kind": "pauses", "reported": True, "reason": "ok", "count": 1, "regular_min": 12.0, "durations_s": [15.0]},
        {"t": 9.0, "wall": d["started_at"], "kind": "talking", "reported": True, "reason": "ok", "minutes": 2.0, "per_hour": [2.0]}]
    return d


def test_export_validators_accept_and_reject():
    assert ws.validate(_session()) == []
    assert ws.validate(json.loads((FX.parent / "session.example.json").read_text())) == []   # backward compatible
    for edit in (lambda d: d["events"][-1].update(sound_cue="yes"),
                 lambda d: d["events"][-3].update(merged_into_fall=1),
                 lambda d: d["summaries"][-1].pop("minutes"),
                 lambda d: d["summaries"][-2].update(count=1.5),
                 lambda d: d["summaries"][-3].update(per_hour=[1] * 73),
                 lambda d: d["summaries"][-2].update(durations_s=[-1.0]),
                 lambda d: d["events"][-3].update(kind="apnea")):
        d = _session()
        edit(d)
        assert ws.validate(d)
