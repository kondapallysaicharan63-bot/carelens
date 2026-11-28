"""Deterministic synthetic fixtures + Python reference outputs, shared with the TypeScript port.

``python -m carelens.fixtures``          regenerate ``shared/fixtures/*.json``
``python -m carelens.fixtures --check``  regenerate in memory; exit 1 on any difference (CI)

Synthetic data only (never footage). Every fixture is
``{"name","description","input_level","params_version","input","expected","tolerance"}``.
Inputs are rounded to 6 decimals *before* the reference runs, so the committed input
reproduces the committed output exactly. Not a medical device.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from . import pose as P
from dataclasses import replace

from .events import FallDetector, InactivityDetector
from .features import FeatureExtractor
from .tracking import PresenceTracker
from .params import PARAMS
from .hrv import run_hrv, synth_rr, truth_hrv, with_ectopics, without_beats
from .ppg import run_contact_ppg, synth_trace
from .rppg import run_face_rppg, synth_face_trace
from .synth import ASPECT, Scenario, Segment
from . import synth as SY
from .mobility import MOBILITY_INDICES, run_mobility
from .vitals import BreathingRateEstimator, HeartRateEstimator
from . import audio as AU

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "shared" / "fixtures"
DP = 6
PENDING_FILE = "pending.json"   # hand-maintained list of fixtures with no TS port yet
# Landmarks the features/events code actually reads (core + face + legs), in MediaPipe order.
LM_INDICES = [P.NOSE, P.L_EYE, P.R_EYE, P.L_EAR, P.R_EAR, P.L_SHOULDER, P.R_SHOULDER,
              P.L_HIP, P.R_HIP, P.L_KNEE, P.R_KNEE, P.L_ANKLE, P.R_ANKLE]
LM_FILL_VISIBILITY = 0.2   # synth.py gives unmodelled landmarks this (sub-threshold) visibility


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, np.ndarray):
        return _round(x.tolist())
    if isinstance(x, (float, np.floating)):
        return round(float(x), DP)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ------------------------------------------------------------------ runners (reference)

def run_hr_trace(inp: dict) -> dict:
    """Green-channel HR at ROI-trace level, via HeartRateEstimator.update_roi (the pipeline path)."""
    est = HeartRateEstimator(method="green")
    fps, n = inp["fps"], len(inp["green"])
    bright, motion = inp["brightness"], inp["motion"]
    for i in range(n):
        t = i / fps
        b = bright[i] if isinstance(bright, list) else bright
        m = motion[i] if isinstance(motion, list) else motion
        est.update_roi(t, inp["green"][i], b, m)
    return {"estimates": _round(est.estimates)}


def run_breathing_trace(inp: dict) -> dict:
    est = BreathingRateEstimator()
    fps = inp["fps"]
    for i, y in enumerate(inp["chest_y_px"]):
        t = i / fps
        est.update_displacement(t, y)
    return {"estimates": _round(est.estimates)}


def _landmarks(frame: list, indices: list) -> np.ndarray:
    lm = np.zeros((P.N_LANDMARKS, 3), np.float32)
    lm[:, 2] = LM_FILL_VISIBILITY
    for k, row in zip(indices, frame):
        lm[k] = row
    return lm


def run_landmarks(inp: dict) -> dict:
    """FeatureExtractor -> FallDetector + InactivityDetector over raw (untracked) landmarks."""
    fps, cfg = inp["fps"], inp.get("config", {})
    indices = inp["landmark_indices"]
    feats = FeatureExtractor(aspect=inp["aspect"])
    fall = FallDetector()
    inact = InactivityDetector(seconds=cfg.get("inactivity_seconds", PARAMS.events.inactivity.seconds))
    events, features, last_sec = [], [], None
    for i, frame in enumerate(inp["landmarks"]):
        t = i / fps
        s = feats.update(t, _landmarks(frame, indices) if frame is not None else None)
        if last_sec != math.floor(t):   # one sample per whole second
            last_sec = math.floor(t)
            features.append({"t": round(t, 3), "present": s.present, "posture": s.posture,
                             "torso_angle_deg": s.torso_angle, "motion": s.motion})
        for det in (fall, inact):
            ev = det.update(s)
            if ev:
                events.append({"kind": ev.kind, "t_onset": ev.t_onset, "t_confirmed": ev.t_confirmed,
                               "trigger": ev.details.get("trigger")})
    return _round({"events": events, "features_1hz": features})


def run_pose_chain(inp: dict) -> dict:
    """The whole browser pose chain: landmarks -> PresenceTracker -> FeatureExtractor -> FallDetector +
    InactivityDetector, wired exactly as in ``pipeline.run`` (camera gaps, held frames). Frame times are
    ``t[]`` (default ``i/fps``); ``gaps = [[frame_index, gap_s], ...]`` marks outages before a frame."""
    fps, cfg, aspect = inp["fps"], inp.get("config", {}), inp["aspect"]
    indices = inp["landmark_indices"]
    gaps = {int(i): g for i, g in inp.get("gaps", [])}
    feats = FeatureExtractor(aspect=aspect)
    tracker = PresenceTracker()
    fall = FallDetector()
    inact = InactivityDetector(seconds=cfg.get("inactivity_seconds", PARAMS.events.inactivity.seconds))
    events, features, postures, presence = [], [], [], []
    last_sec, s = None, None
    for i, frame in enumerate(inp["landmarks"]):
        t = inp["t"][i] if "t" in inp else i / fps
        gap = gaps.get(i, 0.0)
        if gap > 0:
            tracker.skip_gap(gap)
            feats.reset()
            fall.pause(gap)
            inact.pause(gap)
        lm = _landmarks(frame, indices) if frame is not None else None
        lm, held = tracker.update(t, lm, aspect)
        s = replace(s, t=t) if (held and s is not None) else feats.update(t, lm)
        state = "held" if held else ("present" if s.present else "absent")
        if not presence or presence[-1]["state"] != state:
            presence.append({"t": round(t, 3), "state": state})
        if not postures or postures[-1]["posture"] != s.posture:
            postures.append({"t": round(t, 3), "posture": s.posture})
        if last_sec != math.floor(t):
            last_sec = math.floor(t)
            features.append({"t": round(t, 3), "present": s.present, "posture": s.posture,
                             "torso_angle_deg": s.torso_angle, "motion": s.motion})
        for det in (fall, inact):
            ev = det.update(s, held=True) if (held and det is fall) else det.update(s)
            if ev:
                events.append({"kind": ev.kind, "t_onset": ev.t_onset, "t_confirmed": ev.t_confirmed,
                               "trigger": ev.details.get("trigger")})
    return _round({"events": events, "features_1hz": features, "posture_runs": postures,
                   "presence_runs": presence})


RUNNERS: dict[str, Callable[[dict], dict]] = {
    "roi_trace_hr": run_hr_trace,
    "chest_trace_breathing": run_breathing_trace,
    "landmarks": run_landmarks,
    "pose_chain": run_pose_chain,
    "ppg_contact": run_contact_ppg,
    "ppg_hrv": run_hrv,
    "face_rppg": run_face_rppg,
    "audio_envelope": AU.run_audio_envelope,
    "audio_breath_env": AU.run_breath_env,
    "audio_label_events": AU.run_label_events,
    "mobility": run_mobility,
}

TOL_VITALS = {"value_bpm": 0.5, "quality": 0.05, "t_s": 0.05, "reported": "exact",
              "estimate_count": "exact"}
TOL_PPG = {"bpm": 0.3, "quality": 0.03, "t_s": 0.05, "ibi_ms": 5.0, "snr": 0.05, "pi": 0.01,
           "reported": "exact", "reason": "exact", "update_count": "exact"}
TOL_HRV = {"hrv_ms": 0.5, "bpm": 0.3, "reported": "exact", "reason": "exact"}
TOL_RPPG = {"bpm": 0.3, "quality": 0.03, "t_s": 0.05, "snr": 0.05,
            "reported": "exact", "reason": "exact", "update_count": "exact"}
TOL_AUDIO_ENV = {"env_log10": 1e-4, "reported": "exact"}
TOL_BREATH_SOUND = {"bpm": 0.3, "quality": 0.03, "t_s": 0.05, "snr": 0.05,
                    "reported": "exact", "reason": "exact", "update_count": "exact"}
TOL_AUDIO_EVENTS = {"event_count": "exact", "t_start": "exact", "t_end": "exact", "peak_score": "exact"}
TOL_MOBILITY = {"mob_time_s": 0.1, "speed_m_s": 0.02, "distance_m": 0.1, "cadence_spm": 1.0, "quality": 0.05,
                "reported": "exact", "reason": "exact", "reps": "exact", "partial_rises": "exact", "stands": "exact",
                "n_segments": "exact", "n_not_side_on": "exact", "rep_in_progress": "exact"}
TOL_EVENTS = {"event_time_s": 0.1, "kind": "exact", "event_count": "exact",
              "torso_angle_deg": 0.05, "motion": 0.01, "posture": "exact"}


def _fixture(name, description, level, inp, tolerance) -> dict:
    inp = _round(inp)
    return {"name": name, "description": description, "input_level": level,
            "params_version": PARAMS.version, "input": inp,
            "expected": _round(RUNNERS[level](inp)), "tolerance": tolerance}


# ------------------------------------------------------------------ generators

def _hr(name, bpm, seed, desc, dur=60.0, fps=30.0, motion_burst=None, light_step=None):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fps)) / fps
    green = (110 + 0.35 * np.sin(2 * np.pi * bpm / 60 * t + rng.uniform(0, 6.28))
             + 0.2 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 0.12, t.size))
    motion = np.full(t.size, 0.02)
    bright = np.full(t.size, 120.0)
    if motion_burst:
        motion[(t >= motion_burst[0]) & (t < motion_burst[1])] = 0.4
    if light_step:
        bright[t >= light_step] = 25.0
    inp = {"fps": fps, "green": green,
           "brightness": bright.tolist() if light_step else 120.0,
           "motion": motion.tolist() if motion_burst else 0.02,
           "truth_bpm": bpm}
    return _fixture(name, desc, "roi_trace_hr", inp, TOL_VITALS)


def _br(name, bpm, seed, dur=75.0, fps=15.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fps)) / fps
    y = (1.5 * np.sin(2 * np.pi * bpm / 60 * t + rng.uniform(0, 6.28))
         + 0.15 * np.sin(2 * np.pi * 0.02 * t) + rng.normal(0, 0.1, t.size))
    return _fixture(
        name, f"Chest vertical displacement (px) at {bpm} breaths/min with noise.",
        "chest_trace_breathing", {"fps": fps, "chest_y_px": y, "truth_breaths_per_min": bpm},
        TOL_VITALS)


def _lm(name, desc, segs, seed, config=None, fps=15.0):
    scn = Scenario(segs, jitter=0.0008, seed=seed)
    frames = []
    for i in range(int(scn.duration * fps)):
        lm = scn.pose_landmarks(i / fps)
        frames.append(None if lm is None else lm[LM_INDICES].tolist())
    inp = {"fps": fps, "aspect": ASPECT, "landmark_indices": LM_INDICES, "landmarks": frames}
    if config:
        inp["config"] = config
    return _fixture(name, desc, "landmarks", inp, TOL_EVENTS)


def _chain(name, desc, segs, seed, fps=15.0, dropouts=(), gaps=(), config=None):
    """pose_chain fixture: scenario landmarks with ``dropouts`` [(t0, t1)) set to no-person and camera ``gaps``
    [(t_observed, gap_s)] (timestamps after a gap jump by gap_s; the pipeline is told via ``gaps``)."""
    scn = Scenario(segs, jitter=0.0008, seed=seed)
    frames, times, gap_list, jump = [], [], [], 0.0
    pending = sorted(gaps)
    for i in range(int(scn.duration * fps)):
        t_obs = i / fps
        while pending and t_obs >= pending[0][0] - 1e-9:
            g = pending.pop(0)[1]
            jump += g
            gap_list.append([i, g])
        lm = scn.pose_landmarks(t_obs)
        if any(a <= t_obs < b for a, b in dropouts):
            lm = None
        frames.append(None if lm is None else lm[LM_INDICES].tolist())
        times.append(t_obs + jump)
    inp = {"fps": fps, "aspect": ASPECT, "landmark_indices": LM_INDICES, "landmarks": frames, "t": times}
    if gap_list:
        inp["gaps"] = gap_list
    if config:
        inp["config"] = config
    return _fixture(name, desc, "pose_chain", inp, TOL_EVENTS)


def _ppg(name, desc, dur=40.0, bpm=72.0, seed=0, **kw):
    tr = synth_trace(dur, bpm, seed=seed, **kw)
    inp = {k: [round(v, 4) for v in tr[k]] for k in ("t", "r", "g", "b", "clip_frac")}
    inp["truth_bpm"] = bpm
    return _fixture(name, desc, "ppg_contact", inp, TOL_PPG)


def _hrv(name, desc, dur, rmssd, seed, ectopic=(), missed=(), noise=0.05):
    """Fingertip trace whose beats follow a known RR series (true RMSSD ``rmssd`` ms); ``ectopic`` / ``missed`` are beat indices."""
    bt = synth_rr(dur + 10, rmssd, seed=seed)
    t_rmssd, t_sdnn = truth_hrv(np.diff(bt) * 1000.0)
    if ectopic:
        bt = with_ectopics(bt, ectopic)
    if missed:
        bt = without_beats(bt, missed)
    tr = synth_trace(dur, beat_times=bt, seed=seed, noise=noise)
    inp = {k: [round(v, 4) for v in tr[k]] for k in ("t", "r", "g", "b", "clip_frac")}
    inp["truth_rmssd_ms"], inp["truth_sdnn_ms"] = round(t_rmssd, 3), round(t_sdnn, 3)
    return _fixture(name, desc, "ppg_hrv", inp, TOL_HRV)


def _rppg(name, desc, dur=32.0, bpm=72.0, seed=0, skin="medium", **kw):
    kw.setdefault("noise", 0.03)
    tr = synth_face_trace(dur, bpm, skin=skin, seed=seed, **kw)
    inp = {"t": [round(v, 4) for v in tr["t"]],
           "roi_rgb": [None if r is None else [round(c, 3) for c in r] for r in tr["roi_rgb"]],
           "roi_px": tr["roi_px"], "face_motion": [round(v, 3) for v in tr["face_motion"]],
           "brightness": [round(v, 2) for v in tr["brightness"]], "truth_bpm": bpm, "skin": skin}
    if min(tr["skin_frac"]) < 1.0:
        inp["skin_frac"] = [round(v, 3) for v in tr["skin_frac"]]
    return _fixture(name, desc, "face_rppg", inp, TOL_RPPG)


def _audio_env(name, desc, x):
    return _fixture(name, desc, "audio_envelope", {"fs": PARAMS.audio.fs_hz, "pcm_b64": AU.encode_pcm(x)}, TOL_AUDIO_ENV)


def _breath(name, desc, dur=50.0, bpm=15.0, seed=0, speech=None, clip=None, speech_mod=None, **kw):
    """Envelope-level breath-sound fixture. ``speech`` = (t0, t1): YAMNet Speech scores high in that span, and the
    envelope gets a syllabic (4 Hz) modulation; ``clip`` = (t0, t1, frac): per-frame clip fraction in that span."""
    fs = PARAMS.audio.fs_hz / PARAMS.audio.envelope.hop
    env = AU.synth_breath_envelope(dur, bpm, seed=seed, **kw)
    t = np.arange(env.size) / fs
    inp: dict = {"fs_env": fs, "truth_bpm": bpm}
    if speech is not None:
        m = (t >= speech[0]) & (t < speech[1])
        env = env + m * 0.5 * np.sin(2 * np.pi * 4.0 * t)
        rng = np.random.default_rng(seed + 1000)
        hop = PARAMS.audio.yamnet.hop_samples / PARAMS.audio.fs_hz
        frames = np.arange(AU.YAMNET_WINDOW_S, dur, hop)
        inp["speech"] = [[float(tt), float(np.clip((0.8 if speech[0] <= tt < speech[1] else 0.02) + abs(rng.normal(0, 0.05)), 0, 1))]
                         for tt in frames]
    if clip is not None:
        inp["clip"] = [clip[2] if clip[0] <= tt < clip[1] else 0.0 for tt in t]
    inp["env"] = [round(float(v), 3) for v in env]
    return _fixture(name, desc, "audio_breath_env", inp, TOL_BREATH_SOUND)


def _scores(name, desc, n, hits, seed=0, floor=0.03, noise=0.02, hop=0.48, label="Cough"):
    """Cough-score fixture: ``hits`` = {frame_index: score} over a noisy floor."""
    rng = np.random.default_rng(seed)
    s = floor + np.abs(rng.normal(0.0, noise, n))
    for i, v in hits.items():
        s[i] = v
    frames = [[round(AU.YAMNET_WINDOW_S + i * hop, 6), round(float(v), 6)] for i, v in enumerate(s)]
    return _fixture(name, desc, "audio_label_events", {"label": label, "frames": frames}, TOL_AUDIO_EVENTS)


def _audio_fixtures() -> list:
    x = AU.synth_shaped_noise(3.0, seed=401)
    x[24000:24300] = np.clip(x[24000:24300] * 40.0, -1.0, 1.0)   # an overdriven burst: clip_frac > 0 in a few frames
    return [
        _audio_env("audio_envelope_shaped_noise",
                   "3 s of synthetic coloured noise at 16 kHz (slow modulation, a click, a clipped burst): the 50 Hz log "
                   "band-power envelope and per-frame clip fraction.", x),
        _breath("audio_breath_8bpm", "Synthetic breath-sound envelope (50 Hz), 8 breaths/min, one bump per breath.", 50, 8, 301),
        _breath("audio_breath_12bpm", "12 breaths/min.", 50, 12, 302),
        _breath("audio_breath_18bpm", "18 breaths/min.", 50, 18, 303),
        _breath("audio_breath_30bpm",
                "30 breaths/min, one broad bump per breath: the single-bump evidence (bump spans most of the interval, no "
                "alternation, no line at 15/min) is clear, so 30 is reported.", 50, 30, 304),
        _breath("audio_breath_double_peaks_15bpm",
                "15 breaths/min with inhale AND exhale audible (two bumps per breath, exhale half as loud): the bumps "
                "alternate, so the true 15/min is reported, not the 30/min bump rate.", 50, 15, 305, double=True, exhale=0.5),
        _breath("audio_breath_double_equal_15bpm",
                "Inhale and exhale equally loud and short (two equal bumps per breath, 15/min): fundamentally ambiguous with "
                "30/min single bumps of a narrower shape, so withheld (reason ambiguous), never a 2x or 1/2x number.",
                50, 15, 309, double=True, exhale=1.0),
        _breath("audio_breath_talking", "15 breaths/min; the person talks 22-32 s (YAMNet Speech high, syllabic envelope): "
                "withheld as talking until the speech leaves the 30 s window, then ok again.", 70, 15, 306, speech=(22.0, 32.0)),
        _breath("audio_breath_too_quiet", "Almost no envelope variation (muted or distant microphone): too_quiet.",
                50, 15, 307, amp=0.03, jitter=0.03, drift=0.02),
        _breath("audio_breath_clipping", "15 breaths/min; 8 % of samples clip 15-25 s (gain too high): clipping until it leaves the 30 s window, then ok.",
                65, 15, 308, clip=(15.0, 25.0, 0.08)),
        _scores("audio_cough_single", "One cough (a short burst of high Cough scores): one event.", 62,
                {19: 0.38, 20: 0.85, 21: 0.68, 22: 0.43, 23: 0.21}, 501),
        _scores("audio_cough_bout_merge",
                "Bout 1: two bursts 0.96 s apart (merged); a single cough; one more cough; two single frames 1.44 s apart "
                "(inside the refractory time: absorbed). Four events.", 62,
                {20: 0.8, 21: 0.6, 22: 0.1, 23: 0.7, 32: 0.9, 33: 0.4, 38: 0.6, 48: 0.7, 51: 0.7}, 502),
        _scores("audio_cough_borderline_hysteresis",
                "Scores that hover: 0.499/0.45 plateaus never start an event; 0.55 then 0.42, 0.33 stays on until 0.29; a "
                "later 0.52 stays on through exactly 0.30 and ends at 0.299. Two events.", 62,
                {5: 0.45, 6: 0.45, 7: 0.499, 8: 0.45, 15: 0.55, 16: 0.42, 17: 0.33, 18: 0.29, 19: 0.4,
                 30: 0.52, 31: 0.31, 32: 0.30, 33: 0.299, 34: 0.45,
                 45: 0.4, 46: 0.35}, 503),
        _scores("audio_cough_sustained_noise",
                "A minute of loud non-cough noise: Cough scores wander 0.1-0.45, never reaching the start threshold. No events.", 125,
                {i: 0.1 + 0.35 * abs(np.sin(i * 0.37)) * abs(np.cos(i * 0.11)) for i in range(125)}, 504),
        _scores("audio_cough_speech_only", "Someone talking: Cough scores stay low with a few 0.25-0.35 blips. No events.", 62,
                {10: 0.25, 11: 0.35, 25: 0.3, 40: 0.28, 41: 0.33}, 505),
    ]


def _mob(name, desc, test, script, go_t=None, fps=15.0, dropouts=(), height_m=None):
    """mobility fixture: a side-on synthetic script rendered to landmarks (3 decimals) -> tracker -> the named test."""
    ts, frames = script.frames(fps, dropouts)
    rows = [None if lm is None else np.round(lm[MOBILITY_INDICES], 3).tolist() for lm in frames]
    inp = {"test": test, "fps": fps, "aspect": ASPECT, "landmark_indices": MOBILITY_INDICES, "landmarks": rows}
    if go_t is not None:
        inp["go_t"] = go_t
    if height_m is not None:
        inp["height_m"] = height_m
    return _fixture(name, desc, "mobility", inp, TOL_MOBILITY)


def _mobility_fixtures() -> list:
    out = []
    for n, cycle, jit, seed in ((8, 2.9, 0.0008, 701), (12, 2.1, 0.002, 702), (15, 1.7, 0.002, 703)):
        scr, _ = SY.chair_stand_script(n, partial_after=(2, 5), cycle=cycle, t0=1.0, duration=32.0, jitter=jit, seed=seed)
        out.append(_mob(f"mobility_chair_stand_{n}",
                        f"30 s chair stand, side-on synthetic figure: {n} full reps plus 2 partial rises (to a 142 deg knee) that "
                        "must not count; Go at 1 s.", "chair_stand_30s", scr, go_t=1.0))
    scr, _ = SY.chair_stand_script(12, partial_after=(3, 7), cycle=2.1, t0=1.0, duration=32.0, jitter=0.004, seed=704)
    out.append(_mob("mobility_chair_stand_dropouts",
                    "12 reps + 2 partial rises with 4 px-class landmark noise and three 0.3 s pose dropouts bridged by the tracker: "
                    "still 12 reps, quality below 1.", "chair_stand_30s", scr, go_t=1.0,
                    dropouts=[(5.0, 5.3), (13.1, 13.4), (20.5, 20.8)]))
    scr, _ = SY.chair_stand_script(5, cycle=2.4, t0=1.5, duration=16.0, jitter=0.002, seed=705)
    out.append(_mob("mobility_sts_5x", "Five sit-to-stands, rising 0.5 s before Go (Go at 2 s): the clock starts at the rise onset.",
                    "sit_to_stand_5x", scr, go_t=2.0))
    scr, ev, _ = SY.tug_script(jitter=0.002, seed=706)
    out.append(_mob("mobility_tug", "Timed Up and Go, 3 m each way, side-on: rise, walk, turn, walk back, turn, sit; reported with phases.",
                    "tug", scr, go_t=ev["go"]))
    scr, ev, _ = SY.tug_script(return_trip=False, jitter=0.002, seed=707)
    out.append(_mob("mobility_tug_no_turn", "Walks 3 m away and sits down at the far end: no turn, withheld (no_turn).", "tug", scr, go_t=ev["go"]))
    for tag, v, secs in (("0p6", 0.6, 6.0), ("1p0", 1.0, 4.0), ("1p4", 1.4, 3.2)):
        out.append(_mob(f"mobility_walk_{tag}", f"Side-on walk at {v} m/s (1.65 m person), 108 steps/min: speed within 10 %.",
                        "walking_speed", SY.walk_script(v, secs, lead=1.0, jitter=0.002, seed=710 + int(v * 10)), height_m=1.65))
    out.append(_mob("mobility_walk_crossing_3s", "Out of view for 2 s, a single 3 s side-on crossing at 1.0 m/s, out of view again: quality is measured over "
                    "the in-view span only, so the walk is reported.", "walking_speed",
                    SY.walk_script(1.0, 3.0, lead=2.0, tail=2.0, x0=0.1, jitter=0.002, seed=730), height_m=1.65, dropouts=[(0.0, 2.0), (5.0, 7.0)]))
    out.append(_mob("mobility_walk_toward_camera", "Walks diagonally towards the camera (apparent height grows 2x): not_side_on, no speed.",
                    "walking_speed", SY.approach_script(drift_u=0.6, jitter=0.002, seed=720), height_m=1.65))
    return out


def generate_all() -> dict[str, str]:
    fx = [
        _hr("hr_green_60bpm", 60, 11, "Green-channel ROI mean, 60 bpm, noise."),
        _hr("hr_green_72bpm", 72, 12, "Green-channel ROI mean, 72 bpm, noise."),
        _hr("hr_green_90bpm", 90, 13, "Green-channel ROI mean, 90 bpm, noise."),
        _hr("hr_green_110bpm", 110, 14, "Green-channel ROI mean, 110 bpm, noise."),
        _hr("hr_green_motion_burst", 72, 15,
            "72 bpm with a motion burst 20-28 s: window resets, estimates resume after refill.",
            motion_burst=(20.0, 28.0)),
        _hr("hr_green_low_light", 90, 16,
            "90 bpm; brightness drops below the gate at 30 s: no estimates afterwards.",
            light_step=30.0),
        _br("breathing_10bpm", 10, 21),
        _br("breathing_15bpm", 15, 22),
        _br("breathing_20bpm", 20, 23),
        _lm("fall_sudden", "Walk, stand, sudden drop to lying (0.6 s), stay down: one fall event.",
            [Segment("walk", 3, x0=0.15, x1=0.4), Segment("still", 2, posture="stand", x0=0.4),
             Segment("transition", 0.6, posture="stand", to="lie", x0=0.4),
             Segment("still", 25, posture="lie", x0=0.4)], 31),
        _lm("fall_slow_lie_down", "Slow 6 s lie-down then stay lying: no fall event.",
            [Segment("still", 2, posture="stand", x0=0.4),
             Segment("transition", 6, posture="stand", to="lie", x0=0.4),
             Segment("still", 25, posture="lie", x0=0.4)], 32),
        _lm("inactivity_sit_still", "Walk then sit still 40 s with inactivity_seconds=30: one inactivity event.",
            [Segment("walk", 2, x0=0.2, x1=0.5), Segment("still", 40, posture="sit", x0=0.5)], 33,
            config={"inactivity_seconds": 30.0}),
        _chain("track_dropout_fill",
               "Standing 8 s; a 0.4 s dropout is bridged (held), a 1.0 s dropout drops presence (hold_s = 0.5 s) and "
               "re-entry needs the enter span.",
               [Segment("still", 8, posture="stand", x0=0.5)], 41, dropouts=[(3.0, 3.4), (5.0, 6.0)]),
        _chain("track_enter_hysteresis",
               "A lone detection after 1 s of nothing never creates presence; sustained detection from 2 s does, after "
               "the enter span.",
               [Segment("still", 6, posture="stand", x0=0.5)], 42, dropouts=[(0.0, 1.0), (1.0 + 1e-3, 2.0)]),
        _chain("track_fast_fall_5fps",
               "5 fps (the validated floor is 3): walk, stand, 0.6 s drop to lying, stay down. The big frame-to-frame jump "
               "passes through the tracker raw: one fall event.",
               [Segment("walk", 3, x0=0.15, x1=0.4), Segment("still", 2, posture="stand", x0=0.4),
                Segment("transition", 0.6, posture="stand", to="lie", x0=0.4),
                Segment("still", 25, posture="lie", x0=0.4)], 43, fps=5.0),
        _chain("posture_sequence",
               "Stand 4 s, sit down (1.5 s), sit 4 s, slow lie-down (6 s), lie 4 s: posture labels standing -> sitting -> "
               "lying, no fall.",
               [Segment("still", 4, posture="stand", x0=0.5),
                Segment("transition", 1.5, posture="stand", to="sit", x0=0.5),
                Segment("still", 4, posture="sit", x0=0.5),
                Segment("transition", 6, posture="sit", to="lie", x0=0.5),
                Segment("still", 4, posture="lie", x0=0.5)], 44),
        _chain("fall_camera_gap",
               "Sudden fall at 2 s, lie still; the camera is unobserved for 40 s from 6 s. The pending fall survives the gap "
               "and the gap does not count as stillness: confirmed ~20 s of OBSERVED stillness after onset.",
               [Segment("still", 2, posture="stand", x0=0.4),
                Segment("transition", 0.6, posture="stand", to="lie", x0=0.4),
                Segment("still", 30, posture="lie", x0=0.4)], 45, gaps=[(6.0, 40.0)]),
        _ppg("ppg_clean_60bpm", "Fingertip PPG (mean R/G/B per frame, 30 fps), 60 bpm, clean.", 40, 60, 101),
        _ppg("ppg_clean_90bpm", "Fingertip PPG, 90 bpm, clean.", 40, 90, 102),
        _ppg("ppg_clean_150bpm", "Fingertip PPG, 150 bpm, clean.", 40, 150, 103),
        _ppg("ppg_jitter_drops", "72 bpm; +-8 ms frame jitter, 5% dropped frames, a 0.3 s hiccup at 25 s.",
             45, 72, 104, jitter_s=0.008, drop_frac=0.05, hiccup=(25.0, 0.3)),
        _ppg("ppg_motion_burst", "72 bpm; motion burst 25-31 s: no_finger/settling, then ok again after settle.",
             60, 72, 105, events=[(25.0, 31.0, "burst")]),
        _ppg("ppg_saturation", "72 bpm; torch saturates the sensor 20-32 s: saturated, then recovers.",
             55, 72, 106, events=[(20.0, 32.0, "saturate")]),
        _ppg("ppg_finger_lift", "72 bpm; finger lifted 20-28 s: no_finger, then settling and ok.",
             55, 72, 107, events=[(20.0, 28.0, "lift")]),
        _ppg("ppg_harmonic_trap", "52 bpm with a 2nd harmonic nearly as strong as the fundamental: fundamental wins.",
             40, 52, 52, harmonic=0.5, amp=2.0),
        _hrv("hrv_clean_30ms", "Fingertip trace, 70 s, beats follow an RR series with true RMSSD 30 ms: reported.", 70, 30, 301),
        _hrv("hrv_clean_80ms", "70 s, true RMSSD 80 ms (high RSA): reported.", 70, 80, 302),
        _hrv("hrv_ectopics", "70 s, true RMSSD 30 ms, 2 premature beats and 2 missed beats: rejected, still reported.",
             70, 30, 303, ectopic=(12, 45), missed=(25, 62)),
        _hrv("hrv_too_many_artifacts", "70 s, 9 premature beats: fewer than 90 % of IBIs kept, withheld.",
             70, 30, 304, ectopic=(8, 17, 26, 35, 44, 53, 62, 71, 78)),
        _hrv("hrv_too_short", "50 s of contact: withheld as too_short.", 50, 30, 305),
        _rppg("rppg_clean_light_72bpm", "Face-ROI mean RGB (30 fps), synthetic light-skin profile, 72 bpm, clean.",
              32, 72, 201, "light"),
        _rppg("rppg_clean_medium_90bpm", "Synthetic medium-skin profile, 90 bpm, clean.", 32, 90, 202, "medium"),
        _rppg("rppg_clean_dark_60bpm",
              "Synthetic dark-skin profile (weaker pulse, darker DC), 60 bpm: lower SNR, some windows withheld.",
              32, 60, 203, "dark"),
        _rppg("rppg_motion_burst",
              "78 bpm; face motion + strong intensity noise 20-23 s: motion gate, then ok again once the window clears.",
              50, 78, 204, "medium", events=[(20.0, 23.0, "motion", 0.02)]),
        _rppg("rppg_lighting_step",
              "84 bpm; the light halves at 20 s: lighting, then settling and ok at the new level.",
              45, 84, 205, "light", events=[(20.0, 99.0, "lightstep", 0.5)]),
        _rppg("rppg_face_lost", "72 bpm; face out of view 18-24 s: no_face, then settling and ok.",
              45, 72, 206, "medium", events=[(18.0, 24.0, "noface")]),
        _rppg("rppg_small_face", "72 bpm; face too far away (tiny ROI) 15-20 s: too_small, then settling and ok.",
              38, 72, 207, "light", events=[(15.0, 20.0, "small")]),
        _rppg("rppg_skin_filter",
              "72 bpm; the skin filter rejects ~95 % of the ROI 15-20 s (e.g. strongly coloured light): skin_filter, then settling and ok.",
              38, 72, 209, "light", events=[(15.0, 20.0, "skinless")]),
        _rppg("rppg_disagree_flicker",
              "70 bpm plus a blue-ish colour flicker at 100 bpm: POS and CHROM disagree, so the reading is withheld.",
              32, 70, 208, "light", noise=0.02, interference=(100.0, 0.006, (0.2, 0.2, 1.0))),
    ] + _audio_fixtures() + _mobility_fixtures()
    return {f["name"] + ".json": json.dumps(f, separators=(",", ":"), allow_nan=False) + "\n"
            for f in fx}


# ------------------------------------------------------------------ comparison (shared semantics)

_TOL_KEYS = {"value": "value_bpm", "quality": "quality", "t": "t_s", "t_onset": "event_time_s",
             "t_confirmed": "event_time_s", "torso_angle_deg": "torso_angle_deg", "motion": "motion",
             "bpm": "bpm", "bpm_ibi": "bpm", "bpm_chrom": "bpm", "rmssd_ms": "hrv_ms", "sdnn_ms": "hrv_ms", "mean_hr_bpm": "bpm", "final_ibis_ms": "ibi_ms", "snr": "snr", "pi": "pi",
             "envelope": "env_log10"}
# mobility fixtures only (scoped by input_level so cough onsets, HRV durations etc. keep their exact comparison)
_MOB_TOL_KEYS = {"time_s": "mob_time_s", "rep_times_s": "mob_time_s", "rise": "mob_time_s", "walk_out": "mob_time_s", "turn_1": "mob_time_s",
                 "walk_back": "mob_time_s", "turn_2": "mob_time_s", "sit": "mob_time_s", "t_start": "mob_time_s", "duration_s": "mob_time_s",
                 "speed_m_s": "speed_m_s", "distance_m": "distance_m", "cadence_spm": "cadence_spm", "quality": "quality"}
_TOL_DEFAULTS = {"hrv_ms": 0.5, "bpm": 0.3, "ibi_ms": 5.0, "snr": 0.05, "pi": 0.01, "value_bpm": 0.5, "quality": 0.05, "t_s": 0.05, "event_time_s": 0.1,
                 "torso_angle_deg": 0.05, "motion": 0.01, "env_log10": 1e-4, "mob_time_s": 0.1, "speed_m_s": 0.02, "distance_m": 0.1, "cadence_spm": 1.0}
_TOL_OTHER = 1e-4   # numeric fields with no named tolerance (e.g. brightness)


def compare_outputs(fx: dict, actual, expected=None) -> list[str]:
    """Structural + numeric comparison within the fixture's own tolerance; returns mismatches.

    Numbers compare within tolerance (by field name); everything else (``reported``, ``kind``,
    ``posture``, ...) and every array length compares EXACTLY. Same semantics as the TS
    ``compareOutputs`` in web/src/core/parity.ts."""
    tol = fx["tolerance"]
    errs: list[str] = []

    def tol_for(field):
        key = (_MOB_TOL_KEYS if fx.get("input_level") == "mobility" else _TOL_KEYS).get(field)
        if key is None:
            return _TOL_OTHER
        v = tol.get(key, _TOL_DEFAULTS[key])
        return v if isinstance(v, (int, float)) else _TOL_OTHER

    def walk(e, a, path, field):
        if isinstance(e, bool) or isinstance(a, bool) or e is None or isinstance(e, str):
            if e != a or type(e) is not type(a):
                errs.append(f"{path}: expected {e!r} got {a!r}")
        elif isinstance(e, (int, float)):
            if not isinstance(a, (int, float)) or not abs(e - a) <= tol_for(field) + 1e-9:
                errs.append(f"{path}: expected {e} got {a} (tol {tol_for(field)})")
        elif isinstance(e, list):
            if not isinstance(a, list) or len(a) != len(e):
                errs.append(f"{path}: length {len(a) if isinstance(a, list) else a!r} != {len(e)}")
            else:
                for i, (x, y) in enumerate(zip(e, a)):
                    walk(x, y, f"{path}[{i}]", field)
        elif isinstance(e, dict):
            if not isinstance(a, dict) or set(a) != set(e):
                errs.append(f"{path}: keys differ")
            else:
                for k in e:
                    walk(e[k], a[k], f"{path}.{k}", k)
        else:
            errs.append(f"{path}: unsupported type")

    walk(fx["expected"] if expected is None else expected, actual, "expected", "")
    return errs


def compare_fixture_files(old: dict, new: dict) -> list[str]:
    """Committed vs regenerated: metadata exact; input and expected within tolerance."""
    errs = [f"{k} differs" for k in ("name", "description", "input_level", "params_version", "tolerance")
            if old.get(k) != new.get(k)]
    probe = {"tolerance": {}}   # inputs: tight numeric tolerance, exact structure
    errs += ["input: " + e for e in compare_outputs(probe, new["input"], old["input"])]
    errs += compare_outputs(old, new["expected"])
    return errs


def run_fixture(fx: dict) -> dict:
    """Run the Python reference on a fixture's input (self-consistency)."""
    return RUNNERS[fx["input_level"]](fx["input"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if committed fixtures differ")
    ap.add_argument("--dir", default=str(FIXTURE_DIR))
    a = ap.parse_args(argv)
    out, new = Path(a.dir), generate_all()
    if a.check:
        old = {p.name: p.read_text() for p in out.glob("*.json") if p.name != PENDING_FILE} \
            if out.is_dir() else {}
        bad = sorted(set(old) ^ set(new))
        for k in sorted(set(old) & set(new)):
            errs = compare_fixture_files(json.loads(old[k]), json.loads(new[k]))
            if errs:
                bad.append(f"{k} ({errs[0]})")
        if bad:
            print("fixtures out of date: " + ", ".join(bad)
                  + "\nrun: python -m carelens.fixtures", file=sys.stderr)
            return 1
        print(f"fixtures up to date ({len(new)} files)")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.json"):
        if p.name not in new and p.name != PENDING_FILE:
            p.unlink()
    for k, v in new.items():
        (out / k).write_text(v)
    print(f"wrote {len(new)} fixtures, {sum(len(v) for v in new.values()) / 1e3:.0f} kB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
