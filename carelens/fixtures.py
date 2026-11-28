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
from .events import FallDetector, InactivityDetector
from .features import FeatureExtractor
from .params import PARAMS
from .ppg import run_contact_ppg, synth_trace
from .rppg import run_face_rppg, synth_face_trace
from .synth import ASPECT, Scenario, Segment
from .vitals import BreathingRateEstimator, HeartRateEstimator

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


RUNNERS: dict[str, Callable[[dict], dict]] = {
    "roi_trace_hr": run_hr_trace,
    "chest_trace_breathing": run_breathing_trace,
    "landmarks": run_landmarks,
    "ppg_contact": run_contact_ppg,
    "face_rppg": run_face_rppg,
}

TOL_VITALS = {"value_bpm": 0.5, "quality": 0.05, "t_s": 0.05, "reported": "exact",
              "estimate_count": "exact"}
TOL_PPG = {"bpm": 0.3, "quality": 0.03, "t_s": 0.05, "ibi_ms": 5.0, "snr": 0.05, "pi": 0.01,
           "reported": "exact", "reason": "exact", "update_count": "exact"}
TOL_RPPG = {"bpm": 0.3, "quality": 0.03, "t_s": 0.05, "snr": 0.05,
            "reported": "exact", "reason": "exact", "update_count": "exact"}
TOL_EVENTS = {"event_time_s": 0.1, "kind": "exact", "event_count": "exact",
              "torso_angle_deg": 0.05, "motion": 0.01, "posture": "exact"}


def _fixture(name, description, level, inp, tolerance) -> dict:
    inp = _round(inp)
    return {"name": name, "description": description, "input_level": level,
            "params_version": PARAMS.version, "input": inp,
            "expected": RUNNERS[level](inp), "tolerance": tolerance}


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


def _ppg(name, desc, dur=40.0, bpm=72.0, seed=0, **kw):
    tr = synth_trace(dur, bpm, seed=seed, **kw)
    inp = {k: [round(v, 4) for v in tr[k]] for k in ("t", "r", "g", "b", "clip_frac")}
    inp["truth_bpm"] = bpm
    return _fixture(name, desc, "ppg_contact", inp, TOL_PPG)


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
    ]
    return {f["name"] + ".json": json.dumps(f, separators=(",", ":"), allow_nan=False) + "\n"
            for f in fx}


# ------------------------------------------------------------------ comparison (shared semantics)

_TOL_KEYS = {"value": "value_bpm", "quality": "quality", "t": "t_s", "t_onset": "event_time_s",
             "t_confirmed": "event_time_s", "torso_angle_deg": "torso_angle_deg", "motion": "motion",
             "bpm": "bpm", "bpm_ibi": "bpm", "bpm_chrom": "bpm", "final_ibis_ms": "ibi_ms", "snr": "snr", "pi": "pi"}
_TOL_DEFAULTS = {"bpm": 0.3, "ibi_ms": 5.0, "snr": 0.05, "pi": 0.01, "value_bpm": 0.5, "quality": 0.05, "t_s": 0.05, "event_time_s": 0.1,
                 "torso_angle_deg": 0.05, "motion": 0.01}
_TOL_OTHER = 1e-4   # numeric fields with no named tolerance (e.g. brightness)


def compare_outputs(fx: dict, actual, expected=None) -> list[str]:
    """Structural + numeric comparison within the fixture's own tolerance; returns mismatches.

    Numbers compare within tolerance (by field name); everything else (``reported``, ``kind``,
    ``posture``, ...) and every array length compares EXACTLY. Same semantics as the TS
    ``compareOutputs`` in web/src/core/parity.ts."""
    tol = fx["tolerance"]
    errs: list[str] = []

    def tol_for(field):
        key = _TOL_KEYS.get(field)
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
