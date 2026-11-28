"""Load and validate ``shared/params.json``: the single source of every tunable number.

The TypeScript port reads the same file. Models forbid unknown keys and have no defaults,
so a key added on one side only fails loudly. Not a medical device.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import math

from pydantic import BaseModel, ConfigDict, model_validator

PARAMS_PATH = Path(__file__).resolve().parent.parent / "shared" / "params.json"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Common(_M):
    still_motion_bodysizes_s: float
    min_quality: float


class Signal(_M):
    bandpass_order: int
    bandpass_nyquist_frac: float
    welch_min_segment_s: float
    welch_segment_cycles: float
    welch_nfft_min: int
    welch_nfft_oversample: int
    peak_window_frac: float
    estimator_min_samples: int
    estimator_window_fill_frac: float


class HeartRate(_M):
    window_s: float
    step_s: float
    band_lo_hz: float
    band_hi_hz: float
    min_brightness: float
    max_brightness: float


class Breathing(_M):
    window_s: float
    step_s: float
    band_lo_hz: float
    band_hi_hz: float


class Roi(_M):
    min_landmark_visibility: float
    min_box_px: int
    face_min_eye_distance_px: float
    face_top_eye_to_nose_frac: float
    face_half_width_eye_dist: float
    face_bottom_below_nose_eye_dist: float
    chest_half_width_shoulder_frac: float
    chest_half_width_torso_frac: float
    chest_center_along_torso_frac: float
    chest_half_height_torso_frac: float


class Vitals(_M):
    pipeline_min_roi_px: int   # Python pipeline's face-patch size gate (native pixels; ~3 m subject at 640x480)
    signal: Signal
    heart_rate: HeartRate
    breathing: Breathing
    roi: Roi


class Features(_M):
    min_visibility: float
    walk_speed_bodysizes_s: float
    smooth_s: float
    aspect_w_over_h: float
    lying_torso_angle_deg: float
    standing_leg_ratio: float
    default_leg_ratio: float


class Fall(_M):
    drop_window_s: float
    upright_deg: float
    lying_deg: float
    drop_frac: float
    min_angle_rate_deg_s: float
    hip_drop_min_rate_frac_s: float
    still_seconds: float
    pending_timeout_s: float


class Inactivity(_M):
    seconds: float


class Gait(_M):
    person_height_m: float
    min_walk_s: float
    gap_s: float


class Timeline(_M):
    max_step_s: float


class Events(_M):
    fall: Fall
    inactivity: Inactivity
    gait: Gait
    timeline: Timeline


class Tracking(_M):
    hold_s: float
    enter_s: float
    max_jump_frame_heights: float
    one_euro_min_cutoff_hz: float
    one_euro_beta: float
    one_euro_d_cutoff_hz: float
    one_euro_filter_default_beta: float


class PipelineParams(_M):
    clip_pre_s: float
    clip_buffer_s: float


class Scheduling(_M):
    """Browser frame scheduling (web/src/core/scheduler.ts). Not used by the Python pipeline, whose
    sources report camera gaps themselves; kept here so both sides validate the same file."""
    pose_hz: float
    pose_budget_ms: float
    gap_min_s: float
    inactivity_demo_default_s: float
    inactivity_demo_options_s: List[float]


class ChestShift(_M):
    """Browser chest image->displacement step (web/src/core/chestShift.ts); Python uses cv2.phaseCorrelate."""
    frame_w_px: int
    patch_max_w_px: int
    max_shift_px: int
    min_corr: float


class _Reserved(_M):
    reserved: bool


class BandReserved(_Reserved):
    band_lo_hz: float
    band_hi_hz: float


def contact_ppg_errors(p: dict) -> List[str]:
    """Consistency checks for ``contact_ppg`` (mirrored by assertContactPpgParams in web/src/core/ppg.ts)."""
    e: List[str] = []
    if not 0 < p["taper_stop_lo_hz"] < p["band_lo_hz"] < p["band_hi_hz"] < p["taper_stop_hi_hz"] < p["fs_hz"] / 2:
        e.append("need 0 < taper_stop_lo < band_lo < band_hi < taper_stop_hi < Nyquist")
    if not 0 < p["pi_min"] < p["q_pi_full_lo"] < p["q_pi_full_hi"] < p["pi_max"]:
        e.append("need 0 < pi_min < q_pi_full_lo < q_pi_full_hi < pi_max")
    if min(p["q_w_snr"], p["q_w_agree"], p["q_w_pi"]) < 0 or abs(p["q_w_snr"] + p["q_w_agree"] + p["q_w_pi"] - 1.0) > 1e-9:
        e.append("quality weights must be >= 0 and sum to 1")
    if not (p["min_snr"] > 0 and p["q_snr_full_db"] > 10.0 * math.log10(max(p["min_snr"], 1e-300))):
        e.append("need min_snr > 0 and q_snr_full_db above the gate in dB")
    for k in ("window_s", "step_s", "fs_hz", "max_gap_s", "min_rate_hz", "settle_s", "agree_bpm", "snr_halfwidth_hz",
              "harm_halfwidth_hz", "beat_max_bpm", "min_prom", "finger_dr_max", "finger_ratio_min"):
        if not p[k] > 0:
            e.append(f"{k} must be > 0")
    if p["zero_pad_factor"] < 1 or p["min_ibis"] < 1:
        e.append("zero_pad_factor and min_ibis must be >= 1")
    if not 0 <= p["pi_lo_pct"] < p["pi_hi_pct"] <= 100:
        e.append("need 0 <= pi_lo_pct < pi_hi_pct <= 100")
    if p["harm_w2"] < 0 or p["harm_w3"] < 0 or not 0 < p["max_clip_frac"] <= 1:
        e.append("harm weights >= 0 and 0 < max_clip_frac <= 1")
    return e


class ContactPpg(_M):
    window_s: float
    step_s: float
    fs_hz: float
    max_gap_s: float
    min_rate_hz: float
    band_lo_hz: float
    band_hi_hz: float
    taper_stop_lo_hz: float
    taper_stop_hi_hz: float
    zero_pad_factor: int
    snr_halfwidth_hz: float
    harm_halfwidth_hz: float
    harm_w2: float
    harm_w3: float
    pi_lo_pct: float
    pi_hi_pct: float
    beat_max_bpm: float
    min_prom: float
    min_ibis: int
    finger_r_min: float
    finger_ratio_min: float
    finger_dr_max: float
    max_clip_frac: float
    settle_s: float
    min_snr: float
    agree_bpm: float
    pi_min: float
    pi_max: float
    q_snr_full_db: float
    q_pi_full_lo: float
    q_pi_full_hi: float
    q_w_snr: float
    q_w_agree: float
    q_w_pi: float

    @model_validator(mode="after")
    def _check(self) -> "ContactPpg":
        errs = contact_ppg_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


def hrv_errors(p: dict) -> List[str]:
    """Consistency checks for ``hrv`` (mirrored by hrvErrors in web/src/core/hrv.ts)."""
    e: List[str] = []
    if not 0 < p["ibi_min_ms"] < p["ibi_max_ms"]:
        e.append("need 0 < ibi_min_ms < ibi_max_ms")
    if not 0 < p["min_duration_s"] <= p["max_duration_s"]:
        e.append("need 0 < min_duration_s <= max_duration_s")
    for k in ("min_rate_hz", "max_dev_frac"):
        if not p[k] > 0:
            e.append(f"{k} must be > 0")
    for k in ("min_kept_frac", "min_ok_frac", "min_amp_frac"):
        if not 0 < p[k] <= 1:
            e.append(f"{k} must be in (0, 1]")
    if p["neighbours"] < 1 or p["min_kept_beats"] < 2:
        e.append("neighbours must be >= 1 and min_kept_beats >= 2")
    return e


class Hrv(_M):
    min_duration_s: float
    max_duration_s: float
    min_rate_hz: float
    ibi_min_ms: float
    ibi_max_ms: float
    neighbours: int
    max_dev_frac: float
    min_kept_frac: float
    min_kept_beats: int
    min_ok_frac: float
    min_amp_frac: float

    @model_validator(mode="after")
    def _check(self) -> "Hrv":
        errs = hrv_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


def rppg_errors(p: dict) -> List[str]:
    """Consistency checks for ``rppg`` (mirrored by rppgErrors in web/src/core/rppg.ts)."""
    e: List[str] = []
    if not 0 < p["taper_stop_lo_hz"] < p["band_lo_hz"] < p["band_hi_hz"] < p["taper_stop_hi_hz"] < p["fs_hz"] / 2:
        e.append("need 0 < taper_stop_lo < band_lo < band_hi < taper_stop_hi < Nyquist")
    if min(p["q_w_snr"], p["q_w_agree"], p["q_w_motion"]) < 0 or abs(p["q_w_snr"] + p["q_w_agree"] + p["q_w_motion"] - 1.0) > 1e-9:
        e.append("quality weights must be >= 0 and sum to 1")
    if not (p["min_snr"] > 0 and p["q_snr_full_db"] > 10.0 * math.log10(max(p["min_snr"], 1e-300))):
        e.append("need min_snr > 0 and q_snr_full_db above the gate in dB")
    for k in ("window_s", "step_s", "fs_hz", "max_gap_s", "min_rate_hz", "settle_s", "agree_bpm", "snr_halfwidth_hz",
              "harm_halfwidth_hz", "pos_window_s", "min_roi_px", "motion_max", "max_brightness_rate",
              "brightness_lag_s", "lighting_hold_s"):
        if not p[k] > 0:
            e.append(f"{k} must be > 0")
    if p["zero_pad_factor"] < 1:
        e.append("zero_pad_factor must be >= 1")
    if p["harm_w2"] < 0 or p["harm_w3"] < 0:
        e.append("harm weights must be >= 0")
    if not 8 <= p["pos_window_s"] * p["fs_hz"] <= 0.5 * p["window_s"] * p["fs_hz"]:
        e.append("need 8 samples <= pos_window_s*fs_hz <= half of window_s*fs_hz")
    if not 0 <= p["min_skin_frac"] <= 1:
        e.append("need 0 <= min_skin_frac <= 1")
    if not 0 <= p["motion_frac_max"] < 1:
        e.append("need 0 <= motion_frac_max < 1")
    if not 0 <= p["min_brightness"] < p["max_brightness"] <= 255:
        e.append("need 0 <= min_brightness < max_brightness <= 255")
    if not (0 <= p["skin_cr_min"] < p["skin_cr_max"] <= 255 and 0 <= p["skin_cb_min"] < p["skin_cb_max"] <= 255):
        e.append("need 0 <= skin_c*_min < skin_c*_max <= 255")
    for name, idx in p["roi_landmarks"].items():
        if len(idx) < 3 or len(set(idx)) != len(idx) or min(idx) < 0 or max(idx) > 477:
            e.append(f"roi_landmarks.{name}: need >= 3 distinct indices in 0..477")
    return e


class RoiLandmarks(_M):
    forehead: List[int]
    left_cheek: List[int]
    right_cheek: List[int]


class Rppg(_M):
    window_s: float
    step_s: float
    fs_hz: float
    max_gap_s: float
    min_rate_hz: float
    band_lo_hz: float
    band_hi_hz: float
    taper_stop_lo_hz: float
    taper_stop_hi_hz: float
    zero_pad_factor: int
    snr_halfwidth_hz: float
    harm_halfwidth_hz: float
    harm_w2: float
    harm_w3: float
    pos_window_s: float
    min_roi_px: int
    min_skin_frac: float
    motion_max: float
    motion_frac_max: float
    min_brightness: float
    max_brightness: float
    max_brightness_rate: float
    brightness_lag_s: float
    lighting_hold_s: float
    settle_s: float
    min_snr: float
    agree_bpm: float
    q_snr_full_db: float
    q_w_snr: float
    q_w_agree: float
    q_w_motion: float
    skin_filter: bool
    skin_cr_min: int
    skin_cr_max: int
    skin_cb_min: int
    skin_cb_max: int
    roi_landmarks: RoiLandmarks
    methods: List[str]

    @model_validator(mode="after")
    def _check(self) -> "Rppg":
        errs = rppg_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


class AudioEnvelope(_M):
    frame_len: int
    hop: int
    band_lo_hz: float
    band_hi_hz: float
    eps: float


class Yamnet(_M):
    window_samples: int
    hop_samples: int
    speech_label: str


class AudioBreathing(_M):
    window_s: float
    step_s: float
    band_lo_hz: float
    band_hi_hz: float
    zero_pad_factor: int
    snr_halfwidth_hz: float
    harm_halfwidth_hz: float
    harm_w2: float
    harm_w3: float
    settle_s: float
    min_range_log10: float
    clip_level: float
    clip_frac_max: float
    speech_thr: float
    speech_frac_max: float
    min_snr: float
    q_snr_full_db: float
    sub_ratio: float
    smooth_s: float
    bump_prom_frac: float
    min_bump_s: float
    min_bumps: int
    min_alt_bumps: int
    rate_tol: float
    alt_min: float
    width_min: float


class LabelEvents(_M):
    """Hysteresis event detector over per-frame scores of one YAMNet label (carelens/audio.py)."""
    label: str
    on_thr: float
    off_thr: float
    min_frames: int
    merge_gap_s: float
    refractory_s: float


def audio_errors(p: dict) -> List[str]:
    """Consistency checks for ``audio`` (mirrored by audioErrors in web/src/core/audio.ts)."""
    e: List[str] = []
    env, b = p["envelope"], p["breathing"]
    n = env["frame_len"]
    if n < 2 or n & (n - 1) or env["hop"] < 1 or env["hop"] > n:
        e.append("envelope.frame_len must be a power of two >= 2 and 1 <= hop <= frame_len")
    if not 0 <= env["band_lo_hz"] < env["band_hi_hz"] <= p["fs_hz"] / 2 or not env["eps"] > 0:
        e.append("need 0 <= envelope.band_lo < band_hi <= Nyquist and eps > 0")
    env_fs = p["fs_hz"] / max(env["hop"], 1)
    if not 0 < b["band_lo_hz"] < b["band_hi_hz"] < env_fs / 2:
        e.append("need 0 < breathing.band_lo < band_hi < envelope Nyquist")
    if not 0 < b["settle_s"] <= b["window_s"] or not b["step_s"] > 0 or b["zero_pad_factor"] < 1:
        e.append("need 0 < settle_s <= window_s, step_s > 0, zero_pad_factor >= 1")
    for k in ("snr_halfwidth_hz", "harm_halfwidth_hz", "min_range_log10", "min_snr", "sub_ratio", "smooth_s", "min_bump_s", "rate_tol", "alt_min", "width_min"):
        if not b[k] > 0:
            e.append(f"breathing.{k} must be > 0")
    if not (0 < b["clip_level"] <= 1 and 0 <= b["clip_frac_max"] < 1 and 0 <= b["speech_thr"] <= 1 and 0 <= b["speech_frac_max"] < 1):
        e.append("clip_level/speech_thr in (0, 1], clip_frac_max/speech_frac_max in [0, 1)")
    if b["harm_w2"] < 0 or b["harm_w3"] < 0 or not b["q_snr_full_db"] > 10.0 * math.log10(max(b["min_snr"], 1e-300)):
        e.append("harm weights >= 0 and q_snr_full_db above the gate in dB")
    if not (0 < b["bump_prom_frac"] < 1 and b["min_bumps"] >= 3 and b["min_alt_bumps"] >= 4 and b["rate_tol"] < 0.5):
        e.append("need 0 < bump_level_frac < 1, min_bumps >= 3, rate_tol < 0.5")
    y = p["yamnet"]
    if y["window_samples"] < 1 or not 1 <= y["hop_samples"] <= y["window_samples"]:
        e.append("need 1 <= yamnet.hop_samples <= window_samples")
    lab = p["cough"]
    if not 0 < lab["off_thr"] <= lab["on_thr"] <= 1 or lab["min_frames"] < 1 or lab["merge_gap_s"] < 0 or lab["refractory_s"] < 0:
        e.append("cough: need 0 < off_thr <= on_thr <= 1, min_frames >= 1, merge_gap_s/refractory_s >= 0")
    return e


class Audio(_M):
    classifier: str
    labels: List[str]
    fs_hz: int
    envelope: AudioEnvelope
    yamnet: Yamnet
    breathing: AudioBreathing
    cough: LabelEvents

    @model_validator(mode="after")
    def _check(self) -> "Audio":
        errs = audio_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


class AudioEvents(_M):
    """W10 per-label YAMNet event detectors (snoring, distress sounds, speech) and the fall/sound corroboration window."""
    corroborate_s: float
    cue_labels: List[str]
    snoring: LabelEvents
    screaming: LabelEvents
    crying: LabelEvents
    glass: LabelEvents
    thump: LabelEvents
    speech: LabelEvents

    def sections(self) -> List[LabelEvents]:
        return [self.snoring, self.screaming, self.crying, self.glass, self.thump, self.speech]

    @model_validator(mode="after")
    def _check(self) -> "AudioEvents":
        errs = audio_events_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


def audio_events_errors(p: dict) -> List[str]:
    """Consistency checks for ``audio_events`` (mirrored by audioEventsErrors in web/src/core/sleepSounds.ts)."""
    e: List[str] = []
    secs = [p[k] for k in ("snoring", "screaming", "crying", "glass", "thump", "speech")]
    for k in ("snoring", "screaming", "crying", "glass", "thump", "speech"):
        s = p[k]
        if not 0 < s["off_thr"] <= s["on_thr"] <= 1 or s["min_frames"] < 1 or s["merge_gap_s"] < 0 or s["refractory_s"] < 0:
            e.append(f"{k}: need 0 < off_thr <= on_thr <= 1, min_frames >= 1, merge_gap_s/refractory_s >= 0")
    labels = [s["label"] for s in secs]
    if any(not l for l in labels) or len(set(labels)) != len(labels):
        e.append("section labels must be non-empty and distinct")
    if not p["corroborate_s"] > 0:
        e.append("corroborate_s must be > 0")
    if not p["cue_labels"] or any(l not in labels for l in p["cue_labels"]):
        e.append("cue_labels must be a non-empty subset of the section labels")
    return e


class SleepBump(_M):
    smooth_s: float
    floor_win_s: float
    rise_on_log10: float
    rise_off_frac: float
    min_bump_s: float
    max_bump_s: float
    merge_s: float
    silence_log10: float
    spike_ratio: float
    typ_bumps: int


class SleepSounds(_M):
    """W10 possible pauses in breathing sounds (carelens/sleep_sounds.py)."""
    pause_min_s: float
    pause_max_s: float
    pre_regular_s: float
    pre_min_bumps: int
    pre_cv_max: float
    rate_min_bpm: float
    rate_max_bpm: float
    min_valid_regular_s: float
    speech_thr: float
    overlap_margin_s: float
    bump: SleepBump

    @model_validator(mode="after")
    def _check(self) -> "SleepSounds":
        errs = sleep_sounds_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


def sleep_sounds_errors(p: dict) -> List[str]:
    """Consistency checks for ``sleep_sounds`` (mirrored by sleepSoundsErrors in web/src/core/sleepSounds.ts)."""
    e: List[str] = []
    b = p["bump"]
    if not 0 < p["pause_min_s"] < p["pause_max_s"]:
        e.append("need 0 < pause_min_s < pause_max_s")
    if not (p["pre_regular_s"] > 0 and p["pre_min_bumps"] >= 3 and 0 < p["pre_cv_max"] < 1 and p["min_valid_regular_s"] > 0):
        e.append("need pre_regular_s > 0, pre_min_bumps >= 3, 0 < pre_cv_max < 1, min_valid_regular_s > 0")
    if not 0 < p["rate_min_bpm"] < p["rate_max_bpm"]:
        e.append("need 0 < rate_min_bpm < rate_max_bpm")
    if not (0 <= p["speech_thr"] <= 1 and p["overlap_margin_s"] >= 0):
        e.append("need 0 <= speech_thr <= 1 and overlap_margin_s >= 0")
    if not (b["smooth_s"] > 0 and b["rise_on_log10"] > 0 and 0 < b["rise_off_frac"] < 1 and 0 < b["min_bump_s"] < b["max_bump_s"]
            and b["spike_ratio"] > 1 and b["typ_bumps"] >= 1):
        e.append("bump: need smooth_s > 0, rise_on_log10 > 0, 0 < rise_off_frac < 1, 0 < min_bump_s < max_bump_s, spike_ratio > 1, typ_bumps >= 1")
    if not 0 <= b["merge_s"] <= 60.0 / max(p["rate_max_bpm"], 1e-9):
        e.append("bump.merge_s must be in [0, 60 / rate_max_bpm] (in-band breaths are never merged)")
    if p["rate_min_bpm"] > 0 and not b["floor_win_s"] >= 60.0 / p["rate_min_bpm"]:
        e.append("bump.floor_win_s must cover at least one slowest breath (>= 60 / rate_min_bpm)")
    return e


def mobility_errors(p: dict) -> List[str]:
    """Consistency checks for ``mobility`` (mirrored by mobilityErrors in web/src/core/mobility.ts)."""
    e: List[str] = []
    if not 0 < p["sit_deg"] < p["rise_deg"] < p["stand_deg"] <= 180:
        e.append("need 0 < sit_deg < rise_deg < stand_deg <= 180")
    for k in ("min_visibility", "min_quality", "stand_hip_frac"):
        if not 0 < p[k] <= 1:
            e.append(f"{k} must be in (0, 1]")
    if not 0 <= p["leg_vis_margin"] < 1:
        e.append("leg_vis_margin must be in [0, 1)")
    for k in ("stand_dwell_s", "sit_dwell_s", "chair_stand_s", "sts_early_s", "sts_max_s", "tug_max_s", "walk_window_s"):
        if not p[k] > 0:
            e.append(f"{k} must be > 0")
    if not 2 <= p["sts_reps"] <= 50:
        e.append("sts_reps must be in [2, 50]")
    t, w = p["tug"], p["walk"]
    if not (t["vel_win_s"] > 0 and 0 < t["turn_speed_frac"] < 1 and t["min_excursion_bodies"] > 0 and 0 < t["min_return_frac"] <= 1):
        e.append("tug: need vel_win_s > 0, 0 < turn_speed_frac < 1, min_excursion_bodies > 0, 0 < min_return_frac <= 1")
    if not 0 < w["height_min_m"] <= w["height_default_m"] <= w["height_max_m"]:
        e.append("walk: need 0 < height_min_m <= height_default_m <= height_max_m")
    for k in ("nose_ankle_frac", "height_tol_frac", "steady_frac"):
        if not 0 < w[k] < 1:
            e.append(f"walk.{k} must be in (0, 1)")
    for k in ("min_speed_bodies_s", "vel_win_s", "gap_s", "min_dist_m", "min_dur_s", "cadence_hyst_frac"):
        if not w[k] > 0:
            e.append(f"walk.{k} must be > 0")
    if w["min_samples"] < 3 or w["cadence_min_crossings"] < 3:
        e.append("walk: min_samples and cadence_min_crossings must be >= 3")
    return e


class MobilityTug(_M):
    vel_win_s: float
    turn_speed_frac: float
    min_excursion_bodies: float
    min_return_frac: float


class MobilityWalk(_M):
    height_default_m: float
    height_min_m: float
    height_max_m: float
    nose_ankle_frac: float
    min_speed_bodies_s: float
    vel_win_s: float
    gap_s: float
    height_tol_frac: float
    steady_frac: float
    min_dist_m: float
    min_dur_s: float
    min_samples: int
    cadence_hyst_frac: float
    cadence_min_crossings: int


class Mobility(_M):
    """Guided mobility tests from pose landmarks (W7, Experimental). Wellbeing trends only: no norms or cut-offs."""
    min_visibility: float
    leg_vis_margin: float
    min_quality: float
    sit_deg: float
    rise_deg: float
    stand_deg: float
    stand_hip_frac: float
    stand_dwell_s: float
    sit_dwell_s: float
    chair_stand_s: float
    sts_reps: int
    sts_early_s: float
    sts_max_s: float          # UI time limits: the live test gives up after this long
    tug_max_s: float
    walk_window_s: float
    tug: MobilityTug
    walk: MobilityWalk

    @model_validator(mode="after")
    def _check(self) -> "Mobility":
        errs = mobility_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


def tremor_errors(p: dict) -> List[str]:
    """Consistency checks for ``tremor`` (mirrored by tremorErrors in web/src/core/tremor.ts)."""
    e: List[str] = []
    if not 0 < p["taper_stop_lo_hz"] < p["band_lo_hz"] < p["band_hi_hz"] < p["taper_stop_hi_hz"] < p["fs_hz"] / 2:
        e.append("need 0 < taper_stop_lo < band_lo < band_hi < taper_stop_hi < Nyquist")
    if not 0 < p["motion_band_hi_hz"] < p["taper_stop_lo_hz"]:
        e.append("need 0 < motion_band_hi_hz < taper_stop_lo_hz")
    if not 2.0 * p["band_hi_hz"] < p["min_rate_hz"] <= p["fs_hz"]:
        e.append("need 2 * band_hi_hz < min_rate_hz <= fs_hz (never report an aliased frequency)")
    if not 0 < p["min_window_s"] <= p["test_s"]:
        e.append("need 0 < min_window_s <= test_s")
    if not 0 <= p["edge_trim_s"] < p["min_window_s"] / 4:
        e.append("edge_trim_s must be in [0, min_window_s / 4)")
    for k in ("max_gap_s", "prom_halfwidth_hz", "moving_max_pct", "noise_floor_pct"):
        if not p[k] > 0:
            e.append(f"{k} must be > 0")
    for k in ("min_hand_score", "min_prominence"):
        if not 0 < p[k] <= 1:
            e.append(f"{k} must be in (0, 1]")
    if p["zero_pad_factor"] < 1:
        e.append("zero_pad_factor must be >= 1")
    return e


class Tremor(_M):
    """Rhythmic hand movement (3-12 Hz) from hand landmarks (W8, Experimental). Wellbeing trend only."""
    band_lo_hz: float
    band_hi_hz: float
    taper_stop_lo_hz: float
    taper_stop_hi_hz: float
    fs_hz: float
    test_s: float
    min_window_s: float
    min_rate_hz: float
    max_gap_s: float
    min_hand_score: float
    zero_pad_factor: int
    edge_trim_s: float
    prom_halfwidth_hz: float
    motion_band_hi_hz: float
    moving_max_pct: float
    min_prominence: float
    noise_floor_pct: float

    @model_validator(mode="after")
    def _check(self) -> "Tremor":
        errs = tremor_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


def blink_errors(p: dict) -> List[str]:
    """Consistency checks for ``blink`` (mirrored by blinkErrors in web/src/core/blink.ts)."""
    e: List[str] = []
    for side in ("left", "right"):
        pts = p["eye_landmarks"][side]
        if len(pts) != 6 or len(set(pts)) != 6 or min(pts) < 0:
            e.append(f"eye_landmarks.{side} must be 6 distinct landmark indices (p1..p6: corner, 2 upper lid, corner, 2 lower lid)")
    if sorted(p["pose_landmarks"]) != ["chin", "forehead", "nose"]:
        e.append("pose_landmarks needs exactly nose, forehead, chin")
    if not 0 < p["perclos_frac"] < p["closed_frac"] < p["reopen_frac"] < 1:
        e.append("need 0 < perclos_frac < closed_frac < reopen_frac < 1")
    if not 0 < p["blink_min_ms"] < p["blink_max_ms"]:
        e.append("need 0 < blink_min_ms < blink_max_ms")
    if not 0 < p["min_window_s"] <= p["rate_window_s"]:
        e.append("need 0 < min_window_s <= rate_window_s")
    if not 0 < p["baseline_pct"] <= 100:
        e.append("baseline_pct must be in (0, 100]")
    if not p["min_rate_hz"] >= 1000.0 / p["blink_min_ms"]:
        e.append("min_rate_hz is too low to resolve blink_min_ms")
    for k in ("baseline_s", "fps_window_s", "max_dt_s", "min_eye_width", "max_ear", "yaw_max_ratio", "pitch_tol", "pitch_neutral", "min_rate_hz"):
        if not p[k] > 0:
            e.append(f"{k} must be > 0")
    if not p["max_dt_s"] < p["fps_window_s"]:
        e.append("need max_dt_s < fps_window_s")
    return e


class EyeLandmarks(_M):
    left: List[int]
    right: List[int]


class PoseLandmarks(_M):
    nose: int
    forehead: int
    chin: int


class Blink(_M):
    """Blink rate, eyes-closed time and long closures from FaceLandmarker eye points (W9, Experimental)."""
    eye_landmarks: EyeLandmarks
    pose_landmarks: PoseLandmarks
    baseline_s: float
    baseline_pct: float
    closed_frac: float
    reopen_frac: float
    perclos_frac: float
    blink_min_ms: float
    blink_max_ms: float
    rate_window_s: float
    min_window_s: float
    min_rate_hz: float
    fps_window_s: float
    max_dt_s: float
    min_eye_width: float
    max_ear: float
    yaw_max_ratio: float
    pitch_neutral: float
    pitch_tol: float

    @model_validator(mode="after")
    def _check(self) -> "Blink":
        errs = blink_errors(self.model_dump())
        if errs:
            raise ValueError("; ".join(errs))
        return self


class Activity(_M):
    """Active / still / away time over a session (W9, Experimental): the session is reported once this long."""
    min_report_s: float

    @model_validator(mode="after")
    def _check(self) -> "Activity":
        if not self.min_report_s > 0:
            raise ValueError("min_report_s must be > 0")
        return self


class Params(_M):
    version: int
    description: str
    units: Dict[str, str]
    common: Common
    vitals: Vitals
    features: Features
    events: Events
    tracking: Tracking
    pipeline: PipelineParams
    scheduling: Scheduling
    chest_shift: ChestShift
    contact_ppg: ContactPpg
    hrv: Hrv
    rppg: Rppg
    breathing_signal: BandReserved
    tremor: Tremor
    blink: Blink
    activity: Activity
    mobility: Mobility
    audio: Audio
    audio_events: AudioEvents
    sleep_sounds: SleepSounds


def load_params(path: Path | str = PARAMS_PATH) -> Params:
    return Params.model_validate(json.loads(Path(path).read_text()))


@lru_cache(maxsize=1)
def _cached() -> Params:
    return load_params()


PARAMS: Params = _cached()
