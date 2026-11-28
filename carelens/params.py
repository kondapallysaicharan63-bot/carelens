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


class Rppg(BandReserved):
    methods: List[str]


class Audio(_Reserved):
    classifier: str
    labels: List[str]


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
    contact_ppg: ContactPpg
    rppg: Rppg
    breathing_signal: BandReserved
    tremor: BandReserved
    audio: Audio


def load_params(path: Path | str = PARAMS_PATH) -> Params:
    return Params.model_validate(json.loads(Path(path).read_text()))


@lru_cache(maxsize=1)
def _cached() -> Params:
    return load_params()


PARAMS: Params = _cached()
