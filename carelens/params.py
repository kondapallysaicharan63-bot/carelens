"""Load and validate ``shared/params.json``: the single source of every tunable number.

The TypeScript port reads the same file. Models forbid unknown keys and have no defaults,
so a key added on one side only fails loudly. Not a medical device.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, ConfigDict

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
    contact_ppg: BandReserved
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
