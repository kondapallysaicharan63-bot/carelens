import json

import pytest
from pydantic import ValidationError

from carelens.params import PARAMS, PARAMS_PATH, Params, load_params


def test_loads_and_validates():
    assert PARAMS.version >= 1
    assert load_params() == PARAMS


def test_unknown_key_rejected():
    raw = json.loads(PARAMS_PATH.read_text())
    raw["vitals"]["heart_rate"]["bogus"] = 1
    with pytest.raises(ValidationError):
        Params.model_validate(raw)
    raw = json.loads(PARAMS_PATH.read_text())
    raw["surprise_section"] = {}
    with pytest.raises(ValidationError):
        Params.model_validate(raw)


def test_missing_key_rejected():
    raw = json.loads(PARAMS_PATH.read_text())
    del raw["events"]["fall"]["still_seconds"]
    with pytest.raises(ValidationError):
        Params.model_validate(raw)


def test_reserved_sections_and_values():
    assert (PARAMS.contact_ppg.band_lo_hz, PARAMS.contact_ppg.band_hi_hz) == (0.7, 3.0)  # W1: live, not reserved
    assert PARAMS.rppg.methods == ["POS", "CHROM"]
    assert (PARAMS.breathing_signal.band_lo_hz, PARAMS.breathing_signal.band_hi_hz) == (0.1, 0.7)
    assert (PARAMS.tremor.band_lo_hz, PARAMS.tremor.band_hi_hz) == (3.0, 12.0)
    assert PARAMS.audio.labels == ["Cough", "Snoring", "Screaming", "Crying, sobbing", "Glass", "Thump, thud"]


def test_module_defaults_equal_params():
    from carelens.events import FallDetector, GaitAnalyzer, InactivityDetector
    from carelens.features import FeatureExtractor
    from carelens.pipeline import Config
    from carelens.tracking import OneEuroFilter, TrackingConfig
    from carelens.vitals import BreathingRateEstimator, HeartRateEstimator

    P = PARAMS
    hr, br = HeartRateEstimator(method="green"), BreathingRateEstimator()
    pos = HeartRateEstimator()   # W2: POS is the default method; its window/step come from params.rppg
    assert pos.method == "pos" and (pos.window_s, pos.step_s) == (P.rppg.window_s, P.rppg.step_s)
    assert (hr.window_s, hr.step_s, hr.lo, hr.hi) == (
        P.vitals.heart_rate.window_s, P.vitals.heart_rate.step_s,
        P.vitals.heart_rate.band_lo_hz, P.vitals.heart_rate.band_hi_hz)
    assert (hr.min_quality, hr.still_motion) == (P.common.min_quality, P.common.still_motion_bodysizes_s)
    assert (hr.min_b, hr.max_b) == (P.vitals.heart_rate.min_brightness, P.vitals.heart_rate.max_brightness)
    assert (br.window_s, br.step_s, br.lo, br.hi) == (
        P.vitals.breathing.window_s, P.vitals.breathing.step_s,
        P.vitals.breathing.band_lo_hz, P.vitals.breathing.band_hi_hz)

    f = FallDetector()
    F = P.events.fall
    assert (f.drop_window_s, f.upright_deg, f.lying_deg, f.drop_frac, f.min_angle_rate,
            f.still_seconds, f.pending_timeout_s, f.still_motion) == (
        F.drop_window_s, F.upright_deg, F.lying_deg, F.drop_frac, F.min_angle_rate_deg_s,
        F.still_seconds, F.pending_timeout_s, P.common.still_motion_bodysizes_s)
    assert InactivityDetector().seconds == P.events.inactivity.seconds
    g = GaitAnalyzer()
    assert (g.height_m, g.min_walk_s, g.gap_s) == (
        P.events.gait.person_height_m, P.events.gait.min_walk_s, P.events.gait.gap_s)

    fe = FeatureExtractor()
    assert (fe.min_vis, fe.walk_speed, fe.smooth_s, fe.aspect) == (
        P.features.min_visibility, P.features.walk_speed_bodysizes_s, P.features.smooth_s,
        P.features.aspect_w_over_h)
    assert P.features.aspect_w_over_h == 4 / 3

    t, T = TrackingConfig(), P.tracking
    assert (t.hold_s, t.enter_s, t.max_jump, t.min_cutoff, t.beta, t.d_cutoff) == (
        T.hold_s, T.enter_s, T.max_jump_frame_heights, T.one_euro_min_cutoff_hz,
        T.one_euro_beta, T.one_euro_d_cutoff_hz)
    assert OneEuroFilter().beta == T.one_euro_filter_default_beta

    c = Config()
    assert (c.fall_still_seconds, c.inactivity_seconds, c.still_motion, c.person_height_m,
            c.clip_pre_s, c.clip_buffer_s, c.hr_min_quality, c.br_min_quality) == (
        F.still_seconds, P.events.inactivity.seconds, P.common.still_motion_bodysizes_s,
        P.events.gait.person_height_m, P.pipeline.clip_pre_s, P.pipeline.clip_buffer_s,
        P.common.min_quality, P.common.min_quality)


def test_overridable_defaults():
    from carelens.events import FallDetector
    assert FallDetector(still_seconds=3.0).still_seconds == 3.0
