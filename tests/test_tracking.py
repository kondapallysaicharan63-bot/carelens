"""Presence tracker: dropout filling, hysteresis, smoothing, reset, and safety vs. events."""
from types import SimpleNamespace

import numpy as np

from carelens import pipeline
from carelens.events import GaitAnalyzer
from carelens.pipeline import Config, run
from carelens.pose import ScriptedPose
from carelens.vitals import BreathingRateEstimator
from carelens.synth import ASPECT, Scenario, Segment, _skeleton
from carelens.tracking import PresenceTracker, TrackingConfig

FPS = 15.0


def pose(x=0.5, noise=0.0, rng=None):
    lm = _skeleton("stand", x * ASPECT).copy()
    lm[:, 0] /= ASPECT
    if noise:
        lm[:, :2] += rng.normal(0, noise, (33, 2))
    return lm


def feed(tr, seq, fps=FPS):
    return [tr.update(i / fps, lm) for i, lm in enumerate(seq)]


def test_enter_after_short_span_and_single_frame_ignored():
    out = feed(PresenceTracker(), [pose()] * 5)
    assert [o[0] is not None for o in out] == [False, False, True, True, True]
    out = feed(PresenceTracker(), [pose(), None, None, pose(), None, None])
    assert all(o[0] is None for o in out)   # spurious single detections never create presence


def test_short_dropout_held_long_dropout_absent():
    n_hold = int(0.5 * FPS)  # 7 frames at 15 fps stay within hold_s
    out = feed(PresenceTracker(), [pose()] * 6 + [None] * (n_hold + 5) + [pose()] * 3)
    held = [o[1] for o in out[6:6 + n_hold]]
    assert all(held) and all(o[0] is not None for o in out[6:6 + n_hold])
    assert all(o[0] is None and not o[1] for o in out[6 + n_hold:6 + n_hold + 5])
    assert out[-2][0] is None and out[-1][0] is not None   # re-entry needs enter_frames


def test_one_euro_reduces_jitter_and_tracks_ramp():
    rng = np.random.default_rng(1)
    raw = [pose(noise=0.004, rng=rng) for _ in range(150)]
    out = [o[0] for o in feed(PresenceTracker(), raw) if o[0] is not None]
    r = np.array(raw[-len(out):])[:, :, :2]
    assert np.var(np.array(out)[:, :, :2] - pose()[:, :2], axis=0).mean() < \
        0.3 * np.var(r - pose()[:, :2], axis=0).mean()
    # ramp: 0.2 frame-widths/s; lag stays bounded
    ramp = [pose(x=0.2 + 0.2 * i / FPS) for i in range(60)]
    out = [o[0] for o in feed(PresenceTracker(), ramp)]
    lag = ramp[-1][P_HIP, 0] - out[-1][P_HIP, 0]
    assert 0 <= lag < 0.05


P_HIP = 23


def test_jump_passes_raw_without_losing_presence():
    out = feed(PresenceTracker(), [pose(0.2)] * 5 + [pose(0.8)] * 3)
    assert all(o[0] is not None for o in out[2:])          # no absence, no re-entry
    np.testing.assert_allclose(out[5][0][P_HIP, 0], pose(0.8)[P_HIP, 0], atol=1e-4)  # no smear


def test_reacquire_elsewhere_after_hold_resets_filter_only():
    out = feed(PresenceTracker(), [pose(0.2)] * 5 + [None] * 3 + [pose(0.8)] * 2)
    assert out[6][1] and out[8][0] is not None and not out[8][1]
    np.testing.assert_allclose(out[8][0][P_HIP, 0], pose(0.8)[P_HIP, 0], atol=1e-4)


def test_enter_debounce_is_in_seconds():
    for fps in (3.0, 15.0, 30.0):
        out = feed(PresenceTracker(), [pose()] * 12, fps)
        first = next(i for i, o in enumerate(out) if o[0] is not None)
        assert 1 <= first and first / fps <= 0.4


ASP = ASPECT


def _run(fn, fps, tmp_path, tracking, secs, **cfg):
    class Src:
        def __init__(self):
            self.fps = fps

        def __iter__(self):
            for i in range(int(secs * fps)):
                yield SimpleNamespace(image=np.full((48, 64, 3), 100, np.uint8), t=i / fps, index=i)
    return run(Src(), ScriptedPose(fn), tmp_path,
               Config(tracking=TrackingConfig() if tracking else None, **cfg), clip_name="t")


def _fall_scn():
    return Scenario([Segment("still", 4, posture="stand"),
                     Segment("transition", 0.3, posture="stand", to="lie"),
                     Segment("still", 8, posture="lie")])


def test_fast_fall_detected_at_all_frame_rates(tmp_path):
    for fps in (3.0, 4.0, 6.0, 15.0):
        on, off = (_run(_fall_scn().pose_landmarks, fps, tmp_path, tr, 12.3, fall_still_seconds=5)
                   ["events"] for tr in (True, False))
        assert len(off) == 1 and len(on) == 1, fps
        assert abs(on[0]["t_onset"] - off[0]["t_onset"]) <= 0.5, fps


def test_slow_lie_down_with_dropout_is_not_a_fall(tmp_path):
    scn = Scenario([Segment("still", 4, posture="stand"),
                    Segment("transition", 3.0, posture="stand", to="lie"),   # peak rate just under 40 deg/s
                    Segment("still", 8, posture="lie")])
    fn = lambda t: None if 5.05 <= t < 5.55 else scn.pose_landmarks(t)
    r = _run(fn, 15.0, tmp_path, True, 14.6, fall_still_seconds=5)
    assert r["events"] == []


def test_dropout_mid_walk_adds_no_gait_samples(tmp_path, monkeypatch):
    seen = []

    class Spy(GaitAnalyzer):
        def update(self, s):
            seen.append(s.t)
            super().update(s)
    monkeypatch.setattr(pipeline, "GaitAnalyzer", Spy)
    scn = Scenario([Segment("walk", 8, x0=0.1, x1=0.9)])
    _run(lambda t: None if 4.0 <= t < 4.4 else scn.pose_landmarks(t), 15.0, tmp_path, True, 8)
    assert len(seen) > 90 and not [t for t in seen if 4.0 + 1 / 15 < t < 4.4]


def test_breathing_estimator_reset_after_held_stretch(tmp_path, monkeypatch):
    resets = []

    class Spy(BreathingRateEstimator):
        def _reset(self):
            resets.append(len(self._buf))
            super()._reset()
    monkeypatch.setattr(pipeline, "BreathingRateEstimator", Spy)
    scn = Scenario([Segment("still", 20, posture="sit")])
    _run(lambda t: None if 10.0 <= t < 10.4 else scn.pose_landmarks(t), 15.0, tmp_path, True, 20)
    assert max(resets) > 50   # a populated window was dropped when the held stretch began


def test_held_frames_do_not_fake_stillness(tmp_path):
    # Steady drift (constant nonzero motion) with a 0.4 s dropout covering the t=5.0 sample.
    fn = lambda t: None if 4.7 <= t < 5.1 else pose(0.2 + 0.06 * t)
    r = _run(fn, 10.0, tmp_path, True, 12)
    m = {p["t"]: p["motion"] for p in r["posture_track"]}
    assert m[4.0] > 0.05 and abs(m[5.0] - m[4.0]) < 0.1 * m[4.0]
