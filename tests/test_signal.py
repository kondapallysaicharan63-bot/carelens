import numpy as np
import pytest

from carelens.vitals import bandpass, dominant_frequency


@pytest.mark.parametrize("bpm", [48, 60, 72, 95, 130])
def test_heart_band_recovers_known_rate(bpm):
    fs, dur = 15.0, 15.0
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(0)
    x = 0.5 * np.sin(2 * np.pi * bpm / 60 * t) + rng.normal(0, 0.5, t.size) + 0.02 * t
    f, q = dominant_frequency(bandpass(x, fs, 0.7, 3.0), fs, 0.7, 3.0)
    assert abs(f * 60 - bpm) < 3
    assert q > 0.2


@pytest.mark.parametrize("brpm", [8, 12, 15, 20, 26, 36])
def test_breathing_band_recovers_known_rate(brpm):
    # band from params (0.1-0.7 Hz since params v6; it was 0.1-0.5 Hz, which could not reach 36 breaths/min)
    from carelens.params import PARAMS
    lo, hi = PARAMS.vitals.breathing.band_lo_hz, PARAMS.vitals.breathing.band_hi_hz
    fs, dur = 15.0, 30.0
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(1)
    x = np.sin(2 * np.pi * brpm / 60 * t) + rng.normal(0, 0.4, t.size)
    f, q = dominant_frequency(bandpass(x, fs, lo, hi), fs, lo, hi)
    assert abs(f * 60 - brpm) < 1.5
    assert q > 0.3


def test_pure_noise_has_low_quality():
    rng = np.random.default_rng(2)
    fs = 15.0
    qs = []
    for _ in range(20):
        x = rng.normal(0, 1, int(15 * fs))
        _, q = dominant_frequency(bandpass(x, fs, 0.7, 3.0), fs, 0.7, 3.0)
        qs.append(q)
    assert np.median(qs) < 0.3


def test_out_of_band_signal_is_rejected():
    fs = 15.0
    t = np.arange(0, 15, 1 / fs)
    x = np.sin(2 * np.pi * 5.0 * t)     # 300 bpm: outside the heart band
    xf = bandpass(x, fs, 0.7, 3.0)
    assert np.std(xf) < 0.15 * np.std(x)  # >85% rejected; filtfilt edge ringing is real
