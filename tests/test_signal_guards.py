import numpy as np

from carelens.vitals import estimate_window


def test_estimate_window_degenerate_inputs_return_none():
    assert estimate_window([], [], 0.7, 3.0, 0.35) is None
    assert estimate_window([1.0], [1.0], 0.7, 3.0, 0.35) is None
    assert estimate_window(np.zeros(30), np.arange(30.0), 0.7, 3.0, 0.35) is None   # ts[-1] <= ts[0]
    assert estimate_window(np.arange(5) / 30, np.arange(5.0), 0.7, 3.0, 0.35) is None  # too few samples
    ts = np.arange(450) / 30.0
    r = estimate_window(ts, np.sin(2 * np.pi * 1.2 * ts), 0.7, 3.0, 0.35)
    assert r and abs(r["value"] - 72) < 1 and r["reported"]
