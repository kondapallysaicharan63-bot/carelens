import json

import pytest

from carelens import fixtures as fx
from carelens.params import PARAMS


def test_generation_deterministic():
    assert fx.generate_all() == fx.generate_all()


def test_check_passes_on_committed_files():
    assert fx.main(["--check"]) == 0


def test_check_fails_on_diff(tmp_path):
    for name, text in fx.generate_all().items():
        (tmp_path / name).write_text(text)
    assert fx.main(["--check", "--dir", str(tmp_path)]) == 0
    p = next(tmp_path.glob("hr_*.json"))
    p.write_text(p.read_text().replace('"truth_bpm":', '"truth_bpm":1', 1))
    assert fx.main(["--check", "--dir", str(tmp_path)]) == 1


def _perturb(x, eps):
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return x + eps
    if isinstance(x, list):
        return [_perturb(v, eps) for v in x]
    return {k: _perturb(v, eps) for k, v in x.items()}


def test_compare_tolerates_tiny_numeric_noise_but_not_gate_flips():
    d = json.loads((fx.FIXTURE_DIR / "hr_green_72bpm.json").read_text())
    assert fx.compare_outputs(d, _perturb(d["expected"], 1e-6)) == []
    bad = json.loads(json.dumps(d["expected"]))
    bad["estimates"][3]["reported"] = not bad["estimates"][3]["reported"]
    assert fx.compare_outputs(d, bad)
    bad = json.loads(json.dumps(d["expected"]))
    bad["estimates"][3]["value"] += 5
    assert fx.compare_outputs(d, bad)
    bad = json.loads(json.dumps(d["expected"]))
    bad["estimates"].pop()
    assert fx.compare_outputs(d, bad)


def test_compare_events_exact_kind_and_count():
    d = json.loads((fx.FIXTURE_DIR / "fall_sudden.json").read_text())
    assert fx.compare_outputs(d, _perturb(d["expected"], 1e-6)) == []
    bad = json.loads(json.dumps(d["expected"]))
    bad["events"][0]["kind"] = "inactivity"
    assert fx.compare_outputs(d, bad)
    bad["events"] = []
    assert fx.compare_outputs(d, bad)


def test_check_tolerates_numeric_noise_in_committed_files(tmp_path):
    for name, text in fx.generate_all().items():
        d = json.loads(text)
        d["expected"] = _perturb(d["expected"], 1e-6)
        (tmp_path / name).write_text(json.dumps(d))
    assert fx.main(["--check", "--dir", str(tmp_path)]) == 0


def test_features_sampled_by_whole_second():
    d = json.loads((fx.FIXTURE_DIR / "fall_sudden.json").read_text())
    ts = [f["t"] for f in d["expected"]["features_1hz"]]
    assert ts == [float(i) for i in range(len(ts))]


def _committed():
    return sorted(p for p in fx.FIXTURE_DIR.glob("*.json") if p.name != fx.PENDING_FILE)


def test_committed_files_present_small_and_schema():
    files = _committed()
    assert len(files) >= 12
    assert sum(p.stat().st_size for p in files) < 3_000_000
    for p in files:
        d = json.loads(p.read_text())
        assert set(d) == {"name", "description", "input_level", "params_version", "input",
                          "expected", "tolerance"}
        assert d["name"] == p.stem and d["params_version"] == PARAMS.version


@pytest.mark.parametrize("path", _committed(), ids=lambda p: p.stem)
def test_python_reproduces_expected(path):
    d = json.loads(path.read_text())
    assert fx.compare_outputs(d, fx.run_fixture(d)) == []


def test_reference_behaviour_sanity():
    by = {p.stem: json.loads(p.read_text()) for p in _committed()}
    for name, bpm in (("hr_green_60bpm", 60), ("hr_green_90bpm", 90), ("hr_green_110bpm", 110)):
        est = by[name]["expected"]["estimates"]
        assert est and all(e["reported"] and abs(e["value"] - bpm) < 1.0 for e in est)
    assert all(e["t"] < 31 for e in by["hr_green_low_light"]["expected"]["estimates"])
    assert [e["kind"] for e in by["fall_sudden"]["expected"]["events"]] == ["fall"]
    assert by["fall_slow_lie_down"]["expected"]["events"] == []
    assert [e["kind"] for e in by["inactivity_sit_still"]["expected"]["events"]] == ["inactivity"]


def test_fixtures_use_real_vitals_code_paths():
    import inspect

    import numpy as np

    from carelens import vitals
    src = inspect.getsource(fx)
    assert "_push" not in src and "_maybe_estimate" not in src and "_reset" not in src
    d = json.loads((fx.FIXTURE_DIR / "hr_green_72bpm.json").read_text())
    inp, hr = d["input"], vitals.HeartRateEstimator()
    # direct call of the pure window function on the last window equals the fixture's last estimate
    last = d["expected"]["estimates"][-1]
    n = len(inp["green"])
    ts = np.arange(n) / inp["fps"]
    m = (ts <= last["t"] + 1e-9) & (ts >= last["t"] - hr.window_s - 1e-9)
    r = vitals.estimate_window(ts[m], np.array(inp["green"])[m], hr.lo, hr.hi, hr.min_quality)
    assert (r["value"], r["quality"], r["reported"]) == (last["value"], last["quality"], last["reported"])
