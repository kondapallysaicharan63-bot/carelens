"""Web session export: schema validator, drift guard against the JSON schema file, accuracy round trip.
Synthetic data only; no real health data."""
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from carelens import accuracy as acc
from carelens import websession as ws

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "shared" / "session.schema.json").read_text())
EXAMPLE = ROOT / "shared" / "session.example.json"
TZ = timezone(timedelta(hours=-8))


def _good():
    return json.loads(EXAMPLE.read_text())


def _session(metric="hr_fingertip", bpm=72.0, n=60, span=10.0, noise=0.5, seed=1):
    """TS-export-shaped session: one estimate per second, `bpm` + noise."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 11, 20, 10, 0, 0, tzinfo=TZ)
    est = []
    for i in range(12, n):
        v = float(bpm + rng.normal(0, noise))
        est.append({"t": float(i), "wall": (t0 + timedelta(seconds=i)).isoformat(timespec="milliseconds"), "metric": metric,
                    "value": round(v, 3), "quality": 0.8, "reported": True, "reason": "ok", "span_s": span})
    s = _good()
    s.update(started_at=t0.isoformat(timespec="milliseconds"), tz_offset_min=-480, estimates=est, events=[])
    return s


def _ref_csv(path, bpm, t0, lo=-60, hi=200, step=2):
    lines = ["timestamp,bpm"] + [f"{(t0 + timedelta(seconds=s)).astimezone(timezone.utc).isoformat()},{bpm}" for s in range(lo, hi, step)]
    path.write_text("\n".join(lines) + "\n")


def test_example_is_valid_and_json_schema_enums_match_validator():
    assert ws.validate(_good()) == []
    d = SCHEMA["$defs"]
    assert SCHEMA["properties"]["schema_version"]["const"] == ws.SCHEMA_VERSION
    assert tuple(d["estimate"]["properties"]["metric"]["enum"]) == ws.METRICS
    assert tuple(d["event"]["properties"]["kind"]["enum"]) == ws.EVENT_KINDS
    assert tuple(SCHEMA["properties"]["modes"]["items"]["enum"]) == ws.MODES
    assert tuple(SCHEMA["properties"]["device"]["properties"]["user_agent_family"]["enum"]) == ws.UA_FAMILIES
    assert set(SCHEMA["required"]) == set(ws._TOP)
    assert set(SCHEMA["properties"]) == set(ws._TOP) | set(ws._OPTIONAL_TOP)
    assert set(d["estimate"]["required"]) == set(ws._EST_REQ)
    assert set(d["event"]["properties"]) == set(ws._EV_ALL)
    assert tuple(d["summary"]["properties"]["kind"]["enum"]) == ws.SUMMARY_KINDS
    assert set(d["summary"]["required"]) == set(ws._SUM_REQ)
    assert set(d["summary"]["properties"]) == set(ws._SUM_ALL)


def test_summaries_are_optional_and_validated():
    s = _good()
    assert "summaries" in s and ws.validate(s) == []
    del s["summaries"]
    assert ws.validate(s) == []   # older exports have none
    for mut in (lambda r: r.update(kind="stress"), lambda r: r.update(rmssd_ms=-1), lambda r: r.update(kept_frac=1.5),
                lambda r: r.update(n_kept=2.5), lambda r: r.update(foo=1), lambda r: r.pop("sdnn_ms"),
                lambda r: r.pop("reason"), lambda r: r.update(reported="yes")):
        s = _good()
        mut(s["summaries"][0])
        assert ws.validate(s), mut
    s = _good()
    s["summaries"][0].update(reported=False, reason="too_short")
    for k in ("rmssd_ms", "sdnn_ms", "mean_hr_bpm"):
        s["summaries"][0].pop(k)
    assert ws.validate(s) == []   # a withheld result carries no numbers


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s.pop("started_at"), "missing 'started_at'"),
    (lambda s: s.update(schema_version=2), "schema_version"),
    (lambda s: s.update(started_at="2026-11-20 10:00:00"), "started_at"),
    (lambda s: s.update(started_at="2026-11-20T10:00:00.000"), "UTC offset"),
    (lambda s: s["device"].update(label="Pixel 7"), "unexpected field 'label'"),
    (lambda s: s["device"].update(user_agent_family="Mozilla/5.0 (X11)"), "user_agent_family"),
    (lambda s: s.update(device_id="abc"), "unexpected field 'device_id'"),
    (lambda s: s.update(video="data:video/mp4;base64,AAA"), "unexpected field 'video'"),
    (lambda s: s.update(modes=["fingertip", "fingertip"]), "modes"),
    (lambda s: s["estimates"][0].update(metric="spo2"), "estimates[0].metric"),
    (lambda s: s["estimates"][0].update(quality=1.5), "estimates[0].quality"),
    (lambda s: s["estimates"][0].update(value="72"), "estimates[0].value"),
    (lambda s: s["estimates"][0].update(reported=1), "estimates[0].reported"),
    (lambda s: s["estimates"][0].update(span_s=0), "estimates[0].span_s"),
    (lambda s: s["estimates"][0].update(image="x"), "unexpected field 'image'"),
    (lambda s: s["events"][0].update(kind="diagnosis"), "events[0].kind"),
    (lambda s: s["events"][0].update(wall="yesterday"), "events[0].wall"),
    (lambda s: s.update(estimates="none"), "estimates: must be an array"),
])
def test_schema_rejection_cases(mutate, needle):
    s = copy.deepcopy(_good())
    mutate(s)
    errs = ws.validate(s)
    assert errs and any(needle in e for e in errs), errs


def test_load_session_errors_are_clear(tmp_path):
    with pytest.raises(ws.SessionError, match="not found"):
        ws.load_session(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ws.SessionError, match="not valid JSON"):
        ws.load_session(tmp_path / "bad.json")
    s = _good()
    s["estimates"][0]["metric"] = "bogus"
    (tmp_path / "s.json").write_text(json.dumps(s))
    with pytest.raises(ws.SessionError, match=r"estimates\[0\]\.metric"):
        ws.load_session(tmp_path / "s.json")
    (tmp_path / "list.json").write_text("[]")
    with pytest.raises(ws.SessionError, match="JSON object"):
        ws.load_session(tmp_path / "list.json")


@pytest.mark.parametrize("metric,flag", [("hr_fingertip", "--hr-ref"), ("hr_face", "--hr-ref"),
                                         ("br_camera", "--br-ref"), ("br_sound", "--br-ref")])
def test_round_trip_web_session_to_accuracy(tmp_path, metric, flag):
    bpm = 72.0 if metric.startswith("hr_") else 15.0
    s = _session(metric, bpm=bpm, span=10.0 if metric.startswith("hr_") else 30.0)
    (tmp_path / "session.json").write_text(json.dumps(s))
    t0 = datetime.fromisoformat(s["started_at"])
    ref = tmp_path / "ref.csv"
    _ref_csv(ref, bpm, t0)
    if flag == "--br-ref":
        ref.write_text(ref.read_text().replace("timestamp,bpm", "timestamp,breaths_per_min"))
    out = tmp_path / "out"
    rc = acc.main(["--web-session", str(tmp_path / "session.json"), "--metric", metric, flag, str(ref), "--out", str(out)])
    assert rc == 0
    m = json.loads((out / "metrics.json").read_text())
    key = "heart_rate" if metric.startswith("hr_") else "breathing_rate"
    other = "breathing_rate" if key == "heart_rate" else "heart_rate"
    assert m[other] is None
    summ = m[key]["summary"]
    assert summ["n"] == 48 and summ["mae"] < 1.0 and summ["coverage"] == 1.0
    assert m[key]["window_s"] == (10.0 if metric.startswith("hr_") else 30.0)
    frag = (out / "report_fragment.md").read_text()
    assert "Mean absolute error" in frag and f"web-{metric}" in frag


def test_window_centring_uses_span_s_and_started_at_offset(tmp_path):
    # one estimate at t=60 with span 30 s is compared at wall = start + 45 s; the reference is 90 at +45 s and 50 at +60 s
    s = _session(n=14)
    s["estimates"] = [{"t": 60.0, "wall": s["started_at"], "metric": "hr_fingertip", "value": 90.0, "quality": 0.9,
                       "reported": True, "reason": "ok", "span_s": 30.0}]
    (tmp_path / "s.json").write_text(json.dumps(s))
    t0 = datetime.fromisoformat(s["started_at"])
    (tmp_path / "r.csv").write_text("timestamp,bpm\n%s,90\n%s,50\n" % ((t0 + timedelta(seconds=45)).isoformat(), (t0 + timedelta(seconds=120)).isoformat()))
    assert acc.main(["--web-session", str(tmp_path / "s.json"), "--metric", "hr_fingertip", "--hr-ref", str(tmp_path / "r.csv"),
                     "--tolerance-s", "5", "--out", str(tmp_path / "o")]) == 0
    row = json.loads((tmp_path / "o" / "metrics.json").read_text())["heart_rate"]["windows"][0]
    assert row["ref"] == 90.0 and row["error"] == 0.0


def test_null_rows_kept_and_events_forwarded():
    s = _good()
    run = ws.to_run(s, "hr_fingertip")
    wins = run["results"]["heart_rate"]["windows"]
    assert len(wins) == sum(1 for e in s["estimates"] if e["metric"] == "hr_fingertip")   # null rows kept
    assert any(w["value"] is None for w in wins)
    assert run["results"]["events"] == [{"kind": "fall", "t_onset": 5.4}]
    assert run["tzinfo"].utcoffset(None) == timedelta(hours=-8)


def test_coverage_counts_withheld_rows_but_scores_only_reported(tmp_path):
    s = _session(n=12 + 10, noise=0.0)          # 10 reported rows at 72
    ests = s["estimates"]
    for e in ests[:4]:                           # 4 withheld rows (no value)
        e.update(value=None, reported=False, quality=0.0, reason="settling")
    for e in ests[4:6]:                          # 2 shown-but-gated rows (value, not reported)
        e.update(value=90.0, reported=False, quality=0.1, reason="low_signal")
    (tmp_path / "s.json").write_text(json.dumps(s))
    ref = tmp_path / "r.csv"
    _ref_csv(ref, 72.0, datetime.fromisoformat(s["started_at"]))
    assert acc.main(["--web-session", str(tmp_path / "s.json"), "--metric", "hr_fingertip", "--hr-ref", str(ref),
                     "--out", str(tmp_path / "o")]) == 0
    m = json.loads((tmp_path / "o" / "metrics.json").read_text())["heart_rate"]["summary"]
    assert (m["n_windows"], m["n_reported"], m["n_matched"], m["n"], m["n_all_matched"]) == (10, 4, 6, 4, 6)
    assert m["coverage"] == pytest.approx(0.4) and m["mae"] == pytest.approx(0.0)
    assert m["mae_all_matched"] == pytest.approx(36.0 / 6)          # two rows 18 bpm off, four exact
    assert "n/a" in (tmp_path / "o" / "report_fragment.md").read_text()


def test_validator_parity_integers_and_bool_and_reported_value():
    def with_(**kw):
        d = _good(); d.update(kw); return d
    assert ws.validate(json.loads(json.dumps(_good()).replace('"schema_version": 1', '"schema_version": 1.0'))) == []
    assert ws.validate(with_(params_version=7.0, tz_offset_min=-480.0)) == []
    assert ws.validate(with_(schema_version=True))
    assert ws.validate(with_(params_version=True))
    assert ws.validate(with_(tz_offset_min=False))
    assert ws.validate(with_(params_version=7.5)) and ws.validate(with_(schema_version=1.5))
    assert ws.validate(with_(truncated=True)) == [] and ws.validate(with_(truncated="yes"))
    bad = _good(); bad["estimates"][0].update(reported=True, value=None)
    assert any("reported rows need a value" in e for e in ws.validate(bad))


def test_iso_parsing_normalises_fractions_for_py310():
    assert ws.normalize_iso("2026-11-20T10:00:00.25-08:00") == "2026-11-20T10:00:00.250000-08:00"
    assert ws.normalize_iso("2026-11-20T10:00:00.1234567Z") == "2026-11-20T10:00:00.123456+00:00"
    assert ws.normalize_iso("2026-11-20T10:00:00Z") == "2026-11-20T10:00:00+00:00"
    assert ws.parse_iso("2026-11-20T10:00:00.25-08:00").microsecond == 250000
    s = _good(); s["started_at"] = "2026-11-20T10:00:00.25-08:00"
    assert ws.validate(s) == []
    assert acc._parse_dt("2026-11-20T10:00:00.25Z") == datetime(2026, 11, 20, 10, 0, 0, 250000, tzinfo=timezone.utc).timestamp()


def test_ignored_reference_flag_is_announced(tmp_path, capsys):
    s = _session()
    (tmp_path / "s.json").write_text(json.dumps(s))
    ref = tmp_path / "r.csv"
    _ref_csv(ref, 72.0, datetime.fromisoformat(s["started_at"]))
    assert acc.main(["--web-session", str(tmp_path / "s.json"), "--metric", "hr_fingertip", "--hr-ref", str(ref),
                     "--br-ref", str(ref), "--out", str(tmp_path / "o")]) == 0
    assert "--br-ref is ignored" in capsys.readouterr().err


def test_cli_argument_errors(tmp_path, capsys):
    (tmp_path / "s.json").write_text(json.dumps(_good()))
    r = tmp_path / "r.csv"
    r.write_text("timestamp,bpm\n2026-11-20T18:00:00Z,72\n")
    base = ["--web-session", str(tmp_path / "s.json"), "--out", str(tmp_path / "o")]
    assert acc.main(base + ["--hr-ref", str(r)]) == 2 and "--metric" in capsys.readouterr().err
    assert acc.main(base + ["--metric", "br_sound", "--hr-ref", str(r)]) == 2 and "--br-ref" in capsys.readouterr().err
    assert acc.main(base + ["--run", "x.json", "--metric", "hr_face", "--hr-ref", str(r)]) == 2
    assert acc.main(["--run", "x.json", "--metric", "hr_face"]) == 2
    bad = _good()
    bad["estimates"][0]["quality"] = 7
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert acc.main(["--web-session", str(tmp_path / "bad.json"), "--metric", "hr_fingertip", "--hr-ref", str(r), "--out", str(tmp_path / "o")]) == 2
    assert "estimates[0].quality" in capsys.readouterr().err


def test_mobility_summaries_validated():
    base = {"t": 40.0, "wall": "2026-11-20T10:00:40.000-08:00", "reported": True, "reason": "ok"}
    good = [
        {"kind": "chair_stand_30s", "reps": 12, "rep_times_s": [2.1, 4.2], "quality": 0.97},
        {"kind": "sit_to_stand_5x", "time_s": 9.8, "rep_times_s": [2.0] * 5, "quality": 1.0},
        {"kind": "tug", "time_s": 11.3, "phases_s": {"rise": 1.7, "walk_out": 3.4, "turn_1": 1.3, "walk_back": 3.2, "turn_2": 1.3, "sit": 0.4}},
        {"kind": "walking_speed", "speed_m_s": 1.02, "cadence_spm": 108.0, "duration_s": 3.7, "quality": 0.9},
        {"kind": "tug", "reported": False, "reason": "no_turn", "quality": 0.9},   # withheld: no numbers
    ]
    for g in good:
        s = _good()
        s["summaries"] = [{**base, **g}]
        assert ws.validate(s) == [], g
    bad = [
        {"kind": "chair_stand_30s"},                                  # reported without reps
        {"kind": "tug", "time_s": -1}, {"kind": "walking_speed"}, {"kind": "sit_to_stand_5x", "time_s": 1, "rep_times_s": [-1]},
        {"kind": "chair_stand_30s", "reps": 2.5}, {"kind": "walking_speed", "speed_m_s": 1, "cadence_spm": 0},
        {"kind": "tug", "time_s": 1, "phases_s": {"rise": 1, "flight": 2}}, {"kind": "tug", "time_s": 1, "quality": 2},
        {"kind": "chair_stand_30s", "reps": 3, "rep_times_s": [1.0] * 101}, {"kind": "tug", "time_s": 1, "height_cm": 170},
    ]
    for b in bad:
        s = _good()
        s["summaries"] = [{**base, **b}]
        assert ws.validate(s), b
