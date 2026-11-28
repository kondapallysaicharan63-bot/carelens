"""Accuracy harness tests - synthetic data only; no real health data."""
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from carelens import accuracy as acc
from carelens.params import PARAMS
from carelens.pipeline import Config, run
from carelens.store import Store
from carelens.synth import Scenario, Segment, render

T0 = datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone(timedelta(hours=-8)))
T0_TS = T0.timestamp()

XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!ELEMENT HealthData (Record)*>
<!ATTLIST Record type CDATA #REQUIRED>
]>
<HealthData locale="en_US">
 <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="W" unit="count/min" startDate="2026-11-20 10:01:00 -0800" endDate="2026-11-20 10:01:00 -0800" value="70"/>
 <Record type="HKQuantityTypeIdentifierStepCount" startDate="2026-11-20 10:01:00 -0800" value="999"/>
 <Record type="HKQuantityTypeIdentifierHeartRate" startDate="2026-11-20 19:01:30 +0100" value="74"/>
 <Workout startDate="2026-11-20 10:00:00 -0800"/>
</HealthData>
"""


def _w(t, v, q=0.9, rep=True):
    return {"t": t, "value": v, "quality": q, "reported": rep}


def test_apple_xml_parse_filters_and_zones(tmp_path):
    p = tmp_path / "export.xml"
    p.write_text(XML)
    t, v = acc.parse_apple_health(p)
    assert list(v) == [70.0, 74.0]          # steps ignored
    # 10:01:00 -0800 == 19:01:00 +0100 -> 30 s apart
    assert t[1] - t[0] == pytest.approx(30.0)
    assert t[0] == T0_TS + 60


def test_csv_parse(tmp_path):
    p = tmp_path / "hr.csv"
    p.write_text("timestamp,bpm\n2026-11-20T18:00:30+00:00,71\n2026-11-20T18:00:00Z,70\n")
    t, v = acc.load_reference(p, "bpm")
    assert list(v) == [70.0, 71.0] and t[0] == T0_TS    # sorted, offset-aware


def test_alignment_tolerance_and_offset():
    ref_t = np.array([T0_TS + 100.0, T0_TS + 120.0])
    ref_v = np.array([60.0, 70.0])
    rows = acc.align([_w(110, 66)], T0_TS, ref_t, ref_v, tolerance_s=30)
    assert rows[0]["ref"] == 65.0 and rows[0]["n_ref"] == 2 and rows[0]["error"] == 1.0
    assert rows[0]["ref_sd"] == pytest.approx(np.std([60, 70], ddof=1))
    rows = acc.align([_w(10, 66)], T0_TS, ref_t, ref_v, tolerance_s=30)
    assert rows[0]["ref"] is None and rows[0]["error"] is None
    rows = acc.align([_w(10, 66)], T0_TS, ref_t, ref_v, tolerance_s=5, offset_s=90)
    assert rows[0]["ref"] == 60.0


def _rows(errs, rep=True):
    return [{"t": i, "camera": 70 + e, "quality": .9, "reported": rep, "ref": 70.0, "n_ref": 1,
             "ref_sd": 0, "error": float(e)} for i, e in enumerate(errs)]


def test_stats_hand_computed_4_windows():
    s = acc.summarize(_rows([1, -3, 2, 4]))
    assert s["n"] == 4 and s["mae"] == pytest.approx(2.5)
    assert s["rmse"] == pytest.approx(np.sqrt(30 / 4))
    assert s["bias"] == pytest.approx(1.0)
    sd = np.sqrt(((0) ** 2 + 16 + 1 + 9) / 3)
    assert s["loa_low"] == pytest.approx(1 - 1.96 * sd) and s["loa_high"] == pytest.approx(1 + 1.96 * sd)
    assert s["pearson_r"] is None and s["mae_ci95"] is None     # n < 5
    assert s["coverage"] == 1.0


def test_pearson_and_bootstrap_gated_by_n():
    rows = _rows([1, -1, 2, -2, 3])
    for i, r in enumerate(rows):      # give the reference some spread
        r["ref"] = 60.0 + 5 * i
        r["camera"] = r["ref"] + r["error"]
    s = acc.summarize(rows)
    assert s["pearson_r"] > 0.9 and len(s["mae_ci95"]) == 2
    assert s["mae_ci95"][0] <= s["mae"] <= s["mae_ci95"][1]
    assert acc.summarize(rows)["mae_ci95"] == s["mae_ci95"]     # seeded: deterministic
    assert acc.summarize(_rows([1, 2]))["loa_low"] is None      # n < 3


def test_coverage_and_unreported_excluded_from_main_mae():
    rows = _rows([1, 1]) + _rows([10], rep=False) + [
        {"t": 9, "camera": 70, "quality": .1, "reported": False, "ref": None, "n_ref": 0,
         "ref_sd": None, "error": None}]
    s = acc.summarize(rows)
    assert s["n"] == 2 and s["mae"] == 1.0
    assert s["mae_all_matched"] == pytest.approx(4.0) and s["n_all_matched"] == 3
    assert s["coverage"] == pytest.approx(2 / 4)


def test_no_match_nulls_and_offset_hint():
    ts = [0, 300, 600, 900, 1200]
    wins = [_w(t, 72) for t in ts]
    ref_t = T0_TS + np.array(ts) + 120.0       # reference clock is 120 s ahead
    ref_v = np.full(5, 72.0)
    run_ = {"results": {"heart_rate": {"windows": wins}, "events": [], "duration_s": 1300},
            "start_ts": T0_TS, "name": "c"}
    m = acc.evaluate(run_, hr_ref=(ref_t, ref_v), tolerance_s=30)
    s = m["heart_rate"]["summary"]
    assert s["n"] == 0 and s["mae"] is None and s["bias"] is None
    assert "No matched windows" in s["note"]
    assert m["heart_rate"]["suggested_offset_s"] == 120
    assert "No matched windows — check clock offset/time zone" in acc.build_fragment(m)
    m2 = acc.evaluate(run_, hr_ref=(ref_t, ref_v), tolerance_s=30, offset_s=120)
    assert m2["heart_rate"]["summary"]["mae"] == 0.0           # offset applied, never auto-applied


def test_events_hit_miss_false_alarm_none_label():
    ev = [{"kind": "fall", "t_onset": 52.0}, {"kind": "fall", "t_onset": 500.0},
          {"kind": "inactivity", "t_onset": 10.0}]
    lab = [{"clip": "c", "kind": "fall", "onset_s": 50.0}, {"clip": "c", "kind": "fall", "onset_s": 300.0},
           {"clip": "other", "kind": "fall", "onset_s": 1.0}]
    r = acc.score_events(ev, lab, "c", 1800.0, tol_s=10)
    assert (r["hits"], r["misses"], r["false_alarms"]) == (1, 1, 1)
    assert r["sensitivity"] == 0.5 and r["ppv"] == 0.5
    assert r["false_alarms_per_hour"] == pytest.approx(2.0)
    assert r["per_label"][0]["onset_error_s"] == pytest.approx(2.0)
    none = acc.score_events([{"kind": "fall", "t_onset": 5.0}],
                            [{"clip": "c", "kind": "none", "onset_s": None}], "c", 3600.0)
    assert none["false_alarms"] == 1 and none["sensitivity"] is None and none["false_alarms_per_hour"] == 1.0
    quiet = acc.score_events([], [{"clip": "c", "kind": "none", "onset_s": None}], "c", 3600.0)
    assert quiet["false_alarms"] == 0 and quiet["ppv"] is None


def _results(start_iso, tz=None):
    return {"banner": "b", "data_label": "simulated", "source": "s", "start_time": start_iso,
            "duration_s": 120.0, "config": {}, "events": [{"kind": "fall", "t_onset": 40.0,
            "t_confirmed": 43.0, "message": "check-in suggested", "details": {}, "clip": None}],
            "timeline_by_hour": {}, "posture_track": [], "gait_segments": [],
            "heart_rate": {"unit": "bpm", "summary": {"median": 72, "n_reported": 2},
                           "windows": [_w(60, 72), _w(90, 73)]},
            "breathing_rate": {"unit": "breaths/min", "summary": {"median": None, "n_reported": 0},
                               "windows": []}}


def test_db_input(tmp_path):
    with Store(tmp_path / "c.db") as st:
        rid = st.save_run(_results("2026-11-20T10:00:00"), camera_id="x", name="clipA")
    r = acc.load_run(f"db:{tmp_path / 'c.db'}#{rid}")
    assert r["name"] == "clipA" and len(r["results"]["heart_rate"]["windows"]) == 2
    # naive local start interpreted with the run's stored tz_offset_min
    assert r["start_ts"] == pytest.approx(
        datetime.fromisoformat("2026-11-20T10:00:00").replace(
            tzinfo=timezone(timedelta(minutes=_tz_min(tmp_path / "c.db", rid)))).timestamp())
    with pytest.raises(ValueError):
        acc.load_run(f"db:{tmp_path / 'c.db'}#999")


def _tz_min(path, rid):
    with Store(path) as st:
        return st.db.execute("SELECT tz_offset_min FROM runs WHERE id=?", (rid,)).fetchone()[0]


def test_cli_writes_both_files(tmp_path, capsys):
    (tmp_path / "clip_results.json").write_text(json.dumps(_results(T0.isoformat())))
    (tmp_path / "hr.csv").write_text("timestamp,bpm\n" + "".join(
        f"{(T0 + timedelta(seconds=s)).isoformat()},72\n" for s in range(40, 120, 10)))
    (tmp_path / "labels.csv").write_text("clip,kind,onset_s\nclip,fall,41\n")
    out = tmp_path / "acc"
    assert acc.main(["--run", str(tmp_path / "clip_results.json"), "--hr-ref", str(tmp_path / "hr.csv"),
                     "--labels", str(tmp_path / "labels.csv"), "--out", str(out)]) == 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["heart_rate"]["summary"]["n"] == 2 and m["events"]["hits"] == 1
    frag = (out / "report_fragment.md").read_text()
    assert "## 3. Heart rate" in frag and "not a diagnosis" in frag and "## 2. Fall detection" in frag
    assert "MAE" in capsys.readouterr().out
    assert acc.main(["--run", str(tmp_path / "missing.json"), "--out", str(out)]) == 2


def test_fragment_truncates_rows():
    wins = [_w(t, 72) for t in range(300)]
    ref_t = T0_TS + np.arange(0, 300.0)
    m = acc.evaluate({"results": {"heart_rate": {"windows": wins}, "duration_s": 300},
                      "start_ts": T0_TS, "name": "c"}, hr_ref=(ref_t, np.full(300, 72.0)))
    frag = acc.build_fragment(m)
    assert "Showing first 200 of 300" in frag
    assert len(m["heart_rate"]["windows"]) == 300


def test_streaming_parse_large_file(tmp_path):
    p = tmp_path / "big.xml"
    with open(p, "w") as f:
        f.write('<?xml version="1.0"?>\n<!DOCTYPE HealthData [<!ELEMENT HealthData ANY>]>\n<HealthData>\n')
        for i in range(50_000):
            typ = acc.HR_TYPE if i % 2 else "HKQuantityTypeIdentifierStepCount"
            f.write(f'<Record type="{typ}" startDate="2026-11-20 10:00:{i % 60:02d} -0800" value="{60 + i % 40}"/>\n')
        f.write("</HealthData>\n")
    t0 = time.time()
    t, v = acc.parse_apple_health(p)
    assert len(v) == 25_000 and time.time() - t0 < 10


def test_external_entities_not_resolved(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    p = tmp_path / "x.xml"
    p.write_text(f'<?xml version="1.0"?><!DOCTYPE H [<!ENTITY x SYSTEM "file://{secret}">]>'
                 f'<H><Record type="{acc.HR_TYPE}" startDate="2026-11-20 10:00:00 -0800" value="60" '
                 f'sourceName="&x;"/></H>')
    with pytest.raises(Exception):      # stdlib expat refuses to resolve; must not leak the file
        acc.parse_apple_health(p)


def test_e2e_synthetic_alignment(tmp_path):
    """Synthetic-only check of the alignment math, NOT an accuracy claim."""
    scn = Scenario([Segment("still", 35, posture="sit")], hr_bpm=72)
    render(scn, tmp_path / "s.mp4")
    start = datetime(2026, 11, 20, 10, 0, 0)
    store = Store(tmp_path / "c.db")
    res = run(tmp_path / "s.mp4", scn.provider(), tmp_path, Config(data_label="simulated"),
              clip_name="s", store=store, start_time=start)
    store.close()
    tz = timezone(timedelta(hours=-8))
    s_ts = start.replace(tzinfo=tz).timestamp()
    (tmp_path / "hr.csv").write_text("timestamp,bpm\n" + "".join(
        f"{datetime.fromtimestamp(s_ts + s, tz).isoformat()},72\n" for s in range(0, 60, 5)))
    out = tmp_path / "acc"
    assert acc.main(["--run", str(tmp_path / "s_results.json"), "--tz", "-0800",
                     "--hr-ref", str(tmp_path / "hr.csv"), "--out", str(out)]) == 0
    s = json.loads((out / "metrics.json").read_text())["heart_rate"]["summary"]
    assert s["n"] > 0 and s["coverage"] > 0 and s["mae"] < 3


# ---- review fixes ----
def test_alignment_uses_window_centre():
    ref_t = np.array([T0_TS + 100.0])          # only a reference sample near t=100
    ref_v = np.array([60.0])
    w = [_w(130, 61)]                           # window ends at 130, 30 s long -> centre 115
    assert acc.align(w, T0_TS, ref_t, ref_v, tolerance_s=20, window_s=30)[0]["ref"] == 60.0
    assert acc.align(w, T0_TS, ref_t, ref_v, tolerance_s=20, window_s=0)[0]["ref"] is None
    m = acc.evaluate({"results": {"heart_rate": {"window_s": 30, "windows": w}, "duration_s": 200},
                      "start_ts": T0_TS, "name": "c"}, hr_ref=(ref_t, ref_v), tolerance_s=20)
    assert m["heart_rate"]["summary"]["n"] == 1 and m["heart_rate"]["window_s"] == 30.0


def test_pipeline_records_window_s(tmp_path):
    scn = Scenario([Segment("still", 3, posture="sit")], hr_bpm=72)
    render(scn, tmp_path / "s.mp4")
    r = run(tmp_path / "s.mp4", scn.provider(), tmp_path, Config(), clip_name="s")
    assert r["heart_rate"]["window_s"] == PARAMS.rppg.window_s and r["breathing_rate"]["window_s"] == 30.0


def test_suggest_offset_large_reference_fast_and_small():
    import tracemalloc
    ref_t = T0_TS - 1e5 + np.arange(2_000_000) * 0.1     # 2M samples
    wins = [_w(t, 72) for t in range(100, 1000, 100)]
    tracemalloc.start()
    t0 = time.time()
    off = acc.suggest_offset(wins, T0_TS, ref_t, 5.0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert off is not None and time.time() - t0 < 5 and peak < 50e6   # dense ref: any offset matches


def test_naive_csv_uses_run_zone(tmp_path):
    p = tmp_path / "hr.csv"
    p.write_text("timestamp,bpm\n2026-11-20T10:01:00,70\n")
    pst = timezone(timedelta(hours=-8))
    t, _ = acc.load_reference(p, "bpm", pst)
    assert t[0] == T0_TS + 60
    (tmp_path / "r_results.json").write_text(json.dumps(_results("2026-11-20T10:00:00")))
    out = tmp_path / "o"
    assert acc.main(["--run", str(tmp_path / "r_results.json"), "--tz", "-08:00",
                     "--hr-ref", str(p), "--out", str(out)]) == 0
    assert json.loads((out / "metrics.json").read_text())["heart_rate"]["summary"]["n_matched"] >= 1


def test_tz_argument_forms():
    assert acc._fix_argv(["--tz", "-08:00", "--x"]) == ["--tz=-08:00", "--x"]
    assert acc._fix_argv(["--tz", "UTC"]) == ["--tz", "UTC"]
    assert acc._fix_argv(["--tz=-0800"]) == ["--tz=-0800"]
    assert acc._tz_from("-08:00").utcoffset(None) == timedelta(hours=-8)


def test_missing_db_is_clean_error_and_not_created(tmp_path, capsys):
    p = tmp_path / "typo.db"
    assert acc.main(["--run", f"db:{p}#1", "--out", str(tmp_path / "o")]) == 2
    assert not p.exists() and "error:" in capsys.readouterr().err
    bad = tmp_path / "bad.db"
    bad.write_text("not a database")
    assert acc.main(["--run", f"db:{bad}#1", "--out", str(tmp_path / "o")]) == 2


def test_labels_for_other_clip_give_null_not_false_alarms():
    ev = [{"kind": "fall", "t_onset": 5.0}]
    r = acc.score_events(ev, [{"clip": "other", "kind": "fall", "onset_s": 5.0}], "c", 3600)
    assert r["note"] == "No labels for clip c" and r["false_alarms"] is None and r["sensitivity"] is None
    frag = acc.build_fragment({"header": {"reference": "x", "tolerance_s": 30, "offset_s": 0, "run": "c",
                                          "recording_hours": 1}, "events": r,
                               "heart_rate": None, "breathing_rate": None})
    assert "No labels for clip c" in frag


def test_cli_label_clip_explicit(tmp_path):
    (tmp_path / "clip_results.json").write_text(json.dumps(_results(T0.isoformat())))
    (tmp_path / "l.csv").write_text("clip,kind,onset_s\nrenamed,fall,41\n")
    out = tmp_path / "o"
    args = ["--run", str(tmp_path / "clip_results.json"), "--labels", str(tmp_path / "l.csv"), "--out", str(out)]
    assert acc.main(args) == 0
    assert json.loads((out / "metrics.json").read_text())["events"]["hits"] is None
    assert acc.main(args + ["--clip", "renamed"]) == 0
    assert json.loads((out / "metrics.json").read_text())["events"]["hits"] == 1


def test_events_optimal_one_to_one_matching():
    ev = [{"kind": "fall", "t_onset": 58.0}, {"kind": "fall", "t_onset": 45.0}]
    lab = [{"clip": "c", "kind": "fall", "onset_s": 50.0}, {"clip": "c", "kind": "fall", "onset_s": 55.0}]
    r = acc.score_events(ev, lab, "c", 600, tol_s=10)
    assert r["hits"] == 2 and r["false_alarms"] == 0
    assert [p["onset_error_s"] for p in r["per_label"]] == [-5.0, 3.0]


def test_empty_windows_and_single_note():
    run_ = {"results": {"heart_rate": {"windows": []}, "duration_s": 10}, "start_ts": T0_TS, "name": "c"}
    m = acc.evaluate(run_, hr_ref=(np.array([T0_TS]), np.array([70.0])))
    frag = acc.build_fragment(m)
    assert "No heart-rate windows in this run." in frag and "No matched windows" not in frag
    run_["results"]["heart_rate"]["windows"] = [_w(500, 70)]
    frag = acc.build_fragment(acc.evaluate(run_, hr_ref=(np.array([T0_TS]), np.array([70.0]))))
    assert frag.count("No matched windows") == 1


def test_evaluate_errors_exit_2(tmp_path, capsys):
    (tmp_path / "r_results.json").write_text(json.dumps({"start_time": T0.isoformat(), "events": []}))
    (tmp_path / "hr.csv").write_text("timestamp,bpm\n2026-11-20T18:00:00Z,70\n")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert acc.main(["--run", str(tmp_path / "r_results.json"), "--hr-ref", str(tmp_path / "hr.csv"),
                     "--out", str(blocker / "sub")]) == 2          # cannot create output dir
    (tmp_path / "bad.csv").write_text("timestamp,bpm\nnot-a-time,70\n")
    assert acc.main(["--run", str(tmp_path / "r_results.json"), "--hr-ref", str(tmp_path / "bad.csv"),
                     "--out", str(tmp_path / "o")]) == 2
    assert "error:" in capsys.readouterr().err


def test_streaming_parse_memory_flat(tmp_path):
    import tracemalloc
    p = tmp_path / "big.xml"
    with open(p, "w") as f:
        f.write('<?xml version="1.0"?>\n<HealthData>\n')
        for i in range(100_000):
            f.write(f'<Record type="{acc.HR_TYPE}" startDate="2026-11-20 10:00:00 -0800" value="60"><MetadataEntry key="a" value="b"/></Record>\n')
        f.write("</HealthData>\n")
    tracemalloc.start()
    t, v = acc.parse_apple_health(p)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(v) == 100_000 and peak < 40e6


def test_alignment_prefers_recorded_span_and_legacy_default_is_15():
    ref_t = np.array([T0_TS + 100.0])
    ref_v = np.array([60.0])
    short = _w(106, 61); short["span_s"] = 12.0      # centre 100 (a 12 s span)
    legacy = _w(106, 61)                                # no span_s: falls back to window_s
    assert acc.align([short], T0_TS, ref_t, ref_v, tolerance_s=1, window_s=30)[0]["ref"] == 60.0
    assert acc.align([legacy], T0_TS, ref_t, ref_v, tolerance_s=1, window_s=30)[0]["ref"] is None
    assert acc._DEFAULT_WINDOW_S["heart_rate"] == 15.0
