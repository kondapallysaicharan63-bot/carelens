"""Accuracy harness: compare a CareLens run with a reference (watch HR, manual breath counts,
labelled falls) and write a report fragment for docs/ACCURACY_REPORT_TEMPLATE.md.

    python -m carelens.accuracy --run output/clip_results.json --hr-ref export.xml --out accuracy/
    python -m carelens.accuracy --web-session session.json --metric hr_fingertip --hr-ref export.xml --out accuracy/

Wellness trends only - this is not a medical device and the numbers are not a diagnosis.

Metric definitions (all errors are camera minus reference):
  * Window time: a window's ``t`` is its END; the window spans [t - window_s, t] and is
    compared at its CENTRE: wall_time = run start_time + t - window_s/2 + offset_s.
    window_s comes from the results (``window_s`` key) else the estimator default
    (heart rate 15 s, breathing 30 s; runs read from the DB do not store it).
  * Match: a window matches the reference samples within +-tolerance_s of its centre;
    the reference value is their mean (n_ref and spread = sample SD are recorded).
  * Web sessions (--web-session, the browser app's JSON export, validated against shared/session.schema.json):
    one --metric (hr_fingertip|hr_face|br_camera|br_sound) is scored with the same alignment and statistics;
    each estimate's own span_s centres its window, started_at carries its offset (no --tz needed), estimates
    with a null value are dropped, and a naive reference CSV is read in the session's own zone.
  * Naive timestamps (results start_time, reference CSV) are read in the run's zone: --tz, or
    the DB run's tz_offset_min, else the machine's local zone. Apple XML always has offsets.
  * "Main" metrics use REPORTED windows (quality gate passed) that have a match:
    n, MAE = mean|e|, RMSE = sqrt(mean e^2), bias = mean e,
    Bland-Altman 95% limits of agreement = bias +- 1.96*SD (SD ddof=1, only n >= 3),
    Pearson r (only n >= 5, else null), bootstrap 95% CI for MAE (seeded, 2000 resamples,
    percentile method, only n >= 5).
  * Coverage = reported windows / all windows.
  * MAE (all matched) also includes unreported windows, to show what the quality gate buys.
  * Falls: a fall event is a hit if |onset - label onset_s| <= event_tol_s. Sensitivity =
    hit labelled falls / labelled falls. False alarms = fall events matching no label
    (a `none` label for the clip means any fall event is a false alarm). PPV = hits /
    (hits + false alarms). False alarms per hour = false alarms / recording hours.
  Nothing is invented: with no data the values are null.
"""
from __future__ import annotations

import argparse
import re
import csv
import json
import math
import sqlite3
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import websession

HR_TYPE = "HKQuantityTypeIdentifierHeartRate"
MAX_ROWS = 200
_DEFAULT_WINDOW_S = {"heart_rate": 15.0, "breathing_rate": 30.0}   # legacy results without window_s (HR was the 15 s green estimator); new runs record window_s
WEB_METRICS = websession.METRICS
BOOT_N, BOOT_SEED = 2000, 12345


# ---------------------------------------------------------------- parsing
def _parse_dt(s: str, default_tz=None) -> float:
    """Return epoch seconds. Accepts ISO 8601 and Apple's 'YYYY-MM-DD HH:MM:SS +HHMM'."""
    s = s.strip()
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        dt = websession.parse_iso(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz) if default_tz else dt.astimezone()
    return dt.timestamp()


def parse_apple_health(path) -> tuple[np.ndarray, np.ndarray]:
    """Stream export.xml; keep only heart-rate Records. Returns (epoch_s, bpm) sorted by time.
    Finished children are dropped from the root so memory stays flat on huge exports."""
    ts, vals = [], []
    root = None
    for ev, el in ET.iterparse(str(path), events=("start", "end")):
        if ev == "start":
            if root is None:
                root = el
            continue
        if el.tag == "Record" and el.get("type") == HR_TYPE:
            try:
                ts.append(_parse_dt(el.get("startDate")))
                vals.append(float(el.get("value")))
            except (TypeError, ValueError):
                pass
        el.clear()
        if root is not None and len(root) > 1000:
            del root[:-1]   # the last child may still be open; everything before it is done
    return _sorted(ts, vals)


def _sorted(ts, vals):
    t, v = np.asarray(ts, float), np.asarray(vals, float)
    o = np.argsort(t, kind="stable")
    return t[o], v[o]


def parse_csv_ref(path, value_col: str, default_tz=None) -> tuple[np.ndarray, np.ndarray]:
    ts, vals = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts.append(_parse_dt(row["timestamp"], default_tz))
            vals.append(float(row[value_col]))
    return _sorted(ts, vals)


def load_reference(path, value_col: str, default_tz=None):
    p = Path(path)
    if p.suffix.lower() == ".xml":
        return parse_apple_health(p)
    return parse_csv_ref(p, value_col, default_tz)


def parse_labels(path) -> list[dict]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            kind = row["kind"].strip().lower()
            if kind not in ("fall", "none"):
                raise ValueError(f"label kind must be fall|none, got {kind!r}")
            onset = float(row["onset_s"]) if kind == "fall" else None
            out.append({"clip": row["clip"].strip(), "kind": kind, "onset_s": onset})
    return out


# ---------------------------------------------------------------- run loading
def load_run(spec: str, tz: Optional[str] = None) -> dict:
    """Return {results, start_ts, name, tzinfo}. spec = results JSON path or db:<path>#<run_id>."""
    if spec.startswith("db:"):
        from .store import Store
        path, _, rid = spec[3:].rpartition("#")
        if not path or not rid.isdigit():
            raise ValueError("expected db:<path>#<run_id>")
        if not Path(path).is_file():
            raise ValueError(f"database not found: {path}")
        try:
            with Store(path, readonly=True) as st:
                res = st.get_run(int(rid))
                if res is None:
                    raise ValueError(f"run {rid} not found in {path}")
                row = st.db.execute("SELECT name, tz_offset_min FROM runs WHERE id=?",
                                    (int(rid),)).fetchone()
        except (sqlite3.Error, RuntimeError) as e:
            raise ValueError(f"cannot read {path}: {e}") from e
        tzinfo = timezone(timedelta(minutes=row["tz_offset_min"]))
        name = row["name"]
    else:
        res = json.loads(Path(spec).read_text())
        tzinfo = _tz_from(tz) if tz else None
        name = Path(spec).stem.removesuffix("_results")
    start = websession.parse_iso(res["start_time"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=tzinfo) if tzinfo else start.astimezone()
    return {"results": res, "start_ts": start.timestamp(), "name": name, "tzinfo": tzinfo}


def _tz_from(s: str) -> timezone:
    """'+0530', '-08:00', 'UTC' or an IANA name."""
    if s.upper() in ("UTC", "Z"):
        return timezone.utc
    try:
        d = datetime.strptime(s.replace(":", ""), "%z")
        return timezone(d.utcoffset())
    except ValueError:
        from zoneinfo import ZoneInfo
        return ZoneInfo(s)  # type: ignore[return-value]


# ---------------------------------------------------------------- alignment & stats
def align(windows: list[dict], start_ts: float, ref_t: np.ndarray, ref_v: np.ndarray,
          tolerance_s: float = 30.0, offset_s: float = 0.0, window_s: float = 0.0) -> list[dict]:
    """Match each window (centre = t - span/2, span = the window's own ``span_s`` when recorded, else window_s)
    with reference samples within +-tolerance_s."""
    rows = []
    for w in windows:
        wall = start_ts + w["t"] - float(w.get("span_s") or window_s) / 2.0 + offset_s
        i0 = int(np.searchsorted(ref_t, wall - tolerance_s, side="left"))
        i1 = int(np.searchsorted(ref_t, wall + tolerance_s, side="right"))
        seg = ref_v[i0:i1]
        r = {"t": w["t"], "wall_time": datetime.fromtimestamp(wall, timezone.utc).isoformat(),
             "camera": w["value"], "quality": w["quality"], "reported": bool(w["reported"]),
             "ref": None, "n_ref": int(len(seg)), "ref_sd": None, "error": None}
        if len(seg):
            r["ref"] = float(seg.mean())
            r["ref_sd"] = float(seg.std(ddof=1)) if len(seg) > 1 else 0.0
            if w["value"] is not None:   # web sessions keep rows where the estimator withheld a value: counted, never scored
                r["error"] = float(w["value"] - r["ref"])
        rows.append(r)
    return rows


def _mae_ci(err: np.ndarray) -> Optional[list]:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(err), size=(BOOT_N, len(err)))
    maes = np.abs(err)[idx].mean(axis=1)
    return [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))]


def _stats(errs: np.ndarray) -> dict:
    n = len(errs)
    out = {"n": n, "mae": None, "rmse": None, "bias": None, "loa_low": None, "loa_high": None,
           "sd_diff": None}
    if n:
        out.update(mae=float(np.abs(errs).mean()), rmse=float(math.sqrt((errs ** 2).mean())),
                   bias=float(errs.mean()))
    if n >= 3:
        sd = float(errs.std(ddof=1))
        out.update(sd_diff=sd, loa_low=out["bias"] - 1.96 * sd, loa_high=out["bias"] + 1.96 * sd)
    return out


def summarize(rows: list[dict]) -> dict:
    matched = [r for r in rows if r["error"] is not None]
    rep = [r for r in matched if r["reported"]]
    errs = np.array([r["error"] for r in rep], float)
    s = _stats(errs)
    s["pearson_r"], s["mae_ci95"] = None, None
    if len(rep) >= 5:
        cam = np.array([r["camera"] for r in rep], float)
        ref = np.array([r["ref"] for r in rep], float)
        if cam.std() > 0 and ref.std() > 0:
            s["pearson_r"] = float(np.corrcoef(cam, ref)[0, 1])
        s["mae_ci95"] = _mae_ci(errs)
    allerr = np.array([r["error"] for r in matched], float)
    s.update(n_windows=len(rows), n_reported=sum(r["reported"] for r in rows),
             n_matched=len(matched),
             coverage=(sum(r["reported"] for r in rows) / len(rows)) if rows else None,
             mae_all_matched=float(np.abs(allerr).mean()) if len(allerr) else None,
             n_all_matched=len(allerr))
    if rows and not rep:
        s["note"] = ("No matched windows — check clock offset/time zone" if not matched else
                     "No reported windows matched a reference sample")
    return s


def suggest_offset(windows, start_ts, ref_t, tolerance_s, span=600, step=30,
                   window_s: float = 0.0) -> Optional[int]:
    """Offset (s) in +-span that maximises matched windows (ties: smallest mean distance to the
    nearest reference sample). Reported only, never applied. ref_t must be sorted."""
    if not windows or not len(ref_t):
        return None
    wt = np.array([start_ts + w["t"] - float(w.get("span_s") or window_s) / 2.0 for w in windows])
    margin = span + tolerance_s
    ref = ref_t[int(np.searchsorted(ref_t, wt.min() - margin, "left")):
                int(np.searchsorted(ref_t, wt.max() + margin, "right"))]
    if not len(ref):
        return None
    best, best_key = None, (0, 0.0)
    for off in range(-span, span + 1, step):
        q = wt + off
        i = np.searchsorted(ref, q)
        left = np.abs(q - ref[np.clip(i - 1, 0, len(ref) - 1)])
        right = np.abs(ref[np.clip(i, 0, len(ref) - 1)] - q)
        d = np.minimum(left, right)
        ok = d <= tolerance_s
        key = (int(ok.sum()), -float(d[ok].mean()) if ok.any() else 0.0)
        if key[0] > best_key[0] or (key[0] == best_key[0] and key[0] > 0 and key[1] > best_key[1]):
            best, best_key = off, key
    return best


# ---------------------------------------------------------------- events
def score_events(events: list[dict], labels: list[dict], clip: str, duration_s: float,
                 tol_s: float = 10.0) -> dict:
    """One-to-one matching: label/event pairs within tol_s are assigned smallest distance first."""
    falls = [e for e in events if e["kind"] == "fall"]
    mine = [l for l in labels if l["clip"] == clip]
    if not mine:
        return {"clip": clip, "note": f"No labels for clip {clip}", "n_labels_for_clip": 0,
                "n_true_falls": 0, "n_none_labels": 0, "hits": None, "misses": None,
                "false_alarms": None, "sensitivity": None, "ppv": None,
                "false_alarms_per_hour": None, "recording_hours": duration_s / 3600.0,
                "tolerance_s": tol_s, "per_label": []}
    true = [l for l in mine if l["kind"] == "fall"]
    pairs = sorted((abs(e["t_onset"] - l["onset_s"]), li, ei)
                   for li, l in enumerate(true) for ei, e in enumerate(falls)
                   if abs(e["t_onset"] - l["onset_s"]) <= tol_s)
    lab_used, ev_used = {}, set()
    for d, li, ei in pairs:
        if li not in lab_used and ei not in ev_used:
            lab_used[li] = ei
            ev_used.add(ei)
    per = [{"label_onset_s": l["onset_s"], "detected": li in lab_used,
            "onset_error_s": (falls[lab_used[li]]["t_onset"] - l["onset_s"]) if li in lab_used else None}
           for li, l in enumerate(true)]
    hits = len(lab_used)
    fa = len(falls) - hits
    hours = duration_s / 3600.0
    return {"clip": clip, "n_true_falls": len(true), "n_none_labels": sum(l["kind"] == "none" for l in mine),
            "hits": hits, "misses": len(true) - hits, "false_alarms": fa,
            "sensitivity": hits / len(true) if true else None,
            "ppv": hits / (hits + fa) if hits + fa else None,
            "false_alarms_per_hour": fa / hours if hours > 0 else None,
            "recording_hours": hours, "tolerance_s": tol_s, "per_label": per,
            "n_labels_for_clip": len(mine)}


# ---------------------------------------------------------------- report
def _f(x, d=1, unit=""):
    return "n/a" if x is None else f"{x:.{d}f}{unit}"


def _table(rows, unit_label, ref_label):
    out = [f"| Window (clip @ t) | Camera {unit_label} | Quality | Reported | {ref_label} | n ref | Error |",
           "|---|---|---|---|---|---|---|"]
    for r in rows[:MAX_ROWS]:
        out.append(f"| {r['clip']} @ {r['t']:.0f}s | {_f(r['camera'])} | {r['quality']:.2f} | "
                   f"{'yes' if r['reported'] else 'no'} | {_f(r['ref'])} | {r['n_ref']} | "
                   f"{('%+.1f' % r['error']) if r['error'] is not None else 'n/a'} |")
    if len(rows) > MAX_ROWS:
        out.append(f"\n_Showing first {MAX_ROWS} of {len(rows)} windows; full data in metrics.json._")
    return out


def _vital_lines(s, unit, what):
    if not s or s.get("n_windows", 0) == 0:
        return [f"- No {what} windows in this run."]
    L = []
    if s["n"] == 0:
        L.append(f"- **{s['note']}.**")
    else:
        ci = f" (95% CI {_f(s['mae_ci95'][0], 2)}–{_f(s['mae_ci95'][1], 2)})" if s["mae_ci95"] else ""
        L.append(f"- **Mean absolute error: {_f(s['mae'], 2)} {unit}**{ci} over {s['n']} reported matched windows"
                 f" · RMSE {_f(s['rmse'], 2)} · bias {_f(s['bias'], 2)}")
        if s["loa_low"] is not None:
            L.append(f"- Bland–Altman 95% limits of agreement: {_f(s['loa_low'], 2)} to {_f(s['loa_high'], 2)} {unit}")
        L.append(f"- Pearson r: {_f(s['pearson_r'], 2)}" + ("" if s["pearson_r"] is not None else " (needs n ≥ 5)"))
    L.append(f"- Coverage: {s['n_reported']} / {s['n_windows']} windows reported "
             f"({_f(100 * s['coverage'], 0) if s['coverage'] is not None else 'n/a'}%); "
             f"declined: {s['n_windows'] - s['n_reported']}")
    L.append(f"- MAE over ALL matched windows incl. unreported: {_f(s['mae_all_matched'], 2)} {unit} "
             f"(n={s['n_all_matched']}) — shows what the quality gate buys")
    return L


def build_fragment(m: dict) -> str:
    h = m["header"]
    L = ["<!-- generated by python -m carelens.accuracy -->",
         f"**Reference:** {h['reference']} · tolerance ±{h['tolerance_s']:g} s · clock offset "
         f"{h['offset_s']:g} s · run `{h['run']}` ({h['recording_hours'] * 3600:.0f} s)",
         "", "_Wellness trends only — not a diagnosis. Errors are camera − reference._", "",
         "## 2. Fall detection", ""]
    ev = m["events"]
    if ev is None:
        L.append("No labels file supplied (`--labels`).")
    elif ev.get("note"):
        L.append(ev["note"] + ".")
    else:
        L += ["| Clip | True event | Detected? | Onset error (s) | Notes |", "|---|---|---|---|---|"]
        for p in ev["per_label"]:
            L.append(f"| {ev['clip']} | fall @ {p['label_onset_s']:.1f}s | {'yes' if p['detected'] else 'NO (miss)'} | "
                     f"{_f(p['onset_error_s'], 1)} | |")
        if ev["n_none_labels"]:
            L.append(f"| {ev['clip']} | none (should NOT fire) | "
                     f"{'FIRED' if ev['false_alarms'] else 'no'} | n/a | |")
        L += ["", f"- Hits: {ev['hits']} / {ev['n_true_falls']}   False alarms: {ev['false_alarms']}   "
                  f"Misses: {ev['misses']}",
              f"- Sensitivity: {_f(ev['sensitivity'], 2)} · PPV: {_f(ev['ppv'], 2)} · false alarms/hour: "
              f"{_f(ev['false_alarms_per_hour'], 2)} (match tolerance ±{ev['tolerance_s']:g} s)"]
    for key, title, unit, ref_label, what in (
            ("heart_rate", "3. Heart rate (rPPG vs reference)", "bpm", "Reference bpm", "heart-rate"),
            ("breathing_rate", "4. Breathing rate", "/min", "Reference /min", "breathing-rate")):
        L += ["", f"## {title}", ""]
        sec = m[key]
        if sec is None:
            L.append(f"No reference supplied for {what}.")
            continue
        if sec["windows"]:
            L += _table(sec["windows"], unit, ref_label) + [""]
        L += _vital_lines(sec["summary"], unit, what)
        if sec.get("suggested_offset_s") is not None:
            L.append(f"- Hint (not applied): offset {sec['suggested_offset_s']:+d} s would match the most windows.")
    return "\n".join(L) + "\n"


def evaluate(run: dict, hr_ref=None, br_ref=None, labels=None, tolerance_s=30.0, offset_s=0.0,
             event_tol_s=10.0, reference_name="reference", clip: Optional[str] = None) -> dict:
    res, start_ts = run["results"], run["start_ts"]
    clip = clip or run["name"]
    m = {"header": {"reference": reference_name, "tolerance_s": tolerance_s, "offset_s": offset_s,
                    "run": clip, "start_ts": start_ts, "recording_hours": res.get("duration_s", 0) / 3600.0,
                    "disclaimer": "Wellness trends only — not a diagnosis."},
         "heart_rate": None, "breathing_rate": None, "events": None}
    for key, ref in (("heart_rate", hr_ref), ("breathing_rate", br_ref)):
        if ref is None:
            continue
        wins = res.get(key, {}).get("windows", [])
        wsec = res.get(key, {})
        ws = float(wsec.get("window_s") or _DEFAULT_WINDOW_S[key])
        rows = align(wins, start_ts, ref[0], ref[1], tolerance_s, offset_s, ws)
        for r in rows:
            r["clip"] = clip
        summ = summarize(rows)
        sec = {"windows": rows, "summary": summ, "suggested_offset_s": None, "window_s": ws}
        if wins and summ["n_matched"] == 0:
            sec["suggested_offset_s"] = suggest_offset([w for w in wins if w["value"] is not None], start_ts, ref[0], tolerance_s, window_s=ws)
        m[key] = sec
    if labels is not None:
        m["events"] = score_events(res.get("events", []), labels, clip, res.get("duration_s", 0), event_tol_s)
    return m


def write_outputs(m: dict, out_dir) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp, fp = out / "metrics.json", out / "report_fragment.md"
    mp.write_text(json.dumps(m, indent=2))
    fp.write_text(build_fragment(m))
    return mp, fp


def _summary_text(m: dict) -> str:
    L = []
    for key, unit in (("heart_rate", "bpm"), ("breathing_rate", "/min")):
        sec = m[key]
        if sec is None:
            continue
        s = sec["summary"]
        if s["n"] == 0:
            L.append(f"{key}: No matched windows — check clock offset/time zone"
                     if s["n_matched"] == 0 else f"{key}: no reported window matched")
            if sec["suggested_offset_s"] is not None:
                L.append(f"  hint (not applied): offset {sec['suggested_offset_s']:+d} s")
        else:
            loa = f", LoA {_f(s['loa_low'], 2)}..{_f(s['loa_high'], 2)}" if s["loa_low"] is not None else ""
            L.append(f"{key}: n={s['n']} MAE={_f(s['mae'], 2)} {unit} bias={_f(s['bias'], 2)}{loa}")
        L.append(f"  coverage {s['n_reported']}/{s['n_windows']}; MAE incl. unreported {_f(s['mae_all_matched'], 2)}")
    ev = m["events"]
    if ev and ev.get("note"):
        L.append(f"falls: {ev['note']}")
    elif ev:
        L.append(f"falls: hits {ev['hits']}/{ev['n_true_falls']} sensitivity={_f(ev['sensitivity'], 2)} "
                 f"false alarms={ev['false_alarms']} ({_f(ev['false_alarms_per_hour'], 2)}/h)")
    return "\n".join(L) + "\nWellness trends only — not a diagnosis."


_TZ_ARG = re.compile(r"^[-+]\d{1,2}(:?\d{2})?$")


def _fix_argv(argv):
    """Let `--tz -08:00` work: argparse would read '-08:00' as an option, so attach it."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--tz" and i + 1 < len(argv) and _TZ_ARG.match(argv[i + 1]):
            out.append(f"--tz={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="carelens.accuracy", description=__doc__.split("\n")[0])
    ap.add_argument("--run", help="results JSON or db:<path>#<run_id>")
    ap.add_argument("--web-session", help="session JSON exported by the web app (needs --metric)")
    ap.add_argument("--metric", choices=WEB_METRICS, help="with --web-session: which estimate series to score")
    ap.add_argument("--hr-ref", help="Apple Health export.xml or CSV (timestamp,bpm)")
    ap.add_argument("--br-ref", help="CSV (timestamp,breaths_per_min)")
    ap.add_argument("--labels", help="CSV clip,kind,onset_s")
    ap.add_argument("--clip", help="clip name for label matching (default: run name)")
    ap.add_argument("--tz", help="zone for naive times (results start_time, reference CSVs), "
                    "e.g. --tz=-08:00 or America/Los_Angeles (default: machine local)")
    ap.add_argument("--offset-s", type=float, default=0.0)
    ap.add_argument("--tolerance-s", type=float, default=30.0)
    ap.add_argument("--event-tol-s", type=float, default=10.0)
    ap.add_argument("--reference-name", default="Apple Watch (Health export)")
    ap.add_argument("--out", default="accuracy")
    a = ap.parse_args(_fix_argv(list(sys.argv[1:] if argv is None else argv)))
    try:
        if bool(a.run) == bool(a.web_session):
            raise ValueError("give exactly one of --run or --web-session")
        if a.web_session:
            if not a.metric:
                raise ValueError("--web-session needs --metric " + "|".join(WEB_METRICS))
            need_hr = a.metric.startswith("hr_")
            if not (a.hr_ref if need_hr else a.br_ref):
                raise ValueError(f"--metric {a.metric} needs {'--hr-ref' if need_hr else '--br-ref'}")
            if a.tz:
                print("note: --tz is ignored for web sessions (started_at carries its offset)", file=sys.stderr)
            run = websession.to_run(websession.load_session(a.web_session), a.metric)
            ignored = "--br-ref" if need_hr and a.br_ref else "--hr-ref" if not need_hr and a.hr_ref else None
            if ignored:
                print(f"note: {ignored} is ignored: --metric {a.metric} scores only its own reference", file=sys.stderr)
            if need_hr:
                a.br_ref = None   # only the chosen metric is scored
            else:
                a.hr_ref = None
        else:
            if a.metric:
                raise ValueError("--metric only applies with --web-session")
            run = load_run(a.run, a.tz)
        tz = run["tzinfo"]
        hr = load_reference(a.hr_ref, "bpm", tz) if a.hr_ref else None
        br = load_reference(a.br_ref, "breaths_per_min", tz) if a.br_ref else None
        labels = parse_labels(a.labels) if a.labels else None
        m = evaluate(run, hr, br, labels, a.tolerance_s, a.offset_s, a.event_tol_s,
                     a.reference_name, a.clip)
        mp, fp = write_outputs(m, a.out)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, sqlite3.Error, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(_summary_text(m))
    print(f"wrote {mp} and {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
