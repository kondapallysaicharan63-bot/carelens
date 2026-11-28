"""Web session export (shared/session.schema.json): validator and adapter for ``carelens.accuracy``.

The browser app exports one JSON file per session (estimates + check-in events; no media, no device ids).
``validate`` is hand-written (stdlib only) and mirrors web/src/export/schema.ts; tests keep both in step with the
JSON schema file. Wellness trends only - not a medical device.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 1
METRICS = ("hr_fingertip", "hr_face", "br_camera", "br_sound")
EVENT_KINDS = ("possible_fall", "inactivity", "cough", "distress_sound", "possible_pause")
MODES = ("camera", "microphone", "fingertip")
UA_FAMILIES = ("Chrome", "Edge", "Firefox", "Safari", "Other")
# estimator default windows (s) when an estimate carries no span_s
DEFAULT_SPAN_S = {"hr_fingertip": 10.0, "hr_face": 12.0, "br_camera": 30.0, "br_sound": 30.0}

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
_TOP = ("schema_version", "app_version", "params_version", "started_at", "tz_offset_min", "device", "modes",
        "estimates", "events", "disclaimer")
_OPTIONAL_TOP = ("truncated", "summaries")
_EST_REQ = ("t", "wall", "metric", "value", "quality", "reported", "reason")
_EV_REQ = ("t", "wall", "kind")
_EV_ALL = _EV_REQ + ("t_onset", "t_end", "peak_score", "trigger", "sound_cue", "merged_into_fall")
SUMMARY_KINDS = ("hrv", "chair_stand_30s", "sit_to_stand_5x", "tug", "walking_speed", "hand_tremor", "blink", "activity", "snoring", "pauses", "talking")
MOBILITY_KINDS = SUMMARY_KINDS[1:5]
PHASE_KEYS = ("rise", "walk_out", "turn_1", "walk_back", "turn_2", "sit")
MAX_REP_TIMES = 100
# reported rows must carry this headline number (kind -> required fields)
_SUM_HEADLINE = {"hrv": ("rmssd_ms", "sdnn_ms"), "chair_stand_30s": ("reps",), "sit_to_stand_5x": ("time_s",), "tug": ("time_s",),
                 "walking_speed": ("speed_m_s",), "hand_tremor": ("freq_hz", "amp_pct"), "blink": ("blinks_per_min", "eyes_closed_pct"),
                 "activity": ("active_min", "still_min", "absent_min"), "snoring": ("count", "minutes"),
                 "pauses": ("count", "regular_min"), "talking": ("minutes",)}
_SUM_REQ = ("t", "wall", "kind", "reported", "reason")
_SUM_ALL = _SUM_REQ + ("rmssd_ms", "sdnn_ms", "mean_hr_bpm", "n_beats", "n_kept", "kept_frac", "duration_s",
                      "reps", "time_s", "speed_m_s", "cadence_spm", "quality", "rep_times_s", "phases_s", "freq_hz", "amp_pct", "p2p_pct",
                      "blinks_per_min", "eyes_closed_pct", "long_closures", "active_min", "still_min", "absent_min", "longest_still_min",
                      "count", "minutes", "per_hour", "rate_per_h", "regular_min", "durations_s")
MAX_PER_HOUR = 72
MAX_DURATIONS = 500


class SessionError(ValueError):
    """The file is not a valid CareLens web session export; the message lists the problems."""


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x) -> bool:
    """JSON Schema "integer": 7 and 7.0 are the same JSON number (as in JavaScript); booleans are not numbers."""
    if isinstance(x, bool):
        return False
    return isinstance(x, int) or (isinstance(x, float) and math.isfinite(x) and x.is_integer())


_FRAC = re.compile(r"\.(\d+)")


def normalize_iso(s: str) -> str:
    """Make an ISO 8601 string readable by ``datetime.fromisoformat`` on Python 3.10 (which only takes 3 or 6 fractional
    digits and no 'Z'): fractional seconds are padded or truncated to 6 digits, 'Z' becomes +00:00."""
    s = _FRAC.sub(lambda m: "." + (m.group(1) + "000000")[:6], s.strip())
    return s[:-1] + "+00:00" if s.endswith(("Z", "z")) else s


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(normalize_iso(s))


def _iso(x) -> bool:
    if not (isinstance(x, str) and _ISO.match(x)):
        return False
    try:
        parse_iso(x)
    except ValueError:
        return False
    return True


def _extra(o: dict, allowed, where: str, errs: list) -> None:
    errs += [f"{where}: unexpected field {k!r}" for k in o if k not in allowed]


def validate(x) -> list[str]:
    """List of problems (empty = valid)."""
    if not isinstance(x, dict):
        return ["session: must be a JSON object"]
    errs: list[str] = []
    errs += [f"session: missing {k!r}" for k in _TOP if k not in x]
    _extra(x, _TOP + _OPTIONAL_TOP, "session", errs)
    if "truncated" in x and not isinstance(x["truncated"], bool):
        errs.append("truncated: expected a boolean")
    if "schema_version" in x and not (_int(x["schema_version"]) and x["schema_version"] == SCHEMA_VERSION):
        errs.append(f"schema_version: expected {SCHEMA_VERSION}, got {x['schema_version']!r}")
    if "app_version" in x and not (isinstance(x["app_version"], str) and 1 <= len(x["app_version"]) <= 32):
        errs.append("app_version: expected a string of 1-32 characters")
    if "params_version" in x and not (_int(x["params_version"]) and x["params_version"] >= 0):
        errs.append("params_version: expected a non-negative integer")
    if "started_at" in x and not _iso(x["started_at"]):
        errs.append("started_at: expected ISO 8601 with a UTC offset")
    if "tz_offset_min" in x and not (_int(x["tz_offset_min"]) and abs(x["tz_offset_min"]) <= 840):
        errs.append("tz_offset_min: expected an integer within +-840")
    if "disclaimer" in x and not (isinstance(x["disclaimer"], str) and x["disclaimer"]):
        errs.append("disclaimer: expected a non-empty string")
    if "device" in x:
        d = x["device"]
        if not isinstance(d, dict):
            errs.append("device: must be an object")
        else:
            _extra(d, ("user_agent_family",), "device", errs)
            if d.get("user_agent_family") not in UA_FAMILIES:
                errs.append(f"device.user_agent_family: expected one of {'|'.join(UA_FAMILIES)}")
    if "modes" in x:
        m = x["modes"]
        if not (isinstance(m, list) and all(isinstance(i, str) and i in MODES for i in m) and len(set(m)) == len(m)):
            errs.append(f"modes: expected unique items from {'|'.join(MODES)}")
    for key, check in (("estimates", _estimate_errors), ("events", _event_errors), ("summaries", _summary_errors)):
        if key in x:
            if not isinstance(x[key], list):
                errs.append(f"{key}: must be an array")
            else:
                for i, e in enumerate(x[key]):
                    check(e, f"{key}[{i}]", errs)
    return errs[:50]


def _estimate_errors(e, w: str, errs: list) -> None:
    if not isinstance(e, dict):
        errs.append(f"{w}: must be an object")
        return
    errs += [f"{w}: missing {k!r}" for k in _EST_REQ if k not in e]
    _extra(e, _EST_REQ + ("span_s",), w, errs)
    if "t" in e and not (_num(e["t"]) and e["t"] >= 0):
        errs.append(f"{w}.t: expected a number >= 0")
    if "wall" in e and not _iso(e["wall"]):
        errs.append(f"{w}.wall: expected ISO 8601 with a UTC offset")
    if "metric" in e and e["metric"] not in METRICS:
        errs.append(f"{w}.metric: expected one of {'|'.join(METRICS)}")
    if "value" in e and not (e["value"] is None or _num(e["value"])):
        errs.append(f"{w}.value: expected a number or null")
    if "quality" in e and not (_num(e["quality"]) and 0 <= e["quality"] <= 1):
        errs.append(f"{w}.quality: expected a number in [0, 1]")
    if "reported" in e and not isinstance(e["reported"], bool):
        errs.append(f"{w}.reported: expected a boolean")
    if e.get("reported") is True and e.get("value") is None and "value" in e:
        errs.append(f"{w}: reported rows need a value")
    if "reason" in e and not (isinstance(e["reason"], str) and len(e["reason"]) <= 40):
        errs.append(f"{w}.reason: expected a string up to 40 characters")
    if "span_s" in e and not (_num(e["span_s"]) and e["span_s"] > 0):
        errs.append(f"{w}.span_s: expected a number > 0")


def _event_errors(e, w: str, errs: list) -> None:
    if not isinstance(e, dict):
        errs.append(f"{w}: must be an object")
        return
    errs += [f"{w}: missing {k!r}" for k in _EV_REQ if k not in e]
    _extra(e, _EV_ALL, w, errs)
    if "t" in e and not (_num(e["t"]) and e["t"] >= 0):
        errs.append(f"{w}.t: expected a number >= 0")
    if "wall" in e and not _iso(e["wall"]):
        errs.append(f"{w}.wall: expected ISO 8601 with a UTC offset")
    if "kind" in e and e["kind"] not in EVENT_KINDS:
        errs.append(f"{w}.kind: expected one of {'|'.join(EVENT_KINDS)}")
    for k in ("t_onset", "t_end"):
        if k in e and not (_num(e[k]) and e[k] >= 0):
            errs.append(f"{w}.{k}: expected a number >= 0")
    if "peak_score" in e and not (_num(e["peak_score"]) and 0 <= e["peak_score"] <= 1):
        errs.append(f"{w}.peak_score: expected a number in [0, 1]")
    if "trigger" in e and not (isinstance(e["trigger"], str) and len(e["trigger"]) <= 40):
        errs.append(f"{w}.trigger: expected a string up to 40 characters")
    for k in ("sound_cue", "merged_into_fall"):
        if k in e and not isinstance(e[k], bool):
            errs.append(f"{w}.{k}: expected a boolean")


def _summary_errors(e, w: str, errs: list) -> None:
    """``summaries[]``: whole-measurement results (W6 HRV). Optional top-level array; absent in older exports."""
    if not isinstance(e, dict):
        errs.append(f"{w}: must be an object")
        return
    errs += [f"{w}: missing {k!r}" for k in _SUM_REQ if k not in e]
    _extra(e, _SUM_ALL, w, errs)
    if "t" in e and not (_num(e["t"]) and e["t"] >= 0):
        errs.append(f"{w}.t: expected a number >= 0")
    if "wall" in e and not _iso(e["wall"]):
        errs.append(f"{w}.wall: expected ISO 8601 with a UTC offset")
    if "kind" in e and e["kind"] not in SUMMARY_KINDS:
        errs.append(f"{w}.kind: expected one of {'|'.join(SUMMARY_KINDS)}")
    if "reported" in e and not isinstance(e["reported"], bool):
        errs.append(f"{w}.reported: expected a boolean")
    if "reason" in e and not (isinstance(e["reason"], str) and len(e["reason"]) <= 40):
        errs.append(f"{w}.reason: expected a string up to 40 characters")
    for k in ("rmssd_ms", "sdnn_ms", "duration_s"):
        if k in e and not (_num(e[k]) and e[k] >= 0):
            errs.append(f"{w}.{k}: expected a number >= 0")
    if "mean_hr_bpm" in e and not (_num(e["mean_hr_bpm"]) and e["mean_hr_bpm"] > 0):
        errs.append(f"{w}.mean_hr_bpm: expected a number > 0")
    for k in ("n_beats", "n_kept"):
        if k in e and not (_int(e[k]) and e[k] >= 0):
            errs.append(f"{w}.{k}: expected an integer >= 0")
    if "kept_frac" in e and not (_num(e["kept_frac"]) and 0 <= e["kept_frac"] <= 1):
        errs.append(f"{w}.kept_frac: expected a number in [0, 1]")
    # W7 mobility tests (wellbeing trends; no norms): all optional, ``kind`` says which apply
    if "reps" in e and not (_int(e["reps"]) and e["reps"] >= 0):
        errs.append(f"{w}.reps: expected an integer >= 0")
    for k in ("time_s", "speed_m_s"):
        if k in e and not (_num(e[k]) and e[k] >= 0):
            errs.append(f"{w}.{k}: expected a number >= 0")
    if "cadence_spm" in e and not (_num(e["cadence_spm"]) and e["cadence_spm"] > 0):
        errs.append(f"{w}.cadence_spm: expected a number > 0")
    if "quality" in e and not (_num(e["quality"]) and 0 <= e["quality"] <= 1):
        errs.append(f"{w}.quality: expected a number in [0, 1]")
    if "rep_times_s" in e:
        r = e["rep_times_s"]
        if not (isinstance(r, list) and len(r) <= MAX_REP_TIMES and all(_num(v) and v >= 0 for v in r)):
            errs.append(f"{w}.rep_times_s: expected up to {MAX_REP_TIMES} numbers >= 0")
    # W8 rhythmic hand movement (wellbeing trend): frequency in Hz, amplitude as % of hand size
    if "freq_hz" in e and not (_num(e["freq_hz"]) and e["freq_hz"] > 0):
        errs.append(f"{w}.freq_hz: expected a number > 0")
    for k in ("amp_pct", "p2p_pct"):
        if k in e and not (_num(e[k]) and e[k] >= 0):
            errs.append(f"{w}.{k}: expected a number >= 0")
    # W9 blink rate / eyes-closed time and activity level (wellbeing trends; minutes and percent, no interpretation)
    for k in ("blinks_per_min", "active_min", "still_min", "absent_min", "longest_still_min"):
        if k in e and not (_num(e[k]) and e[k] >= 0):
            errs.append(f"{w}.{k}: expected a number >= 0")
    if "eyes_closed_pct" in e and not (_num(e["eyes_closed_pct"]) and 0 <= e["eyes_closed_pct"] <= 100):
        errs.append(f"{w}.eyes_closed_pct: expected a number in [0, 100]")
    if "long_closures" in e and not (_int(e["long_closures"]) and e["long_closures"] >= 0):
        errs.append(f"{w}.long_closures: expected an integer >= 0")
    # W10 sound summaries (snoring sounds, possible pauses in breathing sounds, talking time): counts, minutes, per-hour lists
    if "count" in e and not (_int(e["count"]) and e["count"] >= 0):
        errs.append(f"{w}.count: expected an integer >= 0")
    for k in ("minutes", "rate_per_h", "regular_min"):
        if k in e and not (_num(e[k]) and e[k] >= 0):
            errs.append(f"{w}.{k}: expected a number >= 0")
    if "per_hour" in e:
        r = e["per_hour"]
        if not (isinstance(r, list) and len(r) <= MAX_PER_HOUR and all(_num(v) and v >= 0 for v in r)):
            errs.append(f"{w}.per_hour: expected up to {MAX_PER_HOUR} numbers >= 0")
    if "durations_s" in e:
        r = e["durations_s"]
        if not (isinstance(r, list) and len(r) <= MAX_DURATIONS and all(_num(v) and v >= 0 for v in r)):
            errs.append(f"{w}.durations_s: expected up to {MAX_DURATIONS} numbers >= 0")
    if "phases_s" in e:
        ph = e["phases_s"]
        if not isinstance(ph, dict):
            errs.append(f"{w}.phases_s: must be an object")
        else:
            _extra(ph, PHASE_KEYS, f"{w}.phases_s", errs)
            errs += [f"{w}.phases_s.{k}: expected a number >= 0" for k, v in ph.items() if k in PHASE_KEYS and not (_num(v) and v >= 0)]
    if e.get("reported") is True and e.get("kind") in _SUM_HEADLINE:
        need = _SUM_HEADLINE[e["kind"]]
        if not all(_num(e.get(k)) for k in need):
            errs.append(f"{w}: reported rows need {' and '.join(need)}")


def load_session(path) -> dict:
    """Read and validate a session file; raises SessionError (a ValueError) with every problem found."""
    p = Path(path)
    if not p.is_file():
        raise SessionError(f"session file not found: {p}")
    try:
        data = json.loads(p.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SessionError(f"{p.name}: not valid JSON ({e})") from e
    errs = validate(data)
    if errs:
        more = "" if len(errs) < 50 else " (first 50 shown)"
        raise SessionError(f"{p.name} is not a valid CareLens web session export{more}:\n  - " + "\n  - ".join(errs))
    return data


def to_run(session: dict, metric: str, name: str | None = None) -> dict:
    """Adapt a validated session to the run dict ``carelens.accuracy.evaluate`` takes.

    Rows with value null (the estimator withheld a value) are KEPT: ``accuracy`` counts them in n_windows / coverage
    like the unreported windows of a device run, and leaves them out of the error statistics. ``t`` is seconds since started_at and each
    window's span_s centres it (``accuracy.align``); started_at carries its own offset, so no --tz is needed.
    """
    if metric not in METRICS:
        raise SessionError(f"unknown metric {metric!r}; expected one of {', '.join(METRICS)}")
    start = parse_iso(session["started_at"])
    rows = [e for e in session["estimates"] if e["metric"] == metric]
    spans = sorted(float(e["span_s"]) for e in rows if e.get("span_s"))
    span = spans[len(spans) // 2] if spans else DEFAULT_SPAN_S[metric]
    wins = []
    for e in rows:
        w = {"t": float(e["t"]), "value": None if e["value"] is None else float(e["value"]),
             "quality": float(e["quality"]), "reported": bool(e["reported"])}
        w["span_s"] = float(e.get("span_s") or span)
        wins.append(w)
    key = "heart_rate" if metric.startswith("hr_") else "breathing_rate"
    events = [{"kind": "fall", "t_onset": float(ev.get("t_onset", ev["t"]))} for ev in session["events"]
              if ev["kind"] == "possible_fall"]
    ends = [e["t"] for e in session["estimates"]] + [e["t"] for e in session["events"]]
    res = {"start_time": session["started_at"], "duration_s": max(ends, default=0.0), "events": events,
           key: {"windows": wins, "window_s": span}}
    tzinfo = timezone(timedelta(minutes=float(session["tz_offset_min"])))
    return {"results": res, "start_ts": start.timestamp(), "name": name or f"web-{metric}", "tzinfo": tzinfo}
