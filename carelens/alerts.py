"""Caregiver alerting (opt-in, local-first): webhook / desktop notification sinks.

Wellness wording only: alerts suggest a check-in; they never claim a fall or an emergency was
confirmed. CareLens is not a medical device. Nothing leaves the machine unless the user
configures a webhook sink; webhook URLs/tokens come from the environment, never the config
file, and are never logged (errors are redacted).

Flow: ``AlertDispatcher.submit()`` (non-blocking, never raises) -> queue -> worker thread
records alert rows in SQLite (per sink, status ``pending``) -> sends them with retries and
exponential backoff. Rows survive restarts; dedupe is a DB query so it does too.
"""
from __future__ import annotations

import json
import logging
import queue
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, time as dtime
from typing import Callable, Optional, Protocol, Sequence
from urllib.parse import urlsplit

from .config import AlertsConfig, QuietHours, redact_url
from .discovery import host_allowed
from .store import Store

log = logging.getLogger(__name__)

ALERT_KINDS = ("fall", "inactivity", "camera_offline")
DISCLAIMER = ("CareLens is a wellness aid, not a medical device. Not for emergencies - call "
              "local emergency services if someone needs urgent help.")
TITLES = {
    "fall": "Possible fall — check-in suggested",
    "inactivity": "Long period without movement — check-in suggested",
    "camera_offline": "Camera offline — check the camera",
}
_MESSAGES = {
    "fall": "A possible fall was noticed in the {room}. A check-in is suggested.",
    "inactivity": "No movement has been noticed in the {room} for a long period. "
                  "A check-in is suggested.",
    "camera_offline": "The camera in the {room} is not reachable. Please check the camera.",
}
WEBHOOK_TIMEOUT_S = 5.0
CLAIM_LEASE_S = 120.0


class PermanentSinkError(Exception):
    """Raised by a sink when retrying cannot help (marked failed immediately)."""


class Sink(Protocol):
    name: str
    kinds: Sequence[str]

    def send(self, alert: dict) -> None:
        """Deliver the payload; raise on any failure."""


# ---- pure helpers ----------------------------------------------------------------------
def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def in_quiet_hours(qh: Optional[QuietHours], t: dtime) -> bool:
    """True if local time-of-day `t` is inside the window [start, end); may wrap midnight."""
    if qh is None:
        return False
    s, e, now = _minutes(qh.start), _minutes(qh.end), t.hour * 60 + t.minute
    return s <= now < e if s < e else (now >= s or now < e)


def is_suppressed(kind: str, qh: Optional[QuietHours], t: dtime) -> bool:
    """Quiet hours hold back ONLY inactivity; falls and camera-offline always go out."""
    return kind == "inactivity" and in_quiet_hours(qh, t)


_URL_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")
_BEARER_RE = re.compile(r"(?i)\b(bearer|token|authorization)\b[:=\s]+\S+")


def redact_error(msg: object, secrets: Sequence[str] = (), limit: int = 200) -> str:
    """Make an error message safe to store/log: no URLs, userinfo, tokens; <= `limit` chars."""
    s = str(msg)
    for sec in secrets:
        if sec:
            s = s.replace(sec, "***")
    s = _URL_RE.sub("<url>", s)
    s = _BEARER_RE.sub(r"\1 ***", s)
    return s[:limit]


def build_payload(row: dict, room: str) -> dict:
    kind = row["kind"]
    ev = json.loads(row["event_json"]) if row.get("event_json") else {}
    label = ev.get("data_label") or "live"
    ts = datetime.fromtimestamp(row["created_ts"]).astimezone().isoformat()
    return {"id": row["id"], "camera_id": row["camera_id"], "room": room, "kind": kind,
            "data_label": label,
            "title": TITLES[kind] if label == "live" else "[TEST] " + TITLES[kind], "message": _MESSAGES[kind].format(room=room), "ts": ts,
            "clip_name": ev.get("clip"), "disclaimer": DISCLAIMER}


def backoff_delay(attempts: int, base: float, cap: float, jitter: float = 0.0) -> float:
    """Delay before the next try after `attempts` failures: min(cap, base*2**attempts) * (1+jitter)."""
    return min(cap, base * (2 ** attempts)) * (1.0 + jitter)


# ---- sinks -----------------------------------------------------------------------------
def _safe_target(url: str) -> str:
    """scheme://host only: the path or query of a webhook URL may itself be a secret."""
    try:
        sp = urlsplit(url)
        return redact_url(f"{sp.scheme}://{sp.hostname or '?'}")
    except ValueError:
        return "<invalid url>"


def validate_webhook_url(url: str, allow_hosts: Sequence[str] = ()) -> None:
    """Raise ValueError (message never contains the full URL) unless the URL is acceptable."""
    try:
        sp = urlsplit(url)
        has_userinfo = "@" in sp.netloc
        _ = sp.port
    except ValueError:
        raise ValueError("webhook URL is not a valid URL") from None
    if has_userinfo:
        raise ValueError(f"webhook URL ({_safe_target(url)}) must not contain embedded "
                         "credentials; use the token env var instead")
    if not host_allowed(url, allow_hosts):
        raise ValueError(f"webhook host not allowed ({_safe_target(url)}): only http(s) to "
                         "private/loopback/.local hosts, or hosts listed in alerts.allow_public_hosts")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):   # None -> urllib raises the 3xx as HTTPError
        return None


class WebhookSink:
    def __init__(self, url: str, token: Optional[str] = None, allow_hosts: Sequence[str] = (),
                 kinds: Sequence[str] = ALERT_KINDS, name: str = "webhook",
                 timeout: float = WEBHOOK_TIMEOUT_S):
        validate_webhook_url(url, allow_hosts)
        self._url, self._token, self._allow = url, token, tuple(allow_hosts)
        self.kinds, self.name, self.timeout = tuple(kinds), name, timeout
        # no env proxies (local-first), no redirects
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)

    def send(self, alert: dict) -> None:
        if not host_allowed(self._url, self._allow):   # re-check at send time
            raise PermanentSinkError("webhook host not allowed")
        headers = {"Content-Type": "application/json", "User-Agent": "carelens-alerts"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(self._url, json.dumps(alert).encode(), headers, method="POST")
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                r.read(1024)
        except urllib.error.HTTPError as e:
            msg = f"HTTP {e.code}"
            if 300 <= e.code < 500 and e.code not in (408, 429):
                raise PermanentSinkError(msg) from None   # redirect / client error: retry is futile
            raise RuntimeError(msg) from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"URLError: {redact_error(e.reason, self._secrets())}") from None
        except OSError as e:
            raise RuntimeError(f"{type(e).__name__}: {redact_error(e, self._secrets())}") from None

    def _secrets(self):
        return [s for s in (self._url, self._token) if s]


class DesktopSink:
    """Desktop notification via ``notify-send`` (Linux). Missing binary = failed once, no retries."""
    name = "desktop"

    def __init__(self, kinds: Sequence[str] = ALERT_KINDS, name: str = "desktop",
                 timeout: float = 5.0):
        self.kinds, self.name, self.timeout = tuple(kinds), name, timeout

    def send(self, alert: dict) -> None:
        exe = shutil.which("notify-send")
        if not exe:
            raise PermanentSinkError("notify-send not found; desktop notifications unavailable")
        try:
            subprocess.run([exe, "--app-name=CareLens", alert["title"], alert["message"]],
                           check=True, timeout=self.timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError("notify-send timed out") from None
        except (subprocess.CalledProcessError, OSError) as e:
            raise RuntimeError(f"notify-send failed ({type(e).__name__})") from None


def build_sinks(cfg: AlertsConfig, env: dict) -> list:
    """Create sinks from config + environment. Raises ValueError (redacted) on bad settings."""
    sinks, seen = [], {}
    for sc in cfg.sinks:
        seen[sc.type] = seen.get(sc.type, 0) + 1
        name = sc.type if seen[sc.type] == 1 else f"{sc.type}{seen[sc.type]}"
        if sc.type == "webhook":
            url = env.get(sc.url_env or "")
            if not url:
                raise ValueError(f"alert sink: environment variable {sc.url_env} is not set "
                                 "(define it in .env or the environment)")
            if sc.token_env and not env.get(sc.token_env):
                raise ValueError(f"alert sink: environment variable {sc.token_env} is not set "
                                 "(define it in .env or the environment, or remove token_env)")
            token = env.get(sc.token_env) if sc.token_env else None
            sinks.append(WebhookSink(url, token, cfg.allow_public_hosts, sc.kinds, name))
        else:
            sinks.append(DesktopSink(sc.kinds, name))
    return sinks


# ---- dispatcher ------------------------------------------------------------------------
class AlertDispatcher:
    """Records and delivers alerts. `submit` is the only call the pipeline makes."""

    def __init__(self, store_path, cfg: AlertsConfig, sinks: Sequence, rooms: Optional[dict] = None,
                 clock: Callable[[], float] = time.time, start: bool = True,
                 data_label: str = "live",
                 jitter: Optional[Callable[[], float]] = None,
                 local_time: Callable[[float], datetime] = datetime.fromtimestamp):
        self.path, self.cfg, self.clock = store_path, cfg, clock
        self.sinks = {s.name: s for s in sinks}
        self.rooms, self.data_label = dict(rooms or {}), data_label
        self._jitter = jitter or (lambda: random.uniform(-0.1, 0.1))
        self._local_time = local_time
        self._q: queue.Queue = queue.Queue(maxsize=1000)
        self._wake, self._stop = threading.Event(), threading.Event()
        self._store: Optional[Store] = None
        self._warned: set = set()
        self._thread: Optional[threading.Thread] = None
        if start and cfg.enabled:
            self._thread = threading.Thread(target=self._run, name="carelens-alerts", daemon=True)
            self._thread.start()

    # -- producer side ----
    def submit(self, camera_id: str, event: dict) -> None:
        """Queue an event for alerting. Non-blocking; never raises."""
        try:
            if not self.cfg.enabled or event.get("kind") not in ALERT_KINDS:
                return
            self._q.put_nowait((camera_id, dict(event), self.clock()))
            self._wake.set()
        except Exception as e:   # includes queue.Full
            log.error("alert dropped (%s)", type(e).__name__)

    # -- worker side (one thread uses the store) ----
    def _db(self) -> Store:
        if self._store is None:
            self._store = Store(self.path)
            self._store.db.execute("PRAGMA busy_timeout=5000")
        return self._store

    def _ingest(self, camera_id: str, event: dict, now: float) -> None:
        st, kind = self._db(), event["kind"]
        event = {**event, "data_label": self.data_label}
        if self.cfg.dedupe_window_s > 0 and st.recent_alert_exists(
                camera_id, kind, now - self.cfg.dedupe_window_s):
            log.info("alert deduplicated (%s/%s)", camera_id, kind)
            return
        qh = self.cfg.quiet_hours.get(camera_id)
        if is_suppressed(kind, qh, self._local_time(now).time()):
            st.add_alert(camera_id, kind, event, now, "", "suppressed", last_error="quiet hours")
            return
        targets = [s for s in self.sinks.values() if kind in s.kinds]
        if not targets:   # still recorded, so the dashboard banner shows it
            st.add_alert(camera_id, kind, event, now, "", "suppressed",
                         last_error="no sink configured for this kind")
            return
        for s in targets:
            st.add_alert(camera_id, kind, event, now, s.name, "pending", next_attempt_ts=now)

    def _deliver(self, row: dict) -> bool:
        """Attempt one row. Returns True on a transient failure (the caller stops trying this
        sink for the rest of the pass)."""
        st, now = self._db(), self.clock()
        if not st.claim_alert(row["id"], now, CLAIM_LEASE_S):
            return False   # another process took it
        sink = self.sinks.get(row["sink"])
        if sink is None:
            st.mark_alert(row["id"], "failed", last_error="sink not configured")
            return False
        payload = build_payload(row, self.rooms.get(row["camera_id"], row["camera_id"]))
        attempts = row["attempts"]
        try:
            sink.send(payload)
        except Exception as e:
            err = redact_error(f"{type(e).__name__}: {e}", getattr(sink, "_secrets", lambda: [])())
            attempts += 1
            if isinstance(e, PermanentSinkError) or attempts >= self.cfg.max_attempts:
                st.mark_alert(row["id"], "failed", attempts, None, err)
                if (sink.name, err) not in self._warned:
                    self._warned.add((sink.name, err))
                    log.warning("alert sink %s failed: %s", sink.name, err)
            else:
                delay = backoff_delay(attempts, self.cfg.backoff_base_s, self.cfg.backoff_max_s,
                                      self._jitter())
                st.mark_alert(row["id"], "pending", attempts, self.clock() + delay, err)
                return True
            return False
        st.mark_alert(row["id"], "sent", attempts + 1, None, None)
        return False

    def drain_once(self) -> None:
        """Record queued events and attempt every due alert once. Call from one thread only."""
        try:
            while True:
                try:
                    cam, ev, ts = self._q.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._ingest(cam, ev, ts)
                except Exception as e:
                    log.error("alert could not be recorded (%s)", type(e).__name__)
            broken: set = set()   # sinks that failed transiently this pass: tried once per pass
            seen: set = set()
            while True:
                rows = [r for r in self._db().due_alerts(self.clock(), exclude_sinks=broken)
                        if r["id"] not in seen]
                if not rows:
                    break
                for row in rows:
                    if row["sink"] in broken:
                        continue
                    seen.add(row["id"])
                    try:
                        if self._deliver(row):
                            broken.add(row["sink"])
                    except Exception as e:
                        log.error("alert delivery bookkeeping failed (%s)", type(e).__name__)
        except Exception as e:
            log.error("alert worker error (%s)", type(e).__name__)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(1.0)
            self._wake.clear()
            self.drain_once()
        self.drain_once()   # final pass: record + try once
        self._close_store()

    def _close_store(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            finally:
                self._store = None

    def close(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        else:
            self._close_store()
