"""End-to-end pipeline: video source -> pose -> features -> events, timeline, gait, vitals."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import deque
from dataclasses import dataclass, asdict, field, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import AppConfig, StorageConfig, load_config, redact_url, resolve_source, runtime_env
from .events import ActivityTimeline, FallDetector, GaitAnalyzer, InactivityDetector
from .features import FeatureExtractor
from .sources import VideoSource
from .tracking import PresenceTracker, TrackingConfig
from .vitals import BreathingRateEstimator, HeartRateEstimator

BANNER = "Simulated or test data - personal learning project, not a medical device"
LIVE_BANNER = "Live camera data - personal learning project, not a medical device"
LIVE_KINDS = {"usb", "rtsp", "csi"}


@dataclass
class Config:
    fall_still_seconds: float = 20.0
    inactivity_seconds: float = 1800.0
    still_motion: float = 0.08
    person_height_m: float = 1.65
    clip_pre_s: float = 5.0
    clip_buffer_s: float = 45.0
    hr_min_quality: float = 0.35
    br_min_quality: float = 0.35
    data_label: str = "test"   # "test" | "simulated"
    tracking: Optional[TrackingConfig] = field(default_factory=TrackingConfig)  # None disables


class _ClipBuffer:
    """Small rolling buffer of JPEG-encoded, downscaled frames for event clips."""

    def __init__(self, seconds: float, width: int = 320):
        self.seconds, self.width = seconds, width
        self.frames: deque = deque()

    def add(self, t, image):
        h = int(image.shape[0] * self.width / image.shape[1])
        small = cv2.resize(image, (self.width, h))
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.frames.append((t, jpg))
        while self.frames and self.frames[0][0] < t - self.seconds:
            self.frames.popleft()

    def write(self, path: Path, t0: float, t1: float, fps: float, label: str) -> Optional[str]:
        sel = [(t, j) for t, j in self.frames if t0 <= t <= t1]
        if not sel:
            return None
        first = cv2.imdecode(sel[0][1], cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for t, j in sel:
            img = cv2.imdecode(j, cv2.IMREAD_COLOR)
            cv2.putText(img, f"{label}  t={t:5.1f}s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1, cv2.LINE_AA)
            vw.write(img)
        vw.release()
        return path.name


def _summary(estimates):
    good = [e for e in estimates if e["reported"]]
    if not good:
        return {"median": None, "n_reported": 0, "n_windows": len(estimates)}
    vals = np.array([e["value"] for e in good])
    return {"median": round(float(np.median(vals)), 1), "n_reported": len(good),
            "n_windows": len(estimates),
            "mean_quality": round(float(np.mean([e["quality"] for e in good])), 3)}


def run(source, pose_provider, out_dir: str | Path, cfg: Config = Config(),
        start_time: Optional[datetime] = None, max_seconds: Optional[float] = None,
        clip_name: str = "clip", store=None, camera_id: str = "default",
        source_kind: Optional[str] = None) -> dict:
    out = Path(out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    src = source if hasattr(source, "fps") else VideoSource(source, max_seconds=max_seconds)
    start_time = start_time or datetime.now().replace(microsecond=0)

    feats: Optional[FeatureExtractor] = None
    fall = FallDetector(still_seconds=cfg.fall_still_seconds, still_motion=cfg.still_motion)
    inact = InactivityDetector(seconds=cfg.inactivity_seconds, still_motion=cfg.still_motion)
    timeline = ActivityTimeline(start_time)
    gait = GaitAnalyzer(person_height_m=cfg.person_height_m)
    hr = HeartRateEstimator(min_quality=cfg.hr_min_quality, still_motion=cfg.still_motion)
    br = BreathingRateEstimator(min_quality=cfg.br_min_quality, still_motion=cfg.still_motion)
    buf = _ClipBuffer(cfg.clip_buffer_s)
    tracker = PresenceTracker(cfg.tracking) if cfg.tracking else None
    s = None

    events, postures = [], []
    last_t = 0.0
    for fr in src:
        if feats is None:
            feats = FeatureExtractor(aspect=fr.image.shape[1] / fr.image.shape[0])
        lm = pose_provider.process(fr.image, fr.t)
        held = False
        if tracker is not None:
            lm, held = tracker.update(fr.t, lm, fr.image.shape[1] / fr.image.shape[0])
        if held and s is not None:
            # Bridged dropout: carry forward the last *measured* state (incl. its motion) so a
            # held pose never counts as stillness; only the timestamp advances.
            s = replace(s, t=fr.t)
        else:
            s = feats.update(fr.t, lm)
        buf.add(fr.t, fr.image)
        timeline.update(s)
        if held:   # held poses are not measurements: reset vitals as an absent frame did pre-tracker
            hr._reset()
            br._reset()
        else:
            gait.update(s)
            hr.update(fr.t, fr.image, s)
            br.update(fr.t, fr.image, s)
        if fr.index % max(1, int(src.fps)) == 0:
            postures.append({"t": round(fr.t, 1), "posture": s.posture,
                             "motion": round(s.motion, 3)})
        for det in (fall, inact):
            ev = det.update(s, held=True) if (held and det is fall) else det.update(s)
            if ev:
                name = f"{clip_name}_{ev.kind}_{int(ev.t_onset)}s.mp4"
                ev.clip = buf.write(out / "clips" / name, ev.t_onset - cfg.clip_pre_s,
                                    ev.t_confirmed, src.fps, ev.kind.upper())
                events.append(ev.to_dict())
        last_t = fr.t

    results = {
        "banner": LIVE_BANNER if cfg.data_label == "live" else BANNER,
        "data_label": cfg.data_label,
        "source": str(getattr(src, "source", source)),
        "start_time": start_time.isoformat(),
        "duration_s": round(last_t, 1),
        "config": asdict(cfg),
        "events": events,
        "timeline_by_hour": timeline.to_dict(),
        "posture_track": postures,
        "gait_segments": gait.finish(),
        "heart_rate": {"unit": "bpm", "summary": _summary(hr.estimates), "windows": hr.estimates},
        "breathing_rate": {"unit": "breaths/min", "summary": _summary(br.estimates),
                           "windows": br.estimates},
    }
    (out / f"{clip_name}_results.json").write_text(json.dumps(results, indent=2))
    if store is not None:
        store.save_run(results, camera_id=camera_id, source_kind=source_kind or "file",
                       name=clip_name, export_path=str(out / f"{clip_name}_results.json"))
    return results


_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_frame(image: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate a frame clockwise by 0/90/180/270 degrees (0 is a no-op)."""
    return image if not orientation else cv2.rotate(image, _ROTATIONS[orientation])


class OrientedSource:
    """Wraps a VideoSource, rotating each frame; exposes a redacted `source` for logs/results.

    Rotation happens before downscaling so rotated cameras feed the same width as upright ones;
    open the wrapped VideoSource with ``resize_width=None``.
    """

    def __init__(self, src: VideoSource, orientation: int, label: str, resize_width: int | None = 640):
        self._src, self.orientation, self.source = src, orientation, label
        self.fps, self.resize_width = src.fps, resize_width

    def __iter__(self):
        for fr in self._src:
            img = rotate_frame(fr.image, self.orientation)
            if self.resize_width and img.shape[1] > self.resize_width:
                h = int(img.shape[0] * self.resize_width / img.shape[1])
                img = cv2.resize(img, (self.resize_width, h), interpolation=cv2.INTER_AREA)
            fr.image = img
            yield fr


def default_data_label(kind: str) -> str:
    """Live camera sources are 'live'; files and exported cloud clips stay 'test'."""
    return "live" if kind in LIVE_KINDS else "test"


def default_run_name(explicit, kind: str, ident: str, start: datetime, file_stem=None) -> str:
    """Explicit --name wins; file sources use the stem; live/camera runs get a unique
    `<id>_<YYYYmmdd-HHMMSS>` so repeated runs never share a name (or an export/clip path)."""
    if explicit:
        return explicit
    if kind == "file" and file_stem:
        return file_stem
    return f"{ident}_{start.strftime('%Y%m%d-%H%M%S')}"


def resolve_storage(config_path, out_arg=None):
    """Storage settings for --source runs: the config's storage block if the file exists,
    else defaults. Returns (StorageConfig, out_dir)."""
    p = Path(config_path)
    if p.is_file():
        st = load_config(p).storage
        return st, out_arg or str(p.resolve().parent / st.out_dir)
    return StorageConfig(), out_arg or "output"


def resolve_args(a, cfg: AppConfig, env=None, base_dir=None):
    """Merge CLI args with a configured camera. Returns (cam, source, Config, out_dir, name)."""
    cam = cfg.camera(a.camera)
    if not cam.enabled:
        raise ValueError(f"camera {cam.id!r} is disabled in the config (enabled: false)")
    th = cfg.thresholds_for(cam)
    inact = a.inactivity_min if a.inactivity_min is not None else th.inactivity_min
    fall = a.fall_still_s if a.fall_still_s is not None else th.fall_still_s
    height = a.height_m if a.height_m is not None else cam.person_height_m
    pcfg = Config(fall_still_seconds=fall, inactivity_seconds=inact * 60, person_height_m=height,
                  data_label=default_data_label(cam.type.value))
    out = a.out or str(Path(base_dir or ".") / cfg.storage.out_dir)
    return cam, resolve_source(cam, env, base_dir), pcfg, out, a.name or cam.id


def infer_source_kind(source) -> str:
    s = str(source).strip().lower()
    if s.isdigit():
        return "usb"
    if s.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    return "csi" if s.startswith("csi://") else "file"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Camera health monitor (wellness trends only)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="video file, webcam index (0), or stream URL")
    g.add_argument("--camera", help="camera id declared in the config file")
    ap.add_argument("--config", default="config.yaml", help="camera registry (with --camera)")
    ap.add_argument("--out", default=None, help="output dir (default: output, or storage.out_dir)")
    ap.add_argument("--name", default=None, help="name for this run (defaults to file stem)")
    ap.add_argument("--start", default=None, help="wall-clock start of clip, ISO format")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--inactivity-min", type=float, default=None)
    ap.add_argument("--fall-still-s", type=float, default=None)
    ap.add_argument("--height-m", type=float, default=None)
    ap.add_argument("--model", default=None, help="MediaPipe pose .task model path")
    ap.add_argument("--truth-pose", default=None, metavar="TRUTH_JSON",
                    help="synthetic clips only: use the generator's ground-truth poses instead of "
                         "MediaPipe (run is always labelled 'simulated')")
    a = ap.parse_args(argv)

    from .store import Store
    from .pose import MediaPipePose, DEFAULT_MODEL
    start = datetime.fromisoformat(a.start) if a.start else datetime.now().replace(microsecond=0)
    if a.truth_pose:
        from .synth import Scenario
        pose = Scenario.from_truth(a.truth_pose).provider()
    else:
        pose = MediaPipePose(a.model or DEFAULT_MODEL)  # before opening any camera, so a bad model leaks no stream
    cam = None
    if a.camera:
        base = Path(a.config).resolve().parent  # relative paths and .env are config-relative
        try:
            app_cfg = load_config(a.config)
            cam, source, cfg, out, name = resolve_args(a, app_cfg, runtime_env(base / ".env"), base)
        except (KeyError, ValueError, FileNotFoundError) as e:
            pose.close()
            raise SystemExit(f"error: {e.args[0] if isinstance(e, KeyError) else e}")
        storage, camera_id, kind = app_cfg.storage, cam.id, cam.type.value
        name = default_run_name(a.name, kind, cam.id, start)
    else:
        try:
            storage, out = resolve_storage(a.config, a.out)
        except (ValueError, FileNotFoundError) as e:
            pose.close()
            raise SystemExit(f"error: {e}")
        source = a.source
        kind = infer_source_kind(a.source)
        d = Config()
        cfg = Config(fall_still_seconds=d.fall_still_seconds if a.fall_still_s is None else a.fall_still_s,
                     inactivity_seconds=d.inactivity_seconds if a.inactivity_min is None else a.inactivity_min * 60,
                     person_height_m=d.person_height_m if a.height_m is None else a.height_m,
                     data_label=default_data_label(kind))
        stem = Path(a.source).stem if kind == "file" else None
        ident = {"usb": "webcam"}.get(kind, kind)
        name = default_run_name(a.name, kind, ident, start, stem)
        camera_id = "default"
    if a.truth_pose:
        if cam is not None and cam.orientation:
            raise SystemExit("error: --truth-pose cannot be combined with a rotated camera")
        cfg = replace(cfg, data_label="simulated")  # ground-truth poses are never real footage
    try:  # before any camera is opened, so a bad database leaks no stream
        store = Store(Path(out) / storage.db_path)
    except (OSError, sqlite3.Error, RuntimeError) as e:
        pose.close()
        raise SystemExit(f"error: could not open database {Path(out) / storage.db_path}: {e}")
    if cam is not None:
        label = redact_url(str(source))
        try:
            vs = VideoSource(source, max_seconds=a.max_seconds, resize_width=None,
                             rtsp_transport=cam.rtsp_transport, open_timeout_s=cam.open_timeout_s)
        except IOError:
            store.close()
            pose.close()
            raise SystemExit(f"error: could not open camera {cam.id!r} ({label})")
        source = OrientedSource(vs, cam.orientation, label)
    try:
        res = run(source, pose, out, cfg, start_time=start, max_seconds=a.max_seconds,
                  clip_name=name, store=store, camera_id=camera_id, source_kind=kind)
        try:
            keep = store.latest_run_id(camera_id, name)
            purged = store.purge(retention_days=storage.retention_days,
                                 clips_dir=Path(out) / "clips",
                                 rollup_retention_days=storage.rollup_retention_days,
                                 exclude_run_ids=[keep] if keep else [])
            if purged["kept_excluded"]:
                print(f"note: this run is older than the {storage.retention_days}-day retention "
                      "window and will be removed on the next purge", file=sys.stderr)
        except Exception as e:  # a purge problem must not hide a successful run
            print(f"warning: retention purge failed: {e}", file=sys.stderr)
    finally:
        store.close()
        pose.close()
    print(json.dumps({"events": res["events"], "heart_rate": res["heart_rate"]["summary"],
                      "breathing_rate": res["breathing_rate"]["summary"],
                      "gait_segments": res["gait_segments"]}, indent=2))


if __name__ == "__main__":
    main()
