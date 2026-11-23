"""End-to-end pipeline: video source -> pose -> features -> events, timeline, gait, vitals."""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .events import ActivityTimeline, FallDetector, GaitAnalyzer, InactivityDetector
from .features import FeatureExtractor
from .sources import VideoSource
from .vitals import BreathingRateEstimator, HeartRateEstimator

BANNER = "Simulated or test data - personal learning project, not a medical device"


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
        clip_name: str = "clip") -> dict:
    out = Path(out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    src = source if isinstance(source, VideoSource) else VideoSource(source, max_seconds=max_seconds)
    start_time = start_time or datetime.now().replace(microsecond=0)

    feats: Optional[FeatureExtractor] = None
    fall = FallDetector(still_seconds=cfg.fall_still_seconds, still_motion=cfg.still_motion)
    inact = InactivityDetector(seconds=cfg.inactivity_seconds, still_motion=cfg.still_motion)
    timeline = ActivityTimeline(start_time)
    gait = GaitAnalyzer(person_height_m=cfg.person_height_m)
    hr = HeartRateEstimator(min_quality=cfg.hr_min_quality, still_motion=cfg.still_motion)
    br = BreathingRateEstimator(min_quality=cfg.br_min_quality, still_motion=cfg.still_motion)
    buf = _ClipBuffer(cfg.clip_buffer_s)

    events, postures = [], []
    last_t = 0.0
    for fr in src:
        if feats is None:
            feats = FeatureExtractor(aspect=fr.image.shape[1] / fr.image.shape[0])
        lm = pose_provider.process(fr.image, fr.t)
        s = feats.update(fr.t, lm)
        buf.add(fr.t, fr.image)
        timeline.update(s)
        gait.update(s)
        hr.update(fr.t, fr.image, s)
        br.update(fr.t, fr.image, s)
        if fr.index % max(1, int(src.fps)) == 0:
            postures.append({"t": round(fr.t, 1), "posture": s.posture,
                             "motion": round(s.motion, 3)})
        for det in (fall, inact):
            ev = det.update(s)
            if ev:
                name = f"{clip_name}_{ev.kind}_{int(ev.t_onset)}s.mp4"
                ev.clip = buf.write(out / "clips" / name, ev.t_onset - cfg.clip_pre_s,
                                    ev.t_confirmed, src.fps, ev.kind.upper())
                events.append(ev.to_dict())
        last_t = fr.t

    results = {
        "banner": BANNER,
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
    # One line per run, so the dashboard can show trends over days.
    with open(out / "history.jsonl", "a") as fh:
        fh.write(json.dumps({
            "start_time": results["start_time"], "clip": clip_name,
            "data_label": cfg.data_label,
            "hr_median": results["heart_rate"]["summary"]["median"],
            "br_median": results["breathing_rate"]["summary"]["median"],
            "gait_speed": (float(np.median([g["speed_m_s_approx"] for g in results["gait_segments"]]))
                           if results["gait_segments"] else None),
            "n_falls": sum(e["kind"] == "fall" for e in events),
            "n_inactivity": sum(e["kind"] == "inactivity" for e in events),
        }) + "\n")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Camera health monitor (wellness trends only)")
    ap.add_argument("--source", required=True, help="video file, webcam index (0), or stream URL")
    ap.add_argument("--out", default="output")
    ap.add_argument("--name", default=None, help="name for this run (defaults to file stem)")
    ap.add_argument("--start", default=None, help="wall-clock start of clip, ISO format")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--inactivity-min", type=float, default=30.0)
    ap.add_argument("--fall-still-s", type=float, default=20.0)
    ap.add_argument("--height-m", type=float, default=1.65)
    ap.add_argument("--model", default=None, help="MediaPipe pose .task model path")
    a = ap.parse_args(argv)

    from .pose import MediaPipePose, DEFAULT_MODEL
    pose = MediaPipePose(a.model or DEFAULT_MODEL)
    cfg = Config(fall_still_seconds=a.fall_still_s, inactivity_seconds=a.inactivity_min * 60,
                 person_height_m=a.height_m)
    name = a.name or (Path(a.source).stem if not str(a.source).isdigit() else "webcam")
    start = datetime.fromisoformat(a.start) if a.start else None
    try:
        res = run(a.source, pose, a.out, cfg, start_time=start, max_seconds=a.max_seconds,
                  clip_name=name)
    finally:
        pose.close()
    print(json.dumps({"events": res["events"], "heart_rate": res["heart_rate"]["summary"],
                      "breathing_rate": res["breathing_rate"]["summary"],
                      "gait_segments": res["gait_segments"]}, indent=2))


if __name__ == "__main__":
    main()
