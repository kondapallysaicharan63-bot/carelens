"""Record consented test clips from a Logitech (or any) webcam, phone, or RTSP source.

Runs on your own laptop where the camera is plugged in. Each clip is saved at the
camera's native resolution with a burned-in wall-clock timestamp (so you can line it
up with the Apple Watch heart-rate export later), plus a sidecar .txt noting the start
time and source.

Examples:
    python -m carelens.record --name sitting_01   --seconds 30
    python -m carelens.record --name fall_01       --seconds 25 --preview
    python -m carelens.record --name walking_01    --seconds 20 --source 0

Then run the pipeline on what you recorded:
    python -m carelens.pipeline --source recordings/fall_01.mp4 --name fall_01 \
        --start "<the start time printed below>" --inactivity-min 1
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


def record(source, seconds, out_dir, name, fps_hint=None, preview=False):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera source {source!r}. "
                         "Is the webcam plugged in and not used by another app?")
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_hint or (cap_fps if cap_fps and cap_fps > 1 else 20.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    start_dt = datetime.now().astimezone().replace(microsecond=0)
    print(f"\nRecording '{name}'  ({w}x{h} @ {fps:.0f} fps, {seconds}s)")
    print(f"START TIME (use with --start): {start_dt.isoformat()}")
    print("Wear the Apple Watch and note this start time. Ctrl-C or 'q' to stop early.\n")

    t0 = time.monotonic()
    frames = 0
    try:
        while time.monotonic() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            elapsed = time.monotonic() - t0
            clock = (start_dt.timestamp() + elapsed)
            label = datetime.fromtimestamp(clock).strftime("%H:%M:%S")
            cv2.putText(frame, f"CONSENTED TEST CAPTURE  {label}  +{elapsed:4.1f}s",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frames += 1
            if preview:
                cv2.imshow(f"recording {name} - press q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Stopped early.")
    finally:
        cap.release()
        writer.release()
        if preview:
            cv2.destroyAllWindows()

    dur = frames / fps
    (out / f"{name}.txt").write_text(
        f"name: {name}\nsource: {source}\nstart_time: {start_dt.isoformat()}\n"
        f"resolution: {w}x{h}\nfps: {fps:.1f}\nframes: {frames}\nduration_s: {dur:.1f}\n"
        f"consent: obtained from everyone filmed\n")
    print(f"Saved {path}  ({frames} frames, {dur:.1f}s)")
    print(f"Next: python -m carelens.pipeline --source {path} --name {name} "
          f'--start "{start_dt.isoformat()}" --inactivity-min 1\n')
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record a consented test clip from a webcam/phone/RTSP")
    ap.add_argument("--source", default="0", help="webcam index (0), file, or rtsp:// URL")
    ap.add_argument("--name", required=True, help="clip name, e.g. sitting_01 / fall_01")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--fps", type=float, default=None, help="override if the camera misreports fps")
    ap.add_argument("--preview", action="store_true", help="show a live window (needs a display)")
    a = ap.parse_args(argv)
    record(a.source, a.seconds, a.out, a.name, fps_hint=a.fps, preview=a.preview)


if __name__ == "__main__":
    main()
