# 02 — Architecture

Everything runs **on the Jetson**. No component depends on the network except optional
caregiver alerting (LAN-local).

## 1. Data flow

```
          ┌─────────────┐
  camera →│ VideoSource │→ Frame(t, image)
          └─────────────┘
                 │
                 ▼
          ┌─────────────┐   (33,3) landmarks
          │ PoseProvider│──────────────┐
          └─────────────┘              │
                 │                      ▼
                 │              ┌──────────────────┐
                 │              │ FeatureExtractor │ BodyState
                 │              └──────────────────┘
                 │                      │
     ┌───────────┼──────────┬───────────┼───────────┬───────────────┐
     ▼           ▼          ▼           ▼           ▼               ▼
 FallDetector Inactivity ActivityTL  Gait   HeartRate(rPPG)  Breathing
     │           │          │           │           │               │
     └───────────┴──────────┴─────┬─────┴───────────┴───────────────┘
                                   ▼
                        ┌────────────────────┐
                        │  pipeline.run()    │→ results.json + event clips + SQLite store
                        └────────────────────┘
                                   │
                                   ▼
                        ┌────────────────────┐
                        │ Flask dashboard    │ (localhost) + optional alert sink
                        └────────────────────┘
```

## 2. Modules (current)

| Module | File | Responsibility |
|--------|------|----------------|
| Sources | `carelens/sources.py` | Open file/USB/CSI/RTSP, yield timestamped, resized frames |
| Pose | `carelens/pose.py` | `PoseProvider` protocol; MediaPipe + scripted (test) impls |
| Tracking | `carelens/tracking.py` | Single-person presence debounce, <=0.5 s dropout hold, One Euro keypoint smoothing (`Config.tracking=None` disables). Held frames carry forward the last measured state (motion included, so no fake stillness) and are kept out of the vitals windows. Centroid-jump reset is a heuristic, not identity tracking |
| Features | `carelens/features.py` | Per-frame posture, motion, angle, hip position |
| Events | `carelens/events.py` | Fall, inactivity, timeline, gait |
| Vitals | `carelens/vitals.py` | rPPG heart rate, breathing; shared signal core |
| Pipeline | `carelens/pipeline.py` | Wire it all; write JSON, clips, history |
| Synth | `carelens/synth.py` | Ground-truth synthetic video + scripted poses |
| Dashboard | `dashboard/` | Local web UI |

## 3. Key design decisions

- **One `PoseProvider` interface** so the backend (MediaPipe, or a TensorRT pose model
  on Jetson) is swappable without touching event/vitals logic. This is the main seam for
  on-device acceleration — see `03-TECH-STACK.md`.
- **Source-agnostic input.** The camera is one `VideoSource`; nothing downstream knows or
  cares whether it's a clip or a live RTSP stream.
- **Stateless-ish detectors.** Each detector consumes `BodyState` frames and emits events;
  they hold only small rolling windows, so the pipeline streams in bounded memory and runs
  indefinitely on-device.
- **Quality-gated vitals.** rPPG/breathing only emit when still + well-lit, and every
  estimate carries a quality score; the dashboard shows reported vs withheld windows. This
  keeps the "honest, not a medical device" promise visible.
- **SQLite store** (`carelens/store.py`, `output/carelens.db`) is the source of truth; daily rollups survive the raw purge. It replaced `history.jsonl`. A
  `_results.json` remains as a per-run export.

## 4. Deployment topology

- **Capture + inference + dashboard**: all on the Jetson, single Python process (plus the
  Flask dashboard process).
- **Camera link**: USB/CSI (wired to the board) or RTSP (wired PoE or Wi-Fi) on the local
  network. See `05-CAMERA-INTEGRATION.md`.
- **Caregiver access**: dashboard bound to localhost; for LAN access, front it with a
  reverse proxy + auth (see `04-SECURITY-PRIVACY.md`). No WAN exposure by default.

## 5. Concurrency model (target for on-device)

v0 is single-threaded for clips (`VideoSource`). Live sources use `LiveSource` (built, P2.6): a capture
thread feeds a drop-old buffer consumed by the inference loop (`run()`), with auto-reconnect and a
stall watchdog (see `05-CAMERA-INTEGRATION.md` section 4). The full target pipeline:

```
[capture thread] → frame queue (drop-old) → [inference thread] → state queue → [detectors+sinks]
```

Dropping stale frames under load keeps latency bounded. Define in build phase P2.
