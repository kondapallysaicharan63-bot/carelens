# CareLens — Product Requirements & Build Plan (PRD)

**Owner:** Sai Charan Kondapally · **Status:** v1 skeleton built; this PRD defines the path to a Jetson-deployed, multi-camera system · **Date:** 2026-11-22

> **Positioning:** a personal learning + portfolio project. **Not a medical device.** It
> produces wellness *trends* and "check-in suggested" events — never a diagnosis.
> Companion to the Sukhi pendant: the camera watches the room, the pendant is worn.

---

## 1. Problem & goal

Older adults living alone are at risk from falls and long periods of inactivity, and
family caregivers have no low-friction way to know "is everything normal today?" without
intrusive, always-watched video or body-worn devices the person forgets to wear.

**Goal:** a local-first, camera-based ambient monitor that runs on an edge device in the
home, detects falls and inactivity, estimates contactless heart rate, breathing rate, and
activity/gait trends, and surfaces them to a caregiver — with video never leaving the room.

**Success for the portfolio:** a self-contained demo running on a Jetson, an honest
accuracy report (including what fails), and a clean, tested, documented codebase that
reads as production-minded engineering.

## 2. Users

| User | Role | Needs |
|---|---|---|
| Monitored adult | Passive subject | Privacy, dignity, no device to wear, nothing leaves home |
| Family caregiver | Primary dashboard user | "Is today normal?", fall/inactivity alerts, trends, event clips |
| Developer (Sai Charan) | Builder/operator | Easy config, hardware swap, reproducible tests, honest metrics |
| (Later) elder-care facility | Multi-room operator | Many cameras, roles, retention policy |

## 3. Scope

**In scope (v1 → v3):** single home, one-to-several cameras, one monitored person per
camera, fall + inactivity events, rPPG HR + breathing, activity timeline + gait trends,
local dashboard with alerts, Jetson deployment, multi-camera + IP-camera discovery.

**Out of scope / non-goals:** medical diagnosis; cloud storage by default; emergency
services dispatch; multi-person-per-frame tracking (v1); facial recognition / identity;
audio capture. These stay explicitly out to keep the privacy and regulatory story clean.

## 4. Features

Priority: **P0** = v1 (mostly done) · **P1** = next · **P2** = later.

### 4.1 Camera & input layer

**Universal camera support — any camera works.** CareLens treats the camera as an
interchangeable input behind one `VideoSource` interface, so *any* camera is supported
through one of these classes — the goal is "bring whatever camera you have":

| Camera class | Examples | How it connects | Live? |
|---|---|---|---|
| USB / UVC webcam | Logitech, any webcam | `--source 0` | Live |
| RTSP / ONVIF IP camera (wired PoE or Wi-Fi) | Reolink, Amcrest, Tapo, Hikvision/Dahua OEM | `--source "rtsp://…"`; ONVIF auto-discovery | Live |
| CSI camera (edge) | RPi Cam v2 / IMX219 on Jetson | `--source "csi://0"` | Live |
| Cloud-app camera | **Roku 360 SE**, Blink, Wyze, Nest | export/download clips from the vendor app → file input | Clips |
| Video file | any recording or dataset | `--source clip.mp4` | Recorded |

Any camera that speaks **RTSP/ONVIF** connects live — the recommended live path.
**Cloud-app cameras (Roku, Blink, Nest) are supported as clip sources** via their normal app
export; live local streaming isn't available from them. So "any camera" is real: live for
standards-based cameras, clip-based for the cloud ones.

- **P0** One `VideoSource` interface for: exported clip/file, USB/UVC webcam, RTSP/HTTP
  stream, Jetson CSI (`csi://0`). *(built)*
- **P1** **Camera registry**: a YAML config of named cameras (id, room, type, URL/creds,
  orientation, person-height for gait scale), so cameras are declared, not hard-coded.
- **P1** **RTSP/ONVIF wired or wireless IP cameras** as first-class live sources, with
  **ONVIF WS-Discovery** to auto-find cameras on the LAN.
- **P0** **Cloud-app cameras (Roku 360 SE, Blink, Nest) — exported clips only.** Pull saved
  clips from the vendor app and feed them in as files. Live local streaming isn't available
  from these; use a standards-based camera (above) for live.
- **P2** Multi-camera: one worker per camera, shared dashboard, per-room views.

### 4.2 Perception
- **P0** Person detection + body pose keypoints (MediaPipe Pose). *(built)*
- **P1** Short-term tracking + presence debouncing to smooth dropouts.
- **P2** Jetson: swap MediaPipe for a **TensorRT**-accelerated pose model (or NVIDIA
  DeepStream pipeline) for higher fps and lower CPU.

### 4.3 Events & analytics
- **P0** Fall detection: velocity-gated sudden drop (torso angle / hip height) + stillness
  ≥ 20 s; emits timestamped event + short clip. *(built)*
- **P0** Inactivity alert: no meaningful movement for a configurable time while present. *(built)*
- **P0** Activity timeline (sit/stand/walk/lie per hour) + gait trend (speed, steadiness). *(built)*
- **P1** Per-person baselines and day-over-day trend deltas ("walking slower than usual").

### 4.4 Vitals (contactless)
- **P0** Heart rate (rPPG, green channel, 0.7–3 Hz) with quality score, stillness + light gating. *(built)*
- **P0** Breathing rate (chest motion, 0.1–0.5 Hz) with quality score. *(built)*
- **P1** Signal-quality dashboard + automatic suppression of low-confidence windows (built: reported flag; surface it).

### 4.5 Alerting
- **P1** Local notifications on fall/inactivity (dashboard banner + optional desktop/webhook).
- **P2** Caregiver escalation via the user's **own** channel (SMS/Telegram/email) — opt-in,
  keys supplied by the user, never a built-in cloud.
- **P1** Configurable thresholds per camera (fall stillness, inactivity minutes, quiet hours).

### 4.6 Dashboard & storage
- **P0** Local web dashboard: timeline, events + clips, HR/BR trends, cross-recording
  history, persistent "Simulated or test data" banner. *(built)*
- **P1** **Auth** (password/token) + bind to LAN; multi-camera/room view.
- **P1** **SQLite** store for events, vitals windows, and daily rollups (replaces the JSON/JSONL
  files for durable history + retention).
- **P1** **Retention policy**: auto-purge clips and raw windows after N days.

### 4.7 Config & ops
- **P1** Single `config.yaml` (pydantic-validated): cameras, thresholds, retention, dashboard.
- **P2** **systemd** service on Jetson; health/status endpoint; structured logging; auto-restart.

## 5. Architecture

```
            ┌─────────────┐   frames   ┌──────────────┐  keypoints  ┌───────────────┐
 camera ───▶│ CameraSource │──────────▶│  Perception   │───────────▶│  Features /    │
 (file/USB/ │  (per camera) │           │ (pose, track) │            │  State         │
  RTSP/CSI) └─────────────┘            └──────────────┘             └──────┬────────┘
                                                                           │
                        ┌──────────────────────────────────────────────────┼───────────────┐
                        ▼                         ▼                          ▼               ▼
                  FallDetector            InactivityDetector         HR / BR estimators   Timeline/Gait
                        └──────────── events + vitals + trends ──────────────┘               │
                                                 │                                           │
                                                 ▼                                           ▼
                                        ┌────────────────┐   read   ┌──────────────────────────┐
                                        │  Store (SQLite  │◀────────▶│  Dashboard + API (local)  │
                                        │  + clips)       │          │  + Alerting               │
                                        └────────────────┘          └──────────────────────────┘
```

- **One process/worker per camera**; a thin supervisor starts/stops them from the registry.
- **Edge-only**: no component makes an outbound internet call in normal operation.
- Current modules (`carelens/sources|pose|features|events|vitals|pipeline`) already map to
  the boxes above; P1/P2 add the registry, SQLite store, API/auth, and the Jetson perception swap.

## 6. Tech stack (all permissive licenses: MIT / Apache-2.0 / BSD)

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | matches existing code |
| Vision | OpenCV, MediaPipe Pose | → TensorRT / DeepStream on Jetson (P2) |
| Signal/math | NumPy, SciPy | band-pass + Welch spectrum (built) |
| IP cameras | `onvif-zeep` + WS-Discovery, OpenCV/GStreamer RTSP | wired or wireless |
| API + dashboard | FastAPI (API) + static JS, or keep Flask | FastAPI gives schema + auth cleanly |
| Storage | SQLite (+ SQLCipher optional) | events, vitals, rollups; clips on disk |
| Config | pydantic / `pydantic-settings` + YAML | validated, typed |
| Tests | pytest, hypothesis (optional) | synthetic ground truth |
| CI | GitHub Actions | lint + type-check + pytest on synthetic |
| Edge runtime | JetPack, GStreamer, systemd | Orin Nano target |
| Packaging | `pip` + optional Docker (Jetson) | reproducible |

## 7. Security & privacy design

**Threat model:** the feed is biometric/health-adjacent and the device sits on a home
network. Adversaries: a compromised LAN device, an over-broad cloud dependency, casual
access to the dashboard, and secrets leaking into the repo/logs.

**Controls:**
- **Local-first, no cloud by default.** Outbound network egress off (or allowlist-only);
  nothing uploads clips or vitals. Any escalation channel is opt-in with user-supplied keys.
- **Network isolation.** Recommend cameras on a separate VLAN/SSID; the app reaches them,
  they don't reach the internet.
- **Dashboard access control.** Bind to localhost or LAN with auth (token/password), TLS
  optional; roles: caregiver (view) vs admin (configure).
- **Consent.** Per-monitored-person recorded consent; persistent on-screen test/live banner;
  no audio; no identity/face recognition.
- **Data at rest.** Optional encryption for clips + SQLite (SQLCipher / LUKS); configurable
  retention with auto-purge; no raw video retained beyond event clips.
- **Secrets.** RTSP credentials and any API keys via env/OS keyring, never committed; `.env`
  git-ignored; creds encrypted in the registry.
- **Supply chain.** Pinned dependencies, permissive licenses only, periodic `pip-audit`.
- **Logging.** No PII or raw frames in logs; access to the dashboard audit-logged.
- **Cloud-app cameras.** Used via the vendor app's normal clip export only.

## 8. Testing strategy (full-developer)

| Layer | What | How |
|---|---|---|
| **Unit** | band-pass + dominant-frequency, geometry, posture, fall/inactivity logic | pytest on synthetic ground truth *(built: 23 tests)* |
| **Integration** | render synthetic clip → full pipeline → assert events/vitals/banner | `test_end_to_end.py` *(built)* |
| **Golden/regression** | fixed sample clips → expected event JSON | freeze outputs; diff on change |
| **Hardware-in-the-loop** | real USB / RTSP / CSI camera opens, streams, hits target fps | manual + a `--selftest` mode; run on device, not in-agent |
| **Accuracy validation** | HR vs Apple Watch Series 5; staged falls; lighting/motion matrix | the accuracy report (`docs/ACCURACY_REPORT_TEMPLATE.md`) |
| **Performance / soak** | fps, CPU/GPU, memory over 24 h, thermal on Jetson | measure on Orin Nano; log + chart |
| **Security** | egress blocked, auth enforced, no creds in logs/repo | automated checks + `pip-audit` in CI |
| **CI** | lint (ruff) + types (mypy) + pytest (synthetic only, no hardware) | GitHub Actions on every push |

**Principle:** synthetic clips are the fast feedback loop and run in CI; anything needing a
real camera or the watch runs on hardware and is recorded in the accuracy report.

## 9. Jetson deployment (Orin Nano)

1. Flash JetPack; install the JetPack OpenCV (GStreamer + CUDA).
2. CSI camera via `nvarguscamerasrc` (`csi://0`, already wired) or any USB/RTSP camera.
3. P2: replace MediaPipe pose with a **TensorRT** engine (or DeepStream) for fps/power.
4. Run as a **systemd** service; health endpoint; auto-restart; logs to journald.
5. Target (to validate, not assume): ~2–4 cameras at ~10–15 fps on Orin Nano with the
   lite pose model; confirm in the perf/soak test and record real numbers.

## 10. Build roadmap (phases)

- **Phase 0 — v1 skeleton. ✅ DONE.** File/webcam/RTSP/CSI input, pose, fall + inactivity,
  timeline, gait, rPPG HR + breathing, local dashboard, synthetic generator, 23 passing tests.
- **Phase 1 — Harden the core.** Camera registry + `config.yaml`; RTSP/ONVIF discovery;
  SQLite store + retention; presence debouncing/tracking; dashboard auth; surface quality flags.
- **Phase 2 — Multi-camera + alerting.** Per-camera workers + supervisor; per-room dashboard;
  local alerts + opt-in escalation; thresholds per camera; golden regression set.
- **Phase 3 — Jetson port.** TensorRT/GStreamer perception; systemd service; perf + 24 h soak;
  record real fps/power/thermal.
- **Phase 4 — Validate + present.** Real-camera accuracy runs (Roku clips first, then an IP
  camera, then CSI) + Apple Watch HR reference; fill the accuracy report; 2-min demo; 1-page
  portfolio summary.

## 11. How to build this efficiently in Claude Code (token-aware plan)

Working budget: **stop long agent runs at 80% of the weekly plan limit; no extra-usage credits.**

1. **Build on the existing repo** — don't regenerate Phase 0. Import this folder into Claude Code.
2. **Test-driven, small diffs.** For each module: write/extend the pytest first, implement,
   run `pytest -q`, commit. Synthetic tests are the cheap, fast signal — lean on them.
3. **One phase per working session**, each ending at a green test suite and a commit, so no
   session has to re-read a half-finished state. Use a task list per phase.
4. **Keep heavy work off the agent.** Real-camera streaming, the 24 h soak, the Jetson flash,
   and the watch export are **hardware steps you run locally** — the agent plans and writes
   code/tests, it doesn't stream video through tokens.
5. **Cache models and deps.** The pose model is already vendored in `models/`; don't re-download.
6. **Don't page large outputs.** Write results to files and assert on them in tests rather than
   printing big JSON into the conversation.
7. **Scope each request.** Ask for one module/behavior at a time with its test, not "build it all,"
   so each turn is small, reviewable, and cheap.

## 12. Open questions / assumptions

- **Cameras:** assume 1–4 cameras, one monitored person each, for v1–v2. Confirm the real count.
- **Alert channel:** assumed opt-in, user-supplied (no built-in cloud). Confirm preferred channel.
- **Deployment target:** "Jetson Nano" — confirm exact board (Orin Nano vs legacy Nano); TensorRT
  path and fps targets depend on it.
- **Encryption at rest:** on by default or optional? Assumed optional for v1, recommended for any
  real deployment.
- **Retention window:** default assumed 7 days for clips; confirm.

## 13. Predictive & early-warning analytics (P2)

The detectors above are *reactive* (a fall just happened). The higher-value layer is
*predictive trends* — spotting that something is drifting before an incident. Framed as
wellness trends and "check-in suggested", **never a diagnosis or a clinical prediction.**

- **Personal baselines.** Learn each person's normal for gait speed, gait steadiness,
  hours active, nighttime restlessness, resting HR/BR, and time-on-floor — per hour/day.
- **Deviation alerts.** Flag a *sustained* drift from baseline (e.g. gait speed down ~15%
  over two weeks, rising nighttime movement, resting HR trending up) as an early check-in
  signal. Use simple, **explainable** methods first — rolling baseline + EWMA / z-score
  anomaly — not a black box, so the signal can always be explained and stays inside the
  no-diagnosis line.
- **Fall-risk trend (not fall prediction).** Declining gait steadiness / increased trunk
  sway → a rising fall-*risk* indicator over days. We do **not** claim to predict a specific
  fall seconds ahead — that would be an over-claim.
- **Explicitly out:** any clinical/diagnostic prediction, or presenting a trend as a medical
  forecast.

## 14. Senior-review gap list (what a pro would add before "production")

Honest review of what the reactive v1 spec doesn't yet cover:

- **Reliability & observability.** RTSP reconnect with backoff; frame backpressure when
  processing falls behind; a watchdog + auto-restart; structured logs and metrics (fps, drop
  rate, camera-up/down); **self-alerts** when a camera goes offline or the pipeline dies —
  a monitor that silently stops is worse than none.
- **Calibration.** Per-camera floor geometry + person height so gait is in real units;
  a one-time calibration step; auto-flag when the camera is bumped/moved and needs redoing.
- **Fairness & honest limits (important).** **rPPG accuracy is known to vary with skin tone,
  lighting, and motion.** Validate across skin tones and lighting, publish per-condition
  error, and never over-claim. This is both an engineering and an ethics requirement for a
  health-adjacent camera product.
- **Hard edge cases to handle or declare.** Sleeping vs inactivity vs fall; floor exercise /
  yoga as false falls; wheelchair / walker users (the posture model's standing/sitting
  assumptions break); pets and visitors / multiple people in frame; night / IR lighting;
  a bumped camera.
- **Data & ops.** Explicit SQLite schema + migrations; history backup/restore; edge OTA
  update + rollback; device provisioning; scheduled retention/purge.
- **Alert reliability.** Store-and-forward if the network is down; a local fallback; debounce
  and dedupe; quiet hours — a missed fall alert is the worst failure mode.
- **Safety & liability framing.** Persistent "not a medical device / not for emergencies" in
  the UI, and a `LIMITATIONS.md` stating what it cannot do.
- **Testing depth.** A coverage target; CI (lint + type-check + tests); soak/perf numbers on
  the Jetson; an adversarial lighting + skin-tone validation set; golden-output regression.

## 15. Final sweep — dignity, trust & fusion (last additions)

Items a thorough review flags that weren't yet captured:

- **Privacy & dignity (essential for elder care).** **Exclusion zones** — never process
  certain areas (bathroom, bedroom); a person-controlled **privacy / pause mode**; and a
  clear **consent lifecycle** for the monitored person — they know they're monitored, can
  pause it, and can request data deletion. Dignity is a feature, not an afterthought.
- **False-alarm feedback loop.** Let a caregiver mark an event "not a fall"; thresholds adapt
  and the system tracks **false-alarms per day**. Alert fatigue is how these products fail in
  the real world.
- **Sensor fusion with the Sukhi pendant.** The camera and the worn pendant **corroborate** a
  fall → a higher-confidence alert and fewer false alarms. This makes the "camera + pendant
  companion" story concrete and is a genuine differentiator.
- **Metric definitions (so accuracy is honest and comparable).** Fall detection =
  **sensitivity / specificity / false-alarms-per-day / PPV**; vitals = **mean absolute error
  + coverage** (fraction of windows confidently reported). Define these in the accuracy report.
- **Onboarding & BOM.** First-run flow: place the camera, calibrate floor geometry + person
  height, and a baseline-learning window before trends mean anything. A hardware
  **bill-of-materials + cost** (camera + Jetson) for the elder-care deployment — useful for the
  pitch, not just the build.

With sections 1–15, the spec is complete and pro-grade. What remains is **execution** (build
Phases 1–4) and, for the hackathon specifically, the **demo video + public repo** — not
missing *design*.
