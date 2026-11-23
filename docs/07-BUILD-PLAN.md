# 07 — Build Plan (Claude Code executable)

This is the roadmap to drive from Claude Code. Phases are ordered; each task has an
**acceptance criterion** so Claude Code knows when it's done. P0 phase is already built —
start Claude Code at **P1**.

How to run with Claude Code: open this repo, read `CLAUDE.md` first, then work one phase
at a time, keeping `pytest -q` green after every task. Don't batch unrelated changes.

---

## P0 — Skeleton (DONE)

Built and passing 23 tests on synthetic data:
video input (file/USB/CSI/RTSP) · pose · fall · inactivity · timeline · gait · rPPG HR ·
breathing · pipeline · local dashboard · synthetic generator · tests.
Acceptance: `pytest -q` green; `python -m carelens.pipeline --source output/synthetic_demo.mp4`
flags the fall + inactivity and reports HR/BR. ✅

## P1 — Productionize the core (host, no hardware)

1. **Config system** — add `cameras.yaml` + `.env`/`.env.example`; load camera defs and
   RTSP creds from env. *Accept:* pipeline can be launched from a named camera in
   `cameras.yaml`; no secrets in source; `.gitignore` covers `.env`, `output/`, `*.mp4`.
2. **Caregiver alerting (F10)** — pluggable alert sink: event → local notification +
   optional LAN webhook (JSON POST). *Accept:* a fall event triggers the configured sink;
   unit test with a fake sink; failures are logged, never crash the pipeline.
3. **Encrypted clip store + retention (F11)** — encrypt event clips at rest; auto-purge
   clips older than N days (config). *Accept:* clips unreadable without the key; a purge
   routine deletes expired clips and is unit-tested with faked timestamps.
4. **Trend persistence (F14)** — keep `history.jsonl`; add a daily rollup + a dashboard
   trend view across recordings. *Accept:* dashboard shows HR/BR/gait/fall counts over
   multiple runs (the history table already exists — extend to charts).
5. **Accuracy harness (F15)** — a CLI that ingests an Apple Health HR export + a clip's
   results JSON, aligns by time, and emits MAE in bpm into the report template.
   *Accept:* given a watch CSV + results JSON, prints per-window error and overall MAE.

## P2 — Live capture + on-device performance

6. **Robust live `VideoSource`** — auto-reconnect with backoff, stall watchdog, drop-old
   frame queue, threaded capture→inference. *Accept:* a hardware/loopback test survives a
   mid-run RTSP disconnect and keeps latency bounded.
7. **TensorRT pose provider** — add a `PoseProvider` backed by `trt_pose` or
   YOLOv8-pose→TensorRT; keep MediaPipe as the accuracy reference. *Accept:* same landmark
   contract; benchmark both on Jetson; TensorRT path ≥ 15 FPS @ 720p.
8. **Performance + thermal harness** — measure FPS, latency, CPU/GPU, temps via
   `tegrastats`; 30-min soak. *Accept:* report meets the PRD throughput/thermal targets.

## P3 — Jetson deployment

9. **Device setup** — JetPack 6 deps, GStreamer-OpenCV, `requirements-jetson.txt` pinned
   to the board. *Accept:* `pytest -q` (layers 1–2) green on the Jetson.
10. **Autostart service** — `systemd` unit runs the pipeline on boot for a configured
    camera; dashboard service on localhost. *Accept:* reboot → pipeline + dashboard come
    up; logs to journald.
11. **Camera bring-up** — validate CSI, USB, and one RTSP camera end to end on-device.
    *Accept:* each source produces events + vitals on a live/looped feed.

## P4 — Validation + portfolio

12. **Real consented accuracy run** — record the clip set, run the harness, fill
    `docs/ACCURACY_REPORT_TEMPLATE.md` honestly (incl. failures/limits).
13. **2-minute demo video + one-page summary** — write after real numbers exist.
    *Accept:* demo shows a live fall flag + vitals with quality scores and the test-data
    banner; summary states measured accuracy and limits plainly.

## P5 — Stretch (v2)

Camera auto-discovery (ONVIF, F13) · PTZ control (ONVIF) · multi-camera ·
SQLite trend store · multi-person.

---

### Guardrails for every phase
- Keep `pytest -q` green. Add tests with each feature.
- Wellness wording only; keep the test-data banner.
- Cloud-app cameras (Roku/Blink/Nest) = clip-export only via the vendor app.
- Permissive licenses only; no secrets committed; local-first.
