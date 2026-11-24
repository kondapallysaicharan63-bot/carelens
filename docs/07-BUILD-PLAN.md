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

## P1 — Harden the core (DONE)

Delivered as four tasks (PRD §10 Phase 1), each committed with tests (91 passing):
1. **Camera registry** — pydantic-validated `config.yaml` (id, room, type, source, creds via
   env-var names, orientation, person height, per-camera thresholds); `--camera <id>`. ✅
2. **RTSP/ONVIF** — RTSP transport/timeout per camera; stdlib ONVIF WS-Discovery +
   GetStreamUri (`python -m carelens.discovery`), hardened against hostile LAN devices;
   loopback RTSP test. ✅
3. **SQLite store** — migrations via `PRAGMA user_version`; runs/events/vitals/rollups;
   whole-day retention purge with frozen rollups; dashboard reads SQLite. ✅
4. **Presence debouncing / tracking** — gap fill, enter hysteresis, One Euro smoothing;
   held frames never fake stillness or a fall; fall detection verified at 3–15 fps. ✅

*Phase 1 acceptance:* `python -m carelens.pipeline --camera demo --truth-pose
output/synthetic_demo.truth.json` writes the fall + inactivity + HR/BR run to
`output/carelens.db` and the dashboard shows it (`tests/test_phase1_e2e.py`). ✅

**Carried forward from the old P1 list** (not yet built): caregiver alerting sink (F10),
encrypted clip store (F11), dashboard trend charts over rollups (F14), accuracy harness (F15).

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
- **Follow-up (from P2.6):** live runs persist only at run end — add periodic segment
  rotation/checkpointing (e.g. hourly) so a long-running service writes to SQLite continuously.
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
