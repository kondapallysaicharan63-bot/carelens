# 06 — Testing Strategy

Five layers, cheapest and most deterministic first. The guiding rule: **synthetic data
proves the logic; real consented clips prove the system.** Never conflate the two.

## 1. Unit tests (fast, deterministic) — `tests/test_signal.py`, `tests/test_events.py`

- Signal core: band-pass + dominant-frequency recover known HR/BR; noise and out-of-band
  signals are rejected (low quality).
- Event logic on scripted landmarks: posture classification; a sudden fall is flagged; a
  slow lie-down and a quick recovery are **not**; inactivity fires once; timeline + gait.
- Run on every change: `pytest -q`. Target < 90 s.

## 2. End-to-end on synthetic video — `tests/test_end_to_end.py`

- Render a ground-truth clip (`carelens/synth.py`), run the full pipeline, assert:
  expected fall + inactivity events, HR/BR within tolerance, banner wording, dim-light
  rejection, empty-room = no events.
- These cover wiring + signal processing. **They do not validate real pose or real rPPG.**

## 3. Hardware-in-the-loop (on the Jetson) — new

- Run the live path from a USB/CSI/RTSP camera against a pre-recorded loop or a person.
- Assert: process starts, frames flow, no crash over a soak period, memory bounded.
- Measure **throughput (FPS)**, **end-to-end latency**, **CPU/GPU load**, **thermals**
  (`tegrastats`). Gate: ≥ 15 FPS @ 720p, stable temps over a 30-min soak.
- Reconnect test: pull the RTSP camera's network mid-run; assert auto-reconnect.

## 4. Accuracy validation (real, consented) — `docs/ACCURACY_REPORT_TEMPLATE.md`

- Record consented clips: sitting still, walking, **staged safe fall onto a mattress**,
  lying still, leaving the room. Good even lighting; subject roughly side-on for gait.
- **Heart rate**: wear the **Apple Watch Series 5** (main ref, HR+ECG), note each clip's
  start time, export watch HR from the iPhone Health app, align by time, report **mean
  absolute error in bpm**. Older rose-gold watch = second check.
- **Breathing**: count breaths by hand (or a chest-belt if available) as reference.
- **Falls**: hits / misses / false alarms across clips; record onset error.
- Record lighting, distance, and motion limits where each estimator breaks.
- Use public, properly licensed fall datasets where helpful; log each dataset + license.

## 5. Regression + CI

- A sample-clip test that renders each scenario and checks expected events (already in
  `test_end_to_end.py`) is the regression guard.
- CI: run layers 1–2 on every push (CPU-only, no camera). Layers 3–4 are manual/on-device
  and recorded in the accuracy report.

- Accuracy harness (`python -m carelens.accuracy`, tests in `tests/test_accuracy.py`):
  synthetic fixtures only; its end-to-end test checks alignment math, not real-world accuracy.

## 6. Test data policy

- Synthetic clips: generated, no real person — safe to commit if small, else regenerate.
- Real clips: **never commit**; keep off-repo; store consent records.
- `output/`, `*.mp4`, `accuracy/`, `export.xml`, `export.zip` are git-ignored.

## 7. Definition of done (per the brief)

The pipeline runs on a recorded clip end to end, flags the staged fall and the inactivity
case, shows heart and breathing estimates with quality scores, and the accuracy report is
written honestly — including what does **not** work.
