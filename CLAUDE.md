# CLAUDE.md — working guide for Claude Code

Read this first. This repo is a **personal wellness** camera monitor (fall/inactivity +
contactless heart/breathing rate), deployed on a Jetson Orin Nano. It is **not a medical
device** and must never claim to be.

## Where things are
- `carelens/` — the pipeline (sources, pose, features, events, vitals, pipeline, synth).
- `dashboard/` — local Flask UI.
- `tests/` — pytest suite.
- `docs/` — the spec set. **Read `docs/07-BUILD-PLAN.md` to know what to do next.**
  P0, P1 and the hardware-free P2 work are built. **Main track is now the website:**
  `docs/10-WEB-PLAN.md` (W0–W11). The Jetson/edge phases in 07 are a later track.

## How to work
0. **Orchestrate, don't do everything yourself.** This session is the *coordinator*: dispatch
   subagents (Explore for search, Plan for design, general-purpose for implementation) and use
   skills, keeping your own context small. Full playbook: **`docs/09-AI-DEV-WORKFLOW.md`**.
1. Work **one build-plan task at a time**, in order. Each task has an acceptance criterion.
2. Keep `pytest -q` green after every change; add tests with each feature.
3. Prefer editing existing modules over new ones; keep the `PoseProvider`/`VideoSource`
   interfaces stable — they're the seams for on-device swaps.
4. Run the pipeline to sanity-check: `python -m carelens.pipeline --camera demo --inactivity-min 1 --truth-pose output/synthetic_demo.truth.json`
   (regenerate the clip with `python -m carelens.synth` if needed). `--truth-pose` feeds the generator's
   ground-truth poses — real MediaPipe does not detect the synthetic stick figure. Runs land in
   `output/carelens.db`; view them with `python dashboard/app.py`.

## Hard rules (do not violate)
- **Wellness wording only** — "trend", "check-in suggested". Never "diagnosis".
- **Consented or synthetic footage only.**
- **Cloud-app cameras (Roku/Blink/Nest) = clip-export only** via the vendor app. Use
  standards-based cameras (RTSP/ONVIF/USB/CSI) for live video.
- **Permissive licenses only** (MIT/Apache-2.0/BSD). **No secrets committed.** Local-first;
  no cloud in the runtime path.

## Conventions
- Python 3.10+, OpenCV/MediaPipe/NumPy/SciPy. On Jetson, use GStreamer-enabled OpenCV and
  a TensorRT pose provider for the live path (keep MediaPipe as the accuracy reference).
- Camera configs in `config.yaml` (pydantic-validated, see `carelens/config.py`); credentials from `.env` (never inline).
- `output/`, `*.mp4`, `.env` are git-ignored.

## Token & model discipline
- **Budget:** stop long agent runs at **80% of the weekly plan limit**; keep 20% in reserve.
  **Never spend extra-usage credits.** Check usage with `/cost` (or the `explain-usage` skill).
- **Pick the model by the task** (switch with `/model`):
  - **Haiku** — routine edits, boilerplate, docstrings, running tests/formatters, small fixes.
  - **Sonnet** — default for building features, writing modules and tests (most of the work).
  - **Opus** — only for the hard parts: architecture decisions, tricky debugging, the
    signal-processing math (rPPG/breathing filters), and security/privacy review.
- **Work cheap by default:** one build-plan task per session, small diffs, lean on the fast
  synthetic tests for feedback, and don't print big JSON into chat — write to files and assert.
- **Keep heavy work off tokens:** real-camera streaming, the 24h soak, the Jetson flash, and
  the Apple Watch export are hardware steps run locally — the agent plans and writes code, it
  doesn't push video through the model.

## Definition of done
Pipeline runs a clip end to end, flags the staged fall and inactivity, shows HR/BR with
quality scores, and the accuracy report (`docs/ACCURACY_REPORT_TEMPLATE.md`) is filled
honestly — including what does not work.
