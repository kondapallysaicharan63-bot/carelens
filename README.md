# CareLens web (main track)

CareLens runs as a **static website that works entirely on your device**: it uses your phone or
laptop camera and microphone to show wellbeing *trends* (heart rate, breathing, presence, and
experimental extras). The Python/Jetson pipeline further down is the reference implementation and
a later, optional edge track (see `docs/10-WEB-PLAN.md`). Status: **W0a scaffold** (consent flow,
device picker, shell UI, live preview, MediaPipe pose smoke test). Measurements come in W1+.

> **Not a medical device. For demonstration and wellbeing trends only. Do not use for diagnosis or
> emergencies; call emergency services if you feel unwell.**

## Run locally

```bash
cd web
npm ci
npm run fetch-models   # downloads models into web/public/models with pinned SHA-256 checks
npm run dev            # HTTPS dev server; prints Local and Network (LAN) https URLs
npm test               # unit tests (Vitest)
npm run typecheck
npm run build          # static site -> web/dist/ (also copies MediaPipe wasm into the bundle)
npm run preview        # serve web/dist over HTTPS on the LAN
```

**Phone on the same Wi-Fi:** run `npm run dev` (or `npm run preview`), open the `Network:` https URL
on the phone. Browsers only allow camera/microphone on HTTPS (or localhost). The dev certificate is
self-signed (`@vitejs/plugin-basic-ssl`), so accept the warning once. For a trusted certificate,
install [mkcert](https://github.com/FiloSottile/mkcert), run `mkcert -install` and
`mkcert localhost <your-LAN-IP>`, and point Vite at the files via `server.https` in
`web/vite.config.ts` (drop the basic-ssl plugin). Some laptops expose a black webcam as the first
device; the app detects black frames and offers the next camera.

## Deploy

The build output is fully static: build command `npm run build` (run in `web/`), output directory
`web/dist`. Run `npm run fetch-models` first on the build machine (models are not committed).

- **GitHub Pages** (served under `/carelens/`): `CARELENS_BASE=/carelens/ npm run build`, publish `web/dist`.
- **Cloudflare Pages:** root directory `web`, build `npm ci && npm run fetch-models && npm run build`, output `dist` (base `/`).
- **Netlify:** base directory `web`, build command as above, publish directory `web/dist` (base `/`).

`CARELENS_BASE` sets the Vite `base` (default `/`). Static hosts serve `.wasm` as `application/wasm`.

## Tiers

- **Results (Tier 1):** features meant to be reasonably dependable, each with a signal-quality meter
  (weak / ok / good, shown as text and shape, not colour alone). Below the quality gate no number is shown.
- **Experimental (Tier 2):** early features, clearly labelled, less reliable; treat as curiosity.
- **Not offered** (explained in the app): SpO2, blood pressure, glucose, temperature, any disease or
  diagnosis, emotion/mental-health scores, stroke/facial-asymmetry screening, age or identity.

## Nothing leaves your device

No uploads, accounts or analytics. This is enforced, not just promised: `web/index.html` carries a
Content-Security-Policy with `default-src 'self'` and `connect-src 'self'`, so the page cannot send
data to any other origin, and all models and wasm are self-hosted (nothing loads from a CDN).
Nothing is stored by default; only your theme and last-chosen device IDs are remembered in
`localStorage`. CSP details: `script-src 'self' 'wasm-unsafe-eval'` is required because MediaPipe
compiles WebAssembly (verified: without it the pose model fails to initialise; `'unsafe-eval'` is
not needed). `worker-src 'self' blob:` is kept for the upcoming Web Worker/AudioWorklet code (not
needed by the pose smoke test). Styles are external, so no `'unsafe-inline'` in production (dev
server only, for Vite's HMR). A test checks the built `index.html` for the CSP and external URLs.

## Licences of everything we ship or build with

| Component | Use | Licence |
|---|---|---|
| MediaPipe Tasks Vision `@mediapipe/tasks-vision` 1.2.3 (+ wasm) | pose (now), face/hand (later) | Apache-2.0 |
| MediaPipe Tasks Audio `@mediapipe/tasks-audio` 1.2.3 (+ wasm) | sound events (later) | Apache-2.0 |
| `pose_landmarker_lite.task`, `face_landmarker.task`, `hand_landmarker.task` | MediaPipe model bundles | Apache-2.0 |
| `yamnet.tflite` (audio_classifier/yamnet/float32/1) | sound-event classifier | Apache-2.0 |
| Vite 7 / Vitest 5 | build / unit tests | MIT |
| TypeScript 7, `@types/node` | typecheck | Apache-2.0 / MIT |
| `@vitejs/plugin-basic-ssl` | local HTTPS dev | MIT |
| Playwright (external, not a dependency yet) | browser end-to-end checks | Apache-2.0 |
| MediaPipe Pose (Python), OpenCV, NumPy, SciPy | Python reference pipeline | Apache-2.0 / Apache-2.0 / BSD / BSD |
| Planned: vosk-browser + small English model | optional "help" keyword (off by default) | Apache-2.0 |
| **Excluded:** Ultralytics YOLO (AGPL-3.0), any GPL code | | |

npm licences in `web/package-lock.json`: MIT, Apache-2.0, plus two build-time-only packages that
Vite's PostCSS pulls in: `picocolors` (ISC) and `source-map-js` (BSD-3-Clause). Both are dev
dependencies (`npm ls --omit=dev` is empty) and are not in the shipped bundle. The only third-party
code in the shipped JS is `@mediapipe/tasks-vision` (Apache-2.0, bundled at build time), plus the
self-hosted MediaPipe wasm and model files. Only the wasm files MediaPipe actually requests are
copied into `dist` (`*_wasm_internal.{js,wasm}`). Model files are
fetched by `npm run fetch-models` and verified against pinned SHA-256 checksums.

---

# CareLens

A fall-and-inactivity monitor plus contactless heart-rate and breathing-rate
estimation from an indoor camera. **Personal learning and portfolio project — not
a medical device.** It produces wellness *trends* and "check-in suggested" events,
never a diagnosis.

Companion to the Sukhi pendant idea: the camera watches the room, the pendant is
worn on the body.

## Planning docs (read these to build)

Full engineering spec set in `docs/`, and `CLAUDE.md` is the working guide for Claude Code:

| Doc | Covers |
|-----|--------|
| `CLAUDE.md` | How Claude Code should work in this repo; hard rules |
| `docs/01-PRD.md` | Product requirements, features, success metrics, constraints |
| `docs/02-ARCHITECTURE.md` | Data flow, modules, design decisions |
| `docs/03-TECH-STACK.md` | Libraries, Jetson Orin Nano specifics, TensorRT |
| `docs/04-SECURITY-PRIVACY.md` | Threat model, on-device/local-first, encryption, consent |
| `docs/05-CAMERA-INTEGRATION.md` | Camera matrix (USB/CSI/RTSP), wired vs wireless |
| `docs/06-TESTING.md` | Unit → end-to-end → hardware-in-loop → accuracy validation |
| `docs/07-BUILD-PLAN.md` | **Phased, Claude-Code-executable roadmap — start at P1** |
| `docs/08-DEPLOYMENT-JETSON.md` | JetPack setup, camera bring-up, systemd, perf/thermal |
| `docs/ACCURACY_REPORT_TEMPLATE.md` | Honest accuracy report to fill with real numbers |

## What it does

| Feature | How |
|---|---|
| Video input | One interface for a file, a USB webcam, an RTSP/HTTP stream, or a Jetson CSI camera (`csi://0`) |
| Person detection & pose | MediaPipe Pose Landmarker (Apache-2.0) |
| Fall detection | Sudden drop in torso angle or hip height (velocity-gated), then stillness for 20 s+. Emits an event with timestamp and a short clip |
| Inactivity alert | No meaningful movement for a configurable time while the person is present |
| Activity timeline | Seconds sitting / standing / walking / lying, bucketed per hour |
| Gait metrics | Approximate walking speed and steadiness per room crossing, as a trend over days |
| Heart rate (rPPG) | Green channel of a face skin patch, band-passed 0.7–3 Hz, only when still & well lit, with a quality score |
| Breathing rate | Chest-region vertical motion, band-passed 0.1–0.7 Hz (6–42 breaths/min), when still, with a quality score |
| Dashboard | Local web page (localhost only) with timeline, events, trends, and a "Simulated or test data" banner |

## Install

```bash
python -m pip install -r requirements.txt
# The MediaPipe pose model is included at models/pose_landmarker_lite.task
```

## Run

```bash
# 1. Make a synthetic clip with known ground truth (no real person filmed)
python -m carelens.synth --out output/synthetic_demo.mp4 --hr 72 --br 15

# 2. Run the full pipeline on any source
python -m carelens.pipeline --source output/synthetic_demo.mp4 --name demo --inactivity-min 1
python -m carelens.pipeline --source 0                 # USB webcam
python -m carelens.pipeline --source clip_from_roku_app.mp4   # exported Roku clip
python -m carelens.pipeline --source "csi://0"         # Jetson CSI camera

# 3. View results
python dashboard/app.py --out output      # then open http://127.0.0.1:5000
```

Each run writes `output/<name>_results.json`, any event clips under `output/clips/`,
and stores every run in the SQLite database `output/carelens.db` (the dashboard reads only
this; `_results.json` is a per-run export for the accuracy harness). Raw runs and clips are
purged after `storage.retention_days` (default 7; purge runs after each pipeline run, or
`python -m carelens.store purge --out output --days N`). Purging works in whole local days. Phase-0 data can be imported once with
`python -m carelens.store import-legacy --out output` (not automatic). Daily rollups (HR/BR/gait/event
counts, no imagery) survive the purge for `storage.rollup_retention_days` (default 365).

## Recording test clips (Logitech / any webcam)

Capture consented clips on the laptop the camera is plugged into. Each clip gets a
burned-in wall-clock timestamp (for Apple Watch alignment) and a sidecar `.txt`.

```bash
# Plug in the Logitech webcam, then record each scenario (wear the Series 5):
python -m carelens.record --name sitting_01  --seconds 30 --preview
python -m carelens.record --name walking_01  --seconds 20 --preview
python -m carelens.record --name fall_01     --seconds 25 --preview   # staged fall onto a mattress
python -m carelens.record --name lying_01    --seconds 30 --preview
python -m carelens.record --name leaving_01  --seconds 15 --preview

# Then analyze each (the record step prints the exact --start time to paste in):
python -m carelens.pipeline --source recordings/fall_01.mp4 --name fall_01 \
    --start "<printed start time>" --inactivity-min 1 --height-m 1.75

python dashboard/app.py --out output      # review at http://127.0.0.1:5000
```

`--source 0` is the default (first webcam); pass `--source 1` if the Logitech shows up
as the second device, or an `rtsp://…` URL for an IP camera later.

## Camera options

The pipeline is source-agnostic; the camera is just one input. Supported paths:

1. **USB webcam** (`--source 0`) — live video you fully control, best for development.
2. **Any RTSP / ONVIF camera** (`--source "rtsp://user:pass@host:554/stream"`) — a
   first-class live source, wired (PoE) or wireless; ONVIF auto-discovery finds them on the LAN.
   Find cameras with `python -m carelens.discovery [--timeout 3] [--stream-uri]` (ONVIF
   WS-Discovery, LAN only); it prints `config.yaml` snippets (credentials via `RTSP_USER` /
   `RTSP_PASS` in `.env`, never inline). Per-camera `rtsp_transport` (tcp/udp) and `open_timeout_s`.
3. **NVIDIA Jetson + CSI or USB camera** (`--source "csi://0"`) — recommended for a live,
   self-contained on-device demo. See below.
4. **Cloud-app cameras (Roku 360 SE, Blink, Nest) — exported clips only.** Download saved
   clips from the vendor app and feed them in as files. Live local streaming isn't available
   from these; use options 1–3 for live.

## Running on the Jetson (Orin Nano / Orin NX dev kit)

The whole pipeline can run on the board, so **video never leaves the room** — a strong
privacy point for the portfolio.

- Use the JetPack build of OpenCV (built with GStreamer and CUDA).
- Connect a CSI camera to CAM0/CAM1 (e.g. Raspberry Pi Camera v2 / IMX219) and use
  `--source "csi://0"`, or any USB webcam with `--source 0`.
- The CSI pipeline is in `carelens/sources.py:jetson_csi_pipeline()`; adjust sensor
  resolution/fps there if needed.

## Safety, privacy, and wording

- **Wellness wording only** — "trend", "check-in suggested". Never "diagnosis".
- **Consented or synthetic footage only.** Never use footage of anyone who has not agreed.
- Optional clip encryption at rest: `storage.encrypt_clips: true` + `CARELENS_CLIP_KEY` in `.env`
  (`python -m carelens.cliplock genkey`); see `docs/04-SECURITY-PRIVACY.md` §4. Use LUKS too.
- The dashboard binds to `127.0.0.1` and shows a persistent test-data banner.
- All dependencies are permissively licensed (MIT / Apache-2.0 / BSD); see `LICENSE`.

## Tests

```bash
python -m pytest -q
```

- `test_signal.py` — band-pass + dominant-frequency recover known HR/BR; noise and
  out-of-band signals are rejected.
- `test_events.py` — posture classification; a sudden fall is flagged, a slow lie-down
  and a quick recovery are not; inactivity fires once.
- `test_end_to_end.py` — render each synthetic clip, run the pipeline, check the expected
  events, vitals tolerance, banner wording, dim-light rejection, and empty-room case.

## Scope of the synthetic tests

Synthetic clips verify the **event logic and signal processing** against known ground
truth without filming anyone. They do **not** validate MediaPipe pose accuracy or rPPG
on real skin — that requires real, consented clips and the Apple Watch reference
(see `docs/PLAN.md` steps 6–8 and `docs/ACCURACY_REPORT_TEMPLATE.md`).

## AI tools used (hackathon disclosure)

This project was built with AI assistance, disclosed here in full:

- **Claude (Anthropic), via Claude Code** — used as the coding agent to architect the
  pipeline and write the Python code, the signal-processing (rPPG / breathing), the fall
  and inactivity logic, the synthetic-video test generator, the test suite, the local
  dashboard, and this documentation. Code was reviewed and directed by the author.
- **Human author (Sai Charan Kondapally)** — defined the concept and requirements, made
  the design decisions, recorded the consented test footage, ran all hardware validation,
  produced the accuracy numbers against the Apple Watch reference, and reviewed/approved
  the code.
- **Runtime ML models (not generative AI):** MediaPipe Pose Landmarker (Apache-2.0) for
  body keypoints; classical signal processing (OpenCV, NumPy, SciPy) for the vitals. No
  generative AI runs in the product itself — all inference is on-device and deterministic.

All code is new work created during the hackathon period. No footage of any
non-consenting person is used; all demo data is synthetic or consented.

## Configured cameras

Cameras are declared in `config.yaml` (credentials come from env vars named in the file,
via the environment or a git-ignored `.env`; never inline). Run one by id:

```
python -m carelens.pipeline --camera demo
```

CLI flags (`--inactivity-min`, `--fall-still-s`, `--height-m`, `--out`) override the config.

The `demo` camera points at the synthetic clip. Real MediaPipe does not detect the
synthetic stick figure, so feed the generator's ground-truth poses (the run is always
labelled *simulated*):

```
python -m carelens.pipeline --camera demo --inactivity-min 1 --truth-pose output/synthetic_demo.truth.json
```
