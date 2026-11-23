# 03 — Tech Stack

All dependencies are permissively licensed (MIT / Apache-2.0 / BSD).

## 1. Core (host + Jetson)

| Area | Choice | Why |
|------|--------|-----|
| Language | Python 3.10+ | Ecosystem for CV/signal; matches JetPack 6 |
| CV / IO | OpenCV (GStreamer-enabled) | Capture, RTSP/CSI, frame ops |
| Pose | MediaPipe Pose Landmarker (dev/host) | Fast, accurate, Apache-2.0 |
| Pose (Jetson, perf) | **TensorRT pose** — `trt_pose` or YOLOv8-pose→TensorRT | GPU-accelerated; MediaPipe is CPU-only on Jetson |
| Signal | NumPy + SciPy | Butterworth band-pass, Welch PSD |
| Dashboard | Flask + vanilla JS/SVG | Tiny, local, no build step |
| Tests | pytest | Unit + end-to-end |
| Packaging | venv + requirements.txt (host); JetPack system Python (device) | Simple, reproducible |

## 2. Jetson Orin Nano target

- **JetPack 6.x** (Ubuntu 22.04, Python 3.10, CUDA 12, TensorRT 10).
- **OpenCV**: use a build **with GStreamer + CUDA**. The pip `opencv-python` wheel has no
  GStreamer; build from source or use an NVIDIA/community JetPack OpenCV. The CSI pipeline
  (`carelens/sources.py:jetson_csi_pipeline`) needs `nvarguscamerasrc`.
- **Pose on device**: MediaPipe aarch64 wheels run CPU-only and may miss ~15 FPS at 720p.
  Plan: keep MediaPipe as the reference/accuracy backend, add a **TensorRT `PoseProvider`**
  for the live path and benchmark both (build phase P2). The `PoseProvider` protocol already
  isolates this swap.
- **Power/thermal**: run `nvpmodel`/`jetson_clocks`; log temps during soak tests.

### Legacy Jetson Nano fork (only if it's the old board)
JetPack 4.6, Ubuntu 18.04, Python 3.6, CUDA 10, no modern MediaPipe wheels. In that case:
pin the signal/event code to 3.6-compatible syntax, drop MediaPipe for `trt_pose`
outright, and expect lower throughput. Confirm the board before committing to this path.

## 3. Dependency pinning

Host `requirements.txt` is loose for development. For the Jetson, pin exact versions that
match the installed JetPack (CUDA/TensorRT are system-provided — do **not** pip-install
them). Keep a separate `requirements-jetson.txt` once versions are confirmed on hardware.

## 4. What we deliberately avoid

- No cloud SDKs, no telemetry, no analytics libraries.
- No GPL/AGPL components (license constraint).
- No heavyweight web framework or JS build toolchain for the dashboard.
- No face-recognition / identity libraries (privacy + scope).
