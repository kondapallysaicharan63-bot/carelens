# 05 — Camera Integration

The pipeline is source-agnostic: every camera becomes a `VideoSource`. This doc defines
how each kind connects, how we discover/authenticate it, and the wired-vs-wireless
trade-offs. **Cloud-app cameras (Roku/Blink/Nest) are clip-export only** (see PRD).

## 1. Camera matrix

| Source | Link | `--source` | Live? | Notes |
|--------|------|-----------|-------|-------|
| Exported clip (incl. Roku app export) | file | `path.mp4` | no | Normal-customer use; the only Roku path |
| USB webcam (UVC) | wired USB | `0`, `1`, … | yes | Simplest live dev; plug into Jetson |
| CSI camera (IMX219 / Pi Cam v2) | wired MIPI ribbon | `csi://0` | yes | Best latency; on-board; `nvarguscamerasrc` |
| IP camera (RTSP/ONVIF) | wired PoE **or** Wi-Fi | `rtsp://user:pass@host:554/...` | yes | Any ONVIF cam |
| Network stream | wired/Wi-Fi | `http(s)://.../stream` | yes | MJPEG/HLS where offered |

## 2. Wired vs wireless

- **Wired CSI** — lowest latency, no network, most private, but tethered to the board;
  camera placement limited by ribbon length. Best for a fixed demo rig.
- **Wired USB** — easy, reliable, short cable; good for development on a desk.
- **Wired PoE IP cam** — one cable for power + data, reliable, isolate on a VLAN; good for
  a "real install" story.
- **Wireless Wi-Fi IP cam (RTSP)** — flexible placement, but subject to Wi-Fi reliability
  and bandwidth; keep the cam on a guest/IoT VLAN; expect occasional reconnects → the
  capture layer must auto-reconnect (build task below).

## 3. Discovery & configuration

- **USB**: enumerate with `v4l2-ctl --list-devices` (Linux) or probe indices 0..N.
- **CSI**: fixed sensor IDs 0/1 on the Jetson carrier.
- **IP cameras**: support two modes —
  1. **Manual** — user pastes the RTSP URL (works everywhere). v1 default.
  2. **Auto (ONVIF WS-Discovery)** — scan the LAN, list cameras, fetch stream URIs.
     v2 (F13), gated behind the config UI.
- Store camera configs in a small `cameras.yaml` (id, type, URL/index, label, location),
  credentials pulled from env — never inline.

## 4. Reliability requirements (live sources)

The current `VideoSource` opens once and iterates. For live RTSP/Wi-Fi, add:
- **Auto-reconnect** with backoff when `cap.read()` fails or the stream stalls.
- **Watchdog**: if no frame for T seconds, tear down and reopen.
- **Frame-drop under load**: keep only the latest frame for inference (drop-old queue).
- **Timestamp source**: wall-clock for live (already handled via `is_live`).

These are build tasks in `07-BUILD-PLAN.md` phase P2.

## 5. PTZ control (ONVIF)

ONVIF cameras expose pan-tilt-zoom over the standard ONVIF/HTTP interface. PTZ (patrol /
tracking) can be driven from the pipeline as a v2 feature.

## 6. Multi-camera (v2)

`cameras.yaml` + one pipeline process per camera, or a scheduler round-robining sources.
Keep per-camera `output/<camera_id>/` trees. Out of scope for v1.
