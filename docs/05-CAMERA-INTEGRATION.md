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
- Store camera configs in a small `config.yaml` (id, type, URL/index, label, location),
  credentials pulled from env — never inline.

### ONVIF discovery

`python -m carelens.discovery` sends a WS-Discovery Probe (UDP multicast 239.255.255.250:3702)
and lists ONVIF cameras with a ready-to-paste `config.yaml` snippet. With `--stream-uri` it
asks each camera's ONVIF Media service for its RTSP URI (WS-Security digest auth using
`RTSP_USER`/`RTSP_PASS`); returned URIs are stripped of credentials. It is LAN-only: cloud-app
cameras (Roku/Blink/Nest) do not appear and remain clip-export only. Cameras declare
`rtsp_transport` (default `tcp`) and `open_timeout_s` (default 10).

## 4. Reliability requirements (live sources)

Live kinds (usb/rtsp/csi) use `LiveSource` (`carelens/sources.py`); files keep the synchronous
`VideoSource` (deterministic, never drops frames). `LiveSource` provides:
- **Threaded capture** into a 2-slot latest-frame buffer; when inference is slower the OLDEST frame
  is dropped (`frames_dropped`), so latency stays bounded (`last_latency_s`).
- **Auto-reconnect** with exponential backoff (0.5 s, x2, cap 30 s, +-20 % jitter, forever) on a
  failed open/read, plus a **stall watchdog** (no frame for `stall_s`, default 5 s).
- **Timestamps**: monotonic seconds since start, stamped at capture time.
- **Gaps**: the first frame after an outage has `Frame.gap_before_s` > 0. `run()` resets tracker and
  vitals, and calls `pause(gap)` on the fall/inactivity detectors so unobserved time never counts
  as stillness and a pending fall survives. Outages >= `stall_s` are recorded as `camera_offline`
  events ("Camera offline for N s - check the camera"); they are not counted as falls/inactivity.
- **Outage reporting**: once an outage lasts `stall_s`, `run()` records a `camera_offline` event at
  its ONSET (and calls the `on_event` hook, for alerting) and fills in the duration on recovery or at
  run end. Optional `max_outage_s` (camera config) ends and saves the run so a service manager can
  restart it. A frame that `read()` returns is never discarded; the first read per connection gets
  extra time (keyframe wait) before stall rules apply.
- `status` (`connecting|up|down`), `reconnects`, optional `on_status` callback. URLs are always redacted.

Limitation: the watchdog runs in the capture thread, so a `read()` that never returns is bounded
only by the backend's read timeout (set for RTSP; USB/CSI have none).

## 5. PTZ control (ONVIF)

ONVIF cameras expose pan-tilt-zoom over the standard ONVIF/HTTP interface. PTZ (patrol /
tracking) can be driven from the pipeline as a v2 feature.

## 6. Multi-camera (v2)

`config.yaml` + one pipeline process per camera, or a scheduler round-robining sources.
Keep per-camera `output/<camera_id>/` trees. Out of scope for v1.
