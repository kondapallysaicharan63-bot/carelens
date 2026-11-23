# 08 — Deployment on Jetson Orin Nano

Target: Jetson **Orin Nano** dev kit, JetPack 6.x (Ubuntu 22.04, Python 3.10, CUDA 12,
TensorRT 10), NVMe SSD. (If it's the legacy Nano, see `03-TECH-STACK.md` fork first.)

## 1. One-time setup

1. Flash JetPack 6 (SDK Manager or the SD/NVMe image). Boot, update:
   `sudo apt update && sudo apt full-upgrade`.
2. Set performance mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
3. **OpenCV with GStreamer + CUDA** — the pip wheel won't do CSI capture. Use NVIDIA's
   JetPack OpenCV or build from source with `-D WITH_GSTREAMER=ON -D WITH_CUDA=ON`.
   Verify: `python3 -c "import cv2; print(cv2.getBuildInformation())"` shows GStreamer YES.
4. System libs: `sudo apt install python3-pip python3-venv gstreamer1.0-plugins-*`.
5. Python deps: install from `requirements-jetson.txt` (pin to the board; do **not**
   pip-install CUDA/TensorRT — they're system-provided).
6. Pose model: the MediaPipe `.task` ships in `models/`. For the live path, build/convert
   the TensorRT pose engine on-device (P2, task 7) and drop it in `models/`.

## 2. Camera bring-up

- **CSI** (IMX219/Pi Cam v2): ribbon to CAM0, then `--source "csi://0"`. Test the raw
  pipeline with `gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! autovideosink`.
- **USB**: plug in, find index with `v4l2-ctl --list-devices`, then `--source 0`.
- **RTSP** (wired PoE or Wi-Fi cam): `--source "rtsp://..."`.

## 3. Run

```bash
# live, CSI camera
python3 -m carelens.pipeline --source "csi://0" --name livingroom --inactivity-min 30

# dashboard (localhost)
python3 dashboard/app.py --out output    # http://127.0.0.1:5000
```

## 4. Autostart (systemd)

Create two units (pipeline + dashboard) enabled on boot, logging to journald. Example
(`/etc/systemd/system/carelens.service`):

```ini
[Unit]
Description=CareLens pipeline
After=network-online.target
[Service]
WorkingDirectory=/home/USER/carelens
EnvironmentFile=/home/USER/carelens/.env
ExecStart=/usr/bin/python3 -m carelens.pipeline --source csi://0 --name livingroom
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable --now carelens` then repeat for the dashboard. Reboot to verify.

## 5. Performance & thermal check

- Watch `tegrastats` during a 30-min soak; record FPS (pipeline log), CPU/GPU, temps.
- If below 15 FPS @ 720p on the MediaPipe CPU path, switch the live `PoseProvider` to the
  TensorRT engine (P2). Lower capture resolution as a fallback.

## 6. Hardening for a real install

- Full-disk encryption (LUKS) or encrypted clip store (see `04-SECURITY-PRIVACY.md`).
- Put IP cameras on an isolated VLAN; no WAN exposure; WireGuard for any remote view.
- Set clip retention; confirm `.env`/creds are not world-readable (`chmod 600`).
