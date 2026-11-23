# 04 — Security & Privacy

This system watches a person in their home and derives health signals. Privacy is a
feature, not an afterthought — and it's the strongest story for the portfolio.

## 1. Principles

- **On-device by default.** Capture, inference, storage, and dashboard all run on the
  Jetson. No video or health data leaves the device unless the owner explicitly turns on
  LAN access or alerting.
- **Data minimization.** Prefer derived signals (events, rates, timeline) over raw video.
  Keep raw frames only transiently in a rolling buffer; persist only short event clips.
- **Consent-first.** Only consented or synthetic footage. A consent record is kept for
  every real subject (see §6).
- **Honest output.** Wellness trends, never diagnosis — reflected in UI wording.

## 2. Threat model

| Threat | Vector | Mitigation |
|--------|--------|------------|
| Video exfiltration | Camera cloud app / network | Use local sources (USB/CSI/RTSP); no cloud account in path; device firewall blocks outbound |
| Dashboard snooping | Other device on LAN | Bind to `127.0.0.1`; LAN access only via reverse proxy with auth + TLS |
| Stolen device / SD | Physical theft | Full-disk encryption (LUKS) or app-level encryption of clip store |
| Credential leak | RTSP user/pass in code | Store in `.env`/credentials file, git-ignored; never in source |
| Tampered model/deps | Supply chain | Pin versions + checksums; vendor the pose model file; permissive licenses only |
| Misuse as surveillance | Scope creep | No identity/recognition; one consenting subject; clear banner |

## 3. Network posture

- Default: **no inbound, no outbound.** The pipeline needs no internet at runtime.
- CSI/USB cameras: no network at all (best).
- RTSP cameras: keep the camera on an isolated VLAN/guest network; the Jetson reaches it
  on the LAN; neither needs WAN.
- If the caregiver needs remote view: do **not** port-forward. Use a private overlay
  (e.g. WireGuard/Tailscale) to reach the dashboard, still behind auth.

## 4. Data at rest

- **Event clips**: short, low-res, under `output/clips/`. Encrypt the store (LUKS volume
  or per-file encryption with a key in the OS keyring).
- **Retention**: auto-purge clips older than N days (proposed default 7). Keep derived
  `history.jsonl` (no imagery) longer for trends.
- **Results JSON**: contains timestamps, postures, rates — treat as health data; same
  access control as clips.

## 5. Access control (dashboard)

- v0: localhost only, no auth (single-user on the device).
- LAN/remote: front with nginx/Caddy → require login (basic auth or a simple token), serve
  over TLS (self-signed or internal CA). Rate-limit. Never expose `/clips/` without auth.
- Path-traversal guard already enforced in `dashboard/app.py` (keep the test).

## 6. Consent & compliance posture

- Maintain a `CONSENT.md` / log: who was filmed, when, that they agreed, and for what.
- This is a learning/demo project, **not** a regulated medical device. Do not imply FDA
  clearance. If the Sukhi product line ever pursues a claim, that's a separate regulatory
  track (see the MaudeSignal/PMS work) — keep this repo clearly labeled wellness.
- Biometric note: face video and heart-rate data are sensitive personal data under laws
  like BIPA/GDPR. On-device processing + minimization is the mitigation; document it.

## 7. Secrets handling (build rule for Claude Code)

- Never commit camera credentials, Wi-Fi keys, or tokens.
- Provide `.env.example`; load real values at runtime via environment.
- Add `.gitignore` for `.env`, `output/`, `*.mp4`, model caches.
