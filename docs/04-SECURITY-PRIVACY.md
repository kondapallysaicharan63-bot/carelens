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
- **Alerts are opt-in** (`alerts.enabled: false` by default). The only outbound traffic is a
  user-configured webhook POST (5 s timeout, no redirects, no env proxies). Webhook URLs must be
  on the LAN (private/loopback/`.local`) unless the host is listed in `alerts.allow_public_hosts`;
  URLs with embedded credentials are rejected. URL and bearer token come from env vars
  (`url_env`/`token_env`), never the config file, and are redacted from logs and stored errors.
  Payloads carry the camera id, room, a fixed wellness-worded title, time, clip name and a
  "not a medical device / not for emergencies" disclaimer: no imagery or vitals.
- If the caregiver needs remote view: do **not** port-forward. Use a private overlay
  (e.g. WireGuard/Tailscale) to reach the dashboard, still behind auth.

## 4. Data at rest

- **Event clips**: short, low-res, under `output/clips/`. Set `storage.encrypt_clips: true`
  (off by default, recommended for real deployments) to store each clip as `<name>.mp4.enc`
  (AES-256-GCM, file name bound as associated data). Make a key with
  `python -m carelens.cliplock genkey` and put it in `.env` as `CARELENS_CLIP_KEY` (name set by
  `storage.clip_key_env`); the tool never writes it to disk, and clips are unrecoverable without
  it. Runs refuse to start if encryption is on and the key is missing/invalid. The dashboard
  decrypts in memory on `/clips/<name>.enc` (403 without a key, 409 on a bad/foreign/tampered
  file). Limits: the plaintext clip exists briefly before encryption and unlink is not a secure
  erase on flash, so **also use full-disk encryption (LUKS)**; `storage.clip_tmp_dir` can point
  that brief write at a tmpfs. Files over 200 MB are refused. Purge deletes `.enc` clips like any other;
  a startup sweep removes orphan plaintext clips left by an interrupted run. The dashboard reads
  `clip_key_env` from `--config` (or `./config.yaml` if present) and serves **all** clips, plain or
  encrypted, only with a valid `X-CareLens-Token` or to a loopback client.
- **Retention**: auto-purge clips older than N days (proposed default 7). Keep derived
  daily rollups in SQLite (no imagery) longer for trends.
- **Results JSON**: contains timestamps, postures, rates — treat as health data; same
  access control as clips.

## 5. Access control (dashboard)

- v0: localhost only, no auth (single-user on the device).
- LAN/remote: front with nginx/Caddy → require login (basic auth or a simple token), serve
  over TLS (self-signed or internal CA). Rate-limit. Never expose `/clips/` without auth.
- Alert writes (`POST /api/alerts/<id>/ack|false-alarm`): if `CARELENS_DASH_TOKEN` is set the
  `X-CareLens-Token` header must match (constant-time compare, same-origin only); if unset, only
  loopback clients with a localhost `Origin` (or none) are accepted, others get 403. The dashboard
  warns at startup when bound to a non-loopback host without a token. Bodies are capped at 4 KB.
- Behind a reverse proxy (Caddy/nginx) the browser's `Origin` is the public name, not the
  dashboard's bind address. `X-Forwarded-*` headers are deliberately **not** trusted; list the
  public origin(s) in `dashboard.allowed_origins` (or `CARELENS_DASH_ORIGINS`, comma-separated),
  e.g. `https://carelens.example.org`. They are honoured only when `CARELENS_DASH_TOKEN` is set.
  The token may live in `.env` next to the config (the process environment wins).
- Alerts only fire for runs labelled `live`; simulated/test runs stay silent unless
  `alerts.include_non_live: true`, and then the title is prefixed `[TEST]`.
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
