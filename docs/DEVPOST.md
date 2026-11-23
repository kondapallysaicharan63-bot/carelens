# Devpost submission — CareLens

> Draft for the STEM2Connect AI Hackathon 2026 (Healthcare track). Paste into Devpost and
> trim to taste. **Not a medical device — wellness trends only, never a diagnosis.**

## Tagline
Contactless, local-first wellness monitoring for older adults — fall and inactivity alerts
plus camera-based heart-rate and breathing estimation, with video that never leaves the room.

## Inspiration
Older adults living alone are at real risk from falls and long inactive stretches, and
family caregivers have no low-friction way to know "is today normal?" without either
always-watched video or a body-worn device the person forgets to wear. We wanted a monitor
that is ambient (nothing to wear), private (nothing leaves the home), and honest (trends and
check-in suggestions, never a diagnosis). It's designed as a companion to a wearable pendant:
the camera watches the room, the pendant is worn on the body.

## What it does
From an ordinary indoor camera it:
- Detects a **fall** — a sudden drop in body angle/height followed by stillness for 20s+ —
  and saves a timestamped event with a short clip.
- Raises an **inactivity** alert when there's no meaningful movement for a configurable time.
- Builds an **activity timeline** (sitting / standing / walking / lying per hour) and a
  **gait trend** (walking speed and steadiness over days).
- Estimates **heart rate** contactlessly (rPPG — subtle color changes in a face skin patch)
  and **breathing rate** (chest motion), each only when the person is still and well lit, and
  each with a quality score so low-confidence readings are flagged, not trusted.
- Shows everything on a **local web dashboard** with a persistent "Simulated or test data"
  banner. The dashboard binds to localhost; nothing is uploaded.

## How we built it
Python with OpenCV and MediaPipe Pose for perception, NumPy/SciPy for the signal processing
(band-pass filtering + a Welch spectrum to pull heart and breathing rates out of noisy
signals), and a small Flask dashboard. The pipeline reads any source through one interface —
a file, a USB webcam, an RTSP stream, or a Jetson CSI camera — so the camera is just one input.

To test vitals and fall logic without filming anyone, we built a **synthetic video generator**
with known ground truth: a figure that walks, sits, falls, and lies still, with a green-channel
pulse at a set heart rate and chest motion at a set breathing rate. The automated test suite
(23 tests) renders these clips, runs the full pipeline, and checks the detected events and
estimated rates against the known values.

## Accuracy (honest)
On synthetic ground truth the pipeline flags the staged fall and the inactivity case, and
recovers heart rate within ~2 bpm and breathing within ~1 /min. Real-camera accuracy is
validated separately against an Apple Watch Series 5 heart-rate reference and reported
transparently — including lighting and motion limits and the cases where it does not work.

## Challenges
Separating a genuine fall from a slow, deliberate lie-down (solved with a velocity gate on the
body-angle drop); rejecting low-quality vitals windows instead of reporting a confident wrong
number; and keeping the whole thing honest — wellness wording only, consented or synthetic
footage only, and no cloud.

## What we learned
Contactless vitals are real but fragile — they need stillness and good light, and a quality
score is as important as the estimate itself. Good synthetic ground truth lets you test the
hard logic cheaply and ethically before any real footage exists.

## What's next
A camera registry for multiple rooms, RTSP/ONVIF auto-discovery, on-device TensorRT inference
on an NVIDIA Jetson so everything runs at the edge, and caregiver alert escalation through the
user's own channel.

## Built with
python, opencv, mediapipe, numpy, scipy, flask, nvidia-jetson

## AI disclosure
Built with Claude (Anthropic) as the coding agent, under human direction; full disclosure in
the repo README. No generative AI runs in the product — all inference is on-device and
deterministic.
