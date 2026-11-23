# Accuracy report — CareLens

> Fill this in with **real, consented** recordings. Keep it honest: report misses and
> limits, not just the good cases. Wellness trends only — not a diagnosis.

**Author:** Sai Charan Kondapally
**Date:**
**Hardware:** camera = ____ (Roku clip export / USB webcam / Jetson CSI) · compute = ____
**Reference:** Apple Watch Series 5 (HR + ECG, main) · older rose-gold Watch (HR, second check)

---

## 1. Test set

| Clip | Scenario | Length | Lighting | Person angle | Notes |
|---|---|---|---|---|---|
| sitting_01 | sitting still | | | | |
| walking_01 | walking across room | | | | |
| fall_01 | staged fall onto mattress | | | | |
| lying_01 | lying still | | | | |
| leave_01 | leaving the room | | | | |

Consent obtained from every person filmed: ☐ yes
Public fall datasets used (name + license): ____

## 2. Fall detection

| Clip | True event | Detected? | Onset error (s) | Notes |
|---|---|---|---|---|
| fall_01 | fall @ __:__ | | | |
| sitting_01 | none (should NOT fire) | | | |
| lying_01 (slow) | none (should NOT fire) | | | |

- Hits: __ / __   False alarms: __   Misses: __
- Failure modes seen:

## 3. Heart rate (rPPG vs Apple Watch)

| Window (clip @ t) | Camera bpm | Quality | Watch bpm | Error (bpm) |
|---|---|---|---|---|
| | | | | |

- **Mean absolute error: ____ bpm** (over __ reported windows)
- Windows the camera declined to report (low quality / motion / light): __
- At what lighting / distance / motion does it break down:

## 4. Breathing rate

| Window | Camera /min | Quality | Reference /min | Error |
|---|---|---|---|---|
| | | | | |

- Mean absolute error: ____ /min · reference method used: ____

## 5. Gait (trend, approximate)

| Clip | Speed (m/s, approx) | Trunk sway (°) | Notes |
|---|---|---|---|

- Note: speed uses the person's height for scale and assumes a side-on view; treat as a
  day-to-day trend, not an absolute measurement.

## 6. Limits and honest assessment

- What works well:
- What does not work / is unreliable:
- Lighting range that works:
- Minimum stillness needed for vitals:
- Known false-alarm triggers:
- What I would improve next:
