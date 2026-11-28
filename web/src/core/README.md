DOM-free CareLens core, ported from Python.

Nothing in this folder may touch the DOM, `window`, or MediaStream APIs, so it runs in a
Web Worker and under Vitest in Node. Thresholds come from `shared/params.json`.

- `ppg.ts`: contact (fingertip) PPG, a mirror of `carelens/ppg.py`; hand-written radix-2 FFT.
- `parity.ts`: fixture registry (`IMPLEMENTATIONS`) and the shared comparison rules.
- `dsp.ts`: shared FFT/spectral building blocks (mirror of `carelens/dsp.py`).
- `rppg.ts`: face rPPG (POS + CHROM cross-check), mirror of `carelens/rppg.py`.
- `features.ts`, `tracking.ts`, `events.ts`, `pipeline.ts`: pose chain (presence, posture, fall, inactivity), mirrors of `features.py`, `tracking.py`, `events.py` and the wiring in `pipeline.py`.
- `breathing.ts`, `welch.ts`: breathing rate from chest displacement (scipy-compatible Butterworth/filtfilt/Welch), mirror of `vitals.py`.
- `chestShift.ts`: browser-only image -> vertical shift (row profile + cross-correlation, parabolic); platform specific, not part of Python parity.
- `scheduler.ts`: ~8 Hz pose schedule, over-budget skip, camera-gap detection, face-only-while-active.
