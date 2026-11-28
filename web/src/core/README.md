DOM-free CareLens core, ported from Python.

Nothing in this folder may touch the DOM, `window`, or MediaStream APIs, so it runs in a
Web Worker and under Vitest in Node. Thresholds come from `shared/params.json`.

- `ppg.ts`: contact (fingertip) PPG, a mirror of `carelens/ppg.py`; hand-written radix-2 FFT.
- `parity.ts`: fixture registry (`IMPLEMENTATIONS`) and the shared comparison rules.
- `dsp.ts`: shared FFT/spectral building blocks (mirror of `carelens/dsp.py`).
- `rppg.ts`: face rPPG (POS + CHROM cross-check), mirror of `carelens/rppg.py`.
