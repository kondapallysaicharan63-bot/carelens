// Browser image -> chest displacement step (platform specific; the Python reference uses cv2.phaseCorrelate on a 2-D
// chest ROI image). Parity with Python is defined one level up, at the cumulative-displacement trace
// (BreathingRateEstimator.updateDisplacement), exactly like W2's ROI means: this file only has to produce a
// vertical shift per frame that is good enough for the shared spectral path.
//
// Method (1-D, cheap enough for a phone at ~8 Hz): reduce the chest ROI of two consecutive frames to a row-mean
// intensity profile (columns sub-sampled so the patch is <= patch_max_w_px wide), remove the mean, apply a Hann
// window, then find the vertical shift by normalised cross-correlation over integer lags +-max_shift_px and refine
// the peak with a parabola (sub-pixel). Positive = the content moved DOWN in the image from `prev` to `cur`.
// Limits: tiny shifts are slightly under-estimated (the parabola is biased toward the integer lag), which scales the
// displacement amplitude but not its frequency; periodic stripes (clothing) can alias beyond half their period,
// which max_shift_px keeps out of range. Not a medical device.

import type { Box, ShiftResult } from './breathing';
export type { ChestImageSource } from './breathing';

export interface ChestShiftParams { frame_w_px: number; patch_max_w_px: number; max_shift_px: number; min_corr: number }

/** Row-mean profile of the box in a w-wide grayscale image (length = box height). Columns are sub-sampled by an integer
 *  stride so that at most `maxW` columns contribute per row. */
export function rowProfile(gray: ArrayLike<number>, width: number, box: Box, maxW: number): Float64Array {
  const bw = box.x1 - box.x0, bh = box.y1 - box.y0;
  const stride = Math.max(1, Math.ceil(bw / maxW));
  const out = new Float64Array(bh);
  for (let y = 0; y < bh; y++) {
    let s = 0, n = 0;
    const row = (box.y0 + y) * width;
    for (let x = box.x0; x < box.x1; x += stride) { s += gray[row + x]!; n++; }
    out[y] = s / n;
  }
  return out;
}

function prepare(p: Float64Array): Float64Array {
  const n = p.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += p[i]!;
  m /= n;
  const o = new Float64Array(n);
  for (let i = 0; i < n; i++) o[i] = (p[i]! - m) * (0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / n));
  return o;
}

/** Vertical shift (px, sub-pixel) of `cur` relative to `prev` and the peak correlation in [-1, 1], or null when the
 *  profiles are too short / flat or the peak correlation is below `minCorr`. */
export function estimateShift(prev: ArrayLike<number>, cur: ArrayLike<number>, maxShift: number, minCorr: number): { dy: number; corr: number } | null {
  const n = Math.min(prev.length, cur.length);
  if (n < 2 * maxShift + 4) return null;
  const a = prepare(Float64Array.from(prev).subarray(0, n)), b = prepare(Float64Array.from(cur).subarray(0, n));
  const c = new Float64Array(2 * maxShift + 1);
  for (let s = -maxShift; s <= maxShift; s++) {
    let ab = 0, aa = 0, bb = 0;
    const i0 = Math.max(0, -s), i1 = Math.min(n, n - s);
    for (let i = i0; i < i1; i++) { const x = a[i]!, y = b[i + s]!; ab += x * y; aa += x * x; bb += y * y; }
    c[s + maxShift] = aa > 1e-12 && bb > 1e-12 ? ab / Math.sqrt(aa * bb) : 0;
  }
  let k = 0;
  for (let i = 1; i < c.length; i++) if (c[i]! > c[k]!) k = i;
  const corr = c[k]!;
  if (!(corr >= minCorr)) return null;
  let d = 0;
  if (k > 0 && k < c.length - 1) {
    const l = c[k - 1]!, r = c[k + 1]!, den = l - 2 * corr + r;
    if (den < 0) d = Math.max(-0.5, Math.min(0.5, (0.5 * (l - r)) / den));
  }
  return { dy: k - maxShift + d, corr };
}

/** Grayscale (luma) frame holder implementing the shift logic over plain arrays (DOM-free, testable). */
export class GrayShiftTracker {
  private prev: Uint8ClampedArray | null = null;
  constructor(private readonly p: ChestShiftParams) {}

  reset(): void { this.prev = null; }

  /** `gray` is the current w-wide frame; it is copied as the next frame's reference. Returns the shift in px, 'first'
   *  (no previous frame), 'too_small' (box shorter than the correlation needs) or 'weak' (peak correlation < min_corr). */
  shift(gray: Uint8ClampedArray, w: number, box: Box): ShiftResult {
    const prev = this.prev;
    this.prev = gray.slice();
    if (prev === null || prev.length !== gray.length) return 'first';
    if (box.y1 - box.y0 < 2 * this.p.max_shift_px + 4) return 'too_small';
    const r = estimateShift(rowProfile(prev, w, box, this.p.patch_max_w_px), rowProfile(gray, w, box, this.p.patch_max_w_px), this.p.max_shift_px, this.p.min_corr);
    return r ? r.dy : 'weak';
  }
}
