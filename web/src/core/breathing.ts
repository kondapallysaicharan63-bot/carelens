// Breathing rate from vertical chest displacement: DOM-free TypeScript port of carelens/vitals.py
// (estimate_window, _WindowEstimator, BreathingRateEstimator.update_displacement, chest_roi).
// Same grid, detrend, Butterworth band-pass (zero-phase), Welch spectrum, quality formula and gates, driven by
// shared/params.json -> vitals (band 0.1-0.7 Hz = 6-42 breaths/min). Parity is defined at the displacement level:
// the image -> displacement step is platform specific (Python: cv2.phaseCorrelate on a chest ROI; browser:
// ./chestShift, a 1-D row-profile cross-correlation). A wellbeing trend, never a medical measurement.

import { detrendLinear } from './dsp';
import { butterBandpassSos, sosfiltfilt, welch } from './welch';

export interface VitalsSignalParams {
  bandpass_order: number; bandpass_nyquist_frac: number; welch_min_segment_s: number; welch_segment_cycles: number;
  welch_nfft_min: number; welch_nfft_oversample: number; peak_window_frac: number; estimator_min_samples: number;
  estimator_window_fill_frac: number;
}
export interface BreathingParams { window_s: number; step_s: number; band_lo_hz: number; band_hi_hz: number }
export interface RoiParams {
  min_landmark_visibility: number; min_box_px: number;
  chest_half_width_shoulder_frac: number; chest_half_width_torso_frac: number;
  chest_center_along_torso_frac: number; chest_half_height_torso_frac: number;
}
export interface VitalsParams { signal: VitalsSignalParams; breathing: BreathingParams; roi: RoiParams }
export interface CommonParams { still_motion_bodysizes_s: number; min_quality: number }

export interface BreathingEstimate { t: number; value: number; quality: number; reported: boolean; span_s: number }

const round = (x: number, dp: number): number => { const m = 10 ** dp; return Math.round(x * m) / m; };

/** (peak_hz, quality) from a Welch spectrum restricted to [lo, hi] (carelens.vitals.dominant_frequency). */
export function dominantFrequency(x: ArrayLike<number>, fs: number, lo: number, hi: number, s: VitalsSignalParams): [number | null, number] {
  const n = x.length;
  const nperseg = Math.min(n, Math.trunc(fs * Math.max(s.welch_min_segment_s, s.welch_segment_cycles / lo)));
  const nfft = Math.max(s.welch_nfft_min, 2 ** Math.ceil(Math.log2(n * s.welch_nfft_oversample)));
  const { f, pxx } = welch(x, fs, nperseg, Math.floor(nperseg / 2), nfft);
  const idx: number[] = [];
  for (let k = 0; k < f.length; k++) if (f[k]! >= lo && f[k]! <= hi) idx.push(k);
  let total = 0;
  for (const k of idx) total += pxx[k]!;
  if (idx.length === 0 || !(total > 0)) return [null, 0];
  let pk = idx[0]!;
  for (const k of idx) if (pxx[k]! > pxx[pk]!) pk = k;
  const peak = f[pk]!;
  const win = Math.max(s.peak_window_frac * peak, 2 * (f[1]! - f[0]!));
  let nearPow = 0, nearCnt = 0;
  for (const k of idx) {
    if (Math.abs(f[k]! - peak) <= win || Math.abs(f[k]! - 2 * peak) <= win) { nearPow += pxx[k]!; nearCnt++; }
  }
  const ratio = nearPow / total;
  const chance = nearCnt / idx.length;
  const q = Math.min(1, Math.max(0, (ratio - chance) / (1 - chance + 1e-9)));
  return [peak, q];
}

/** Pure per-window estimate on an ROI-level trace (carelens.vitals.estimate_window). */
export function estimateWindow(
  ts: number[], values: number[], lo: number, hi: number, minQuality: number, s: VitalsSignalParams,
): { value: number; quality: number; reported: boolean } | null {
  const n = ts.length;
  if (n < Math.max(2, s.estimator_min_samples) || !(ts[n - 1]! > ts[0]!)) return null;
  const fs = (n - 1) / (ts[n - 1]! - ts[0]!);
  // np.interp(linspace(t0, t1, n), ts, values)
  const x = new Array<number>(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const g = ts[0]! + ((ts[n - 1]! - ts[0]!) * i) / (n - 1);
    if (g >= ts[n - 1]!) { x[i] = values[n - 1]!; continue; }
    while (ts[j + 1]! <= g) j++;
    const sl = (values[j + 1]! - values[j]!) / (ts[j + 1]! - ts[j]!);
    x[i] = sl * (g - ts[j]!) + values[j]!;
  }
  const xd = detrendLinear(x);
  if (xd.every((v) => Math.abs(v) <= 1e-8)) return null;
  const nyq = fs / 2;
  const hiEff = Math.min(hi, nyq * s.bandpass_nyquist_frac);
  const sos = butterBandpassSos(s.bandpass_order, lo / nyq, hiEff / nyq);
  const xf = sosfiltfilt(sos, xd);
  const [f, q] = dominantFrequency(xf, fs, lo, hi, s);
  if (f === null) return null;
  return { value: round(f * 60, 1), quality: round(q, 3), reported: q >= minQuality };
}

/** Breathing-rate estimator on a cumulative chest-displacement trace (carelens.vitals.BreathingRateEstimator). */
export class BreathingRateEstimator {
  readonly estimates: BreathingEstimate[] = [];
  private buf: [number, number][] = [];
  private lastEmit = -1e9;
  private readonly p: BreathingParams;
  private readonly sig: VitalsSignalParams;

  constructor(vitals: Pick<VitalsParams, 'signal' | 'breathing'>, private readonly minQuality: number) {
    this.p = vitals.breathing;
    this.sig = vitals.signal;
  }

  get windowS(): number { return this.p.window_s; }

  /** Drop the window (person absent, moving, held pose, camera gap). */
  reset(): void { this.buf = []; }

  /** Push the cumulative chest displacement (px) at time t. `due` is true when an estimate was attempted this call;
   *  then `estimate` is null if that attempt failed (flat trace etc.), so callers can drop a stale reading. */
  updateDisplacement(t: number, pos: number): { due: boolean; estimate: BreathingEstimate | null } {
    this.buf.push([t, pos]);
    while (this.buf.length && this.buf[0]![0] < t - this.p.window_s) this.buf.shift();
    if (t - this.lastEmit < this.p.step_s || this.buf.length < this.sig.estimator_min_samples) return { due: false, estimate: null };
    const ts = this.buf.map((b) => b[0]);
    if (ts[ts.length - 1]! - ts[0]! < this.p.window_s * this.sig.estimator_window_fill_frac) return { due: false, estimate: null };
    this.lastEmit = t;
    const r = estimateWindow(ts, this.buf.map((b) => b[1]), this.p.band_lo_hz, this.p.band_hi_hz, this.minQuality, this.sig);
    if (!r) return { due: true, estimate: null };
    const est = { t: round(t, 2), value: r.value, quality: r.quality, reported: r.reported, span_s: round(ts[ts.length - 1]! - ts[0]!, 2) };
    this.estimates.push(est);
    return { due: true, estimate: est };
  }

  /** A reading older than this (s) is no longer shown. */
  get maxAgeS(): number { return this.p.window_s + this.p.step_s; }
}

export interface Box { x0: number; y0: number; x1: number; y1: number }

function clipBox(x0: number, y0: number, x1: number, y1: number, w: number, h: number, minBox: number): Box | null {
  const ax0 = Math.trunc(Math.max(0, Math.min(x0, x1))), ax1 = Math.trunc(Math.min(w, Math.max(x0, x1)));
  const ay0 = Math.trunc(Math.max(0, Math.min(y0, y1))), ay1 = Math.trunc(Math.min(h, Math.max(y0, y1)));
  if (ax1 - ax0 < minBox || ay1 - ay0 < minBox) return null;
  return { x0: ax0, y0: ay0, x1: ax1, y1: ay1 };
}

const L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24;

/** Upper-torso box between shoulders and hips (carelens.vitals.chest_roi). `lm` is [x, y, vis] per landmark with x, y in
 *  frame-height units (x already multiplied by the aspect, as in BodyState.landmarks); the box is in pixels of a frame
 *  frameW x frameH. */
export function chestRoi(lm: ArrayLike<ArrayLike<number>>, frameH: number, frameW: number, r: RoiParams): Box | null {
  const vis = (lm[L_SHOULDER]![2]! + lm[R_SHOULDER]![2]! + lm[L_HIP]![2]! + lm[R_HIP]![2]!) / 4;
  if (vis < r.min_landmark_visibility) return null;
  const px = (k: number): [number, number] => [lm[k]![0]! * frameH, lm[k]![1]! * frameH];
  const [lsx, lsy] = px(L_SHOULDER), [rsx, rsy] = px(R_SHOULDER), [lhx, lhy] = px(L_HIP), [rhx, rhy] = px(R_HIP);
  const sh = [(lsx + rsx) / 2, (lsy + rsy) / 2], hip = [(lhx + rhx) / 2, (lhy + rhy) / 2];
  const sw = Math.hypot(lsx - rsx, lsy - rsy);
  const torso = Math.hypot(hip[0]! - sh[0]!, hip[1]! - sh[1]!);
  const halfW = Math.max(sw * r.chest_half_width_shoulder_frac, torso * r.chest_half_width_torso_frac);
  const cx = sh[0]! + r.chest_center_along_torso_frac * (hip[0]! - sh[0]!);
  const cy = sh[1]! + r.chest_center_along_torso_frac * (hip[1]! - sh[1]!);
  const hh = r.chest_half_height_torso_frac * torso;
  return clipBox(cx - halfW, cy - hh, cx + halfW, cy + hh, frameW, frameH, r.min_box_px);
}

/** Result of the platform image step for one frame: a vertical shift in px, or why there is none.
 *  'first' = no previous frame yet (a zero-displacement sample is still pushed, as Python does);
 *  'too_small' = the chest box is too small to correlate; 'weak' = correlation too low. The last two NEVER become samples. */
export type ShiftResult = number | 'first' | 'too_small' | 'weak';

export interface ChestImageSource {
  size(): { w: number; h: number } | null;
  shiftAndAdvance(box: Box): ShiftResult;
  reset(): void;
}

const MAX_WEAK_FRAMES = 16; // ~2 s at 8 Hz of unusable correlation: start the window over

/** Per-frame breathing channel: mirrors BreathingRateEstimator.update (pose gating, chest ROI, cumulative displacement)
 *  with the image -> displacement step supplied by the platform. `held` frames and any frame where the person is absent,
 *  moving or has no usable chest ROI reset the window, as in carelens/pipeline.py. Unusable image shifts are skipped,
 *  never turned into zeros. */
export class BreathingChannel {
  readonly estimator: BreathingRateEstimator;
  private pos = 0;
  private weak = 0;
  /** why there is no estimate this frame (for guidance text) */
  status: 'ok' | 'absent' | 'moving' | 'no_chest' | 'too_small' | 'held' = 'absent';

  constructor(private readonly vitals: VitalsParams, private readonly common: CommonParams) {
    this.estimator = new BreathingRateEstimator(vitals, common.min_quality);
  }

  reset(src?: { reset(): void }): void {
    this.estimator.reset();
    this.pos = 0;
    this.weak = 0;
    src?.reset();
  }

  update(
    t: number, state: { present: boolean; motion: number; landmarks: number[][] | null }, held: boolean, src: ChestImageSource,
  ): { due: boolean; estimate: BreathingEstimate | null } {
    const none = { due: false, estimate: null };
    if (held) { this.status = 'held'; this.reset(src); return none; }
    if (!state.present || state.landmarks === null) { this.status = 'absent'; this.reset(src); return none; }
    if (state.motion > this.common.still_motion_bodysizes_s) { this.status = 'moving'; this.reset(src); return none; }
    const size = src.size();
    const roi = size ? chestRoi(state.landmarks, size.h, size.w, this.vitals.roi) : null;
    if (roi === null) { this.status = 'no_chest'; this.reset(src); return none; }
    const r = src.shiftAndAdvance(roi);
    if (r === 'too_small') { this.status = 'too_small'; this.reset(src); return none; }
    if (r === 'weak') {
      this.weak++;
      if (this.weak > MAX_WEAK_FRAMES) { this.status = 'no_chest'; this.reset(src); return none; }
      this.status = 'ok';
      return none; // no sample this frame
    }
    this.weak = 0;
    if (r !== 'first') this.pos += r;
    this.status = 'ok';
    return this.estimator.updateDisplacement(t, this.pos);
  }
}
