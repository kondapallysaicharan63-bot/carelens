// Face remote PPG (rPPG): DOM-free TypeScript port of carelens/rppg.py (POS with a CHROM cross-check on the
// mean RGB of a skin ROI). Same frame states, grid, POS/CHROM, spectral path (./dsp), gates and quality formula,
// driven by shared/params.json -> rppg. Read the Python docstring for the full design and its limits.
// rPPG accuracy varies with skin tone, lighting and motion: never over-claim. Not a medical device.

import {
  EPS, bandpass, clamp, detrendLinear, fftSize, gridTimes, interp, mean, powerSpectrum, spectralPeak, spectralSnr,
  type SpectralParams,
} from './dsp';

export interface RppgParams extends SpectralParams {
  window_s: number; step_s: number; fs_hz: number; max_gap_s: number; min_rate_hz: number;
  zero_pad_factor: number; pos_window_s: number; min_roi_px: number; min_skin_frac: number; motion_max: number; motion_frac_max: number;
  min_brightness: number; max_brightness: number; max_brightness_rate: number; brightness_lag_s: number;
  lighting_hold_s: number; settle_s: number; min_snr: number; agree_bpm: number; q_snr_full_db: number;
  q_w_snr: number; q_w_agree: number; q_w_motion: number;
  skin_filter: boolean; skin_cr_min: number; skin_cr_max: number; skin_cb_min: number; skin_cb_max: number;
  roi_landmarks: { forehead: number[]; left_cheek: number[]; right_cheek: number[] };
  methods: string[];
}

export type RppgReason = 'ok' | 'no_face' | 'too_small' | 'skin_filter' | 'motion' | 'lighting' | 'settling' | 'low_signal' | 'disagree' | 'gaps';

export interface RppgUpdate {
  t: number;
  bpm: number | null;
  bpm_chrom: number | null;
  quality: number;
  reported: boolean;
  reason: RppgReason;
  snr: number | null;
  /** analysed window length (s); null when no window was analysed */
  span_s: number | null;
  /** counts valid-face restarts; a measurement must restart when it changes */
  track_epoch: number;
}

export type Rgb = readonly [number, number, number];

/** Consistency checks mirroring carelens.params.rppg_errors; returns the problems found. */
export function rppgErrors(p: RppgParams): string[] {
  const e: string[] = [];
  if (!(0 < p.taper_stop_lo_hz && p.taper_stop_lo_hz < p.band_lo_hz && p.band_lo_hz < p.band_hi_hz && p.band_hi_hz < p.taper_stop_hi_hz && p.taper_stop_hi_hz < p.fs_hz / 2)) {
    e.push('need 0 < taper_stop_lo < band_lo < band_hi < taper_stop_hi < Nyquist');
  }
  if (Math.min(p.q_w_snr, p.q_w_agree, p.q_w_motion) < 0 || Math.abs(p.q_w_snr + p.q_w_agree + p.q_w_motion - 1) > 1e-9) {
    e.push('quality weights must be >= 0 and sum to 1');
  }
  if (!(p.min_snr > 0 && p.q_snr_full_db > 10 * Math.log10(Math.max(p.min_snr, 1e-300)))) {
    e.push('need min_snr > 0 and q_snr_full_db above the gate in dB');
  }
  for (const k of ['window_s', 'step_s', 'fs_hz', 'max_gap_s', 'min_rate_hz', 'settle_s', 'agree_bpm', 'snr_halfwidth_hz',
    'harm_halfwidth_hz', 'pos_window_s', 'min_roi_px', 'motion_max', 'max_brightness_rate', 'brightness_lag_s', 'lighting_hold_s'] as const) {
    if (!(p[k] > 0)) e.push(`${k} must be > 0`);
  }
  if (p.zero_pad_factor < 1) e.push('zero_pad_factor must be >= 1');
  if (p.harm_w2 < 0 || p.harm_w3 < 0) e.push('harm weights must be >= 0');
  const posN = p.pos_window_s * p.fs_hz;
  if (!(8 <= posN && posN <= 0.5 * p.window_s * p.fs_hz)) e.push('need 8 samples <= pos_window_s*fs_hz <= half of window_s*fs_hz');
  if (!(0 <= p.min_skin_frac && p.min_skin_frac <= 1)) e.push('need 0 <= min_skin_frac <= 1');
  if (!(0 <= p.motion_frac_max && p.motion_frac_max < 1)) e.push('need 0 <= motion_frac_max < 1');
  if (!(0 <= p.min_brightness && p.min_brightness < p.max_brightness && p.max_brightness <= 255)) e.push('need 0 <= min_brightness < max_brightness <= 255');
  if (!(0 <= p.skin_cr_min && p.skin_cr_min < p.skin_cr_max && p.skin_cr_max <= 255 && 0 <= p.skin_cb_min && p.skin_cb_min < p.skin_cb_max && p.skin_cb_max <= 255)) {
    e.push('need 0 <= skin_c*_min < skin_c*_max <= 255');
  }
  for (const [name, idx] of Object.entries(p.roi_landmarks)) {
    if (idx.length < 3 || new Set(idx).size !== idx.length || Math.min(...idx) < 0 || Math.max(...idx) > 477) {
      e.push(`roi_landmarks.${name}: need >= 3 distinct indices in 0..477`);
    }
  }
  return e;
}

export function assertRppgParams(p: RppgParams): RppgParams {
  const errs = rppgErrors(p);
  if (errs.length) throw new Error('invalid rppg params: ' + errs.join('; '));
  return p;
}

const clip01 = (v: number): number => clamp(v, 0, 1);

function std(a: ArrayLike<number>): number {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i]! - m) * (a[i]! - m);
  return Math.sqrt(s / a.length);
}

// ------------------------------------------------------------------ pulse extraction

/** POS pulse signal of an RGB trace given as three channel arrays (R, G, B). */
export function posSignal(r: number[], g: number[], b: number[], fs: number, winS: number): number[] {
  const n = r.length;
  const L = Math.floor(winS * fs + 0.5);
  const hSum = new Array<number>(n).fill(0);
  const s1 = new Array<number>(L), s2 = new Array<number>(L), h = new Array<number>(L);
  for (let m = L - 1; m < n; m++) {
    const a = m - L + 1;
    let mr = 0, mg = 0, mb = 0;
    for (let i = 0; i < L; i++) { mr += r[a + i]!; mg += g[a + i]!; mb += b[a + i]!; }
    mr /= L; mg /= L; mb /= L;
    if (!(mr > 1e-6 && mg > 1e-6 && mb > 1e-6)) continue; // a ~0 channel carries no information: skip the sub-window
    for (let i = 0; i < L; i++) {
      const gn = g[a + i]! / mg, bn = b[a + i]! / mb, rn = r[a + i]! / mr;
      s1[i] = gn - bn;
      s2[i] = gn + bn - 2 * rn;
    }
    const sd1 = std(s1), sd2 = std(s2);
    const al = sd2 > 1e-12 ? sd1 / sd2 : 0;
    for (let i = 0; i < L; i++) h[i] = s1[i]! + al * s2[i]!;
    const hm = mean(h);
    for (let i = 0; i < L; i++) hSum[a + i] = hSum[a + i]! + (h[i]! - hm);
  }
  return hSum;
}

/** CHROM pulse signal of an RGB trace: chrominance pair, band-passed, alpha-tuned. */
export function chromSignal(r: number[], g: number[], b: number[], nFft: number, fs: number, p: SpectralParams): number[] {
  const mr = mean(r), mg = mean(g), mb = mean(b);
  const n = r.length;
  const xs = new Array<number>(n), ys = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const rn = r[i]! / mr, gn = g[i]! / mg, bn = b[i]! / mb;
    xs[i] = 3 * rn - 2 * gn;
    ys[i] = 1.5 * rn + gn - 1.5 * bn;
  }
  const xf = bandpass(detrendLinear(xs), nFft, fs, p);
  const yf = bandpass(detrendLinear(ys), nFft, fs, p);
  const sdy = std(yf);
  const a = sdy > 1e-12 ? std(xf) / sdy : 0;
  return xf.map((v, i) => v - a * yf[i]!);
}

function spectrum(sig: number[], nFft: number, p: RppgParams): { f: number; snr: number } {
  const power = powerSpectrum(sig, nFft);
  const f = spectralPeak(power, nFft, p.fs_hz, p);
  return { f, snr: spectralSnr(power, nFft, p.fs_hz, f, p) };
}

export function qualityScore(snr: number, deltaBpm: number, meanMotion: number, p: RppgParams): number {
  const gateDb = 10 * Math.log10(p.min_snr);
  const qSnr = clip01((10 * Math.log10(Math.max(snr, EPS)) - gateDb) / (p.q_snr_full_db - gateDb));
  const qAgree = clip01(1 - Math.abs(deltaBpm) / p.agree_bpm);
  const qMotion = clip01(1 - meanMotion / p.motion_max);
  return p.q_w_snr * qSnr + p.q_w_agree * qAgree + p.q_w_motion * qMotion;
}

// ------------------------------------------------------------------ window analysis

export type RppgWindow =
  | { gaps: true }
  | { gaps: false; motion: true; motion_mean: number; span_s: number }
  | { gaps: false; motion: false; bpm: number; bpm_chrom: number; snr: number; delta: number; motion_mean: number; span_s: number; quality: number };

export function analyzeWindow(ts0: number[], rgb0: Rgb[], motion0: number[], p: RppgParams): RppgWindow {
  const tEnd = ts0[ts0.length - 1]!;
  const w = Math.min(p.window_s, tEnd - ts0[0]!);
  const n = Math.floor(w * p.fs_hz + 1e-6);
  const bigL = Math.floor(p.pos_window_s * p.fs_hz + 0.5);
  if (n < Math.max(16, bigL + 1)) return { gaps: true };
  let idx = 0; // searchsorted(ts, tEnd - w, 'left')
  while (idx < ts0.length && ts0[idx]! < tEnd - w) idx++;
  const lo = Math.max(idx - 1, 0);
  const mot = motion0.slice(idx);
  const ts = ts0.slice(lo), rgb = rgb0.slice(lo);
  if (ts.length < 2) return { gaps: true };
  const rate = (ts.length - 1) / (ts[ts.length - 1]! - ts[0]!);
  let maxGap = 0;
  for (let i = 1; i < ts.length; i++) maxGap = Math.max(maxGap, ts[i]! - ts[i - 1]!);
  if (maxGap > p.max_gap_s || rate < p.min_rate_hz) return { gaps: true };
  const motMean = mean(mot);
  let nMoving = 0;
  for (const m of mot) if (m > p.motion_max) nMoving++;
  if (nMoving / mot.length > p.motion_frac_max) return { gaps: false, motion: true, motion_mean: motMean, span_s: w };
  const grid = gridTimes(tEnd, n, p.fs_hz);
  const ch: number[][] = [];
  for (let c = 0; c < 3; c++) {
    const y = interp(grid, ts, rgb.map((v) => v[c]!));
    if (!(mean(y) > 1e-6)) return { gaps: true };
    ch.push(y);
  }
  const [r, g, b] = ch as [number[], number[], number[]];
  const nFft = fftSize(n, p.zero_pad_factor);
  const pos = spectrum(detrendLinear(posSignal(r, g, b, p.fs_hz, p.pos_window_s)), nFft, p);
  const chr = spectrum(chromSignal(r, g, b, nFft, p.fs_hz, p), nFft, p);
  const bpm = 60 * pos.f, bpmChrom = 60 * chr.f;
  const delta = bpm - bpmChrom;
  return { gaps: false, motion: false, bpm, bpm_chrom: bpmChrom, snr: pos.snr, delta, motion_mean: motMean, span_s: w, quality: qualityScore(pos.snr, delta, motMean, p) };
}

// ------------------------------------------------------------------ streaming

export class FaceRPPG {
  private t: number[] = [];
  private rgb: Rgb[] = [];
  private mot: number[] = [];
  private bright: Array<[number, number]> = [];
  private validSince: number | null = null;
  private lightUntil = -Infinity;
  private epoch = 0;
  private state: 'ok' | 'no_face' | 'too_small' | 'skin_filter' | 'lighting' = 'ok';
  private nextOut: number | null = null;
  private lastT: number | null = null;

  constructor(private readonly p: RppgParams) {
    assertRppgParams(p);
  }

  reset(): void {
    this.t = []; this.rgb = []; this.mot = []; this.bright = [];
    this.validSince = null; this.lightUntil = -Infinity; this.epoch = 0; this.state = 'ok';
    this.nextOut = null; this.lastT = null;
  }

  /** Seconds of continuous valid face so far (0 if none). */
  validSeconds(t: number): number {
    return this.validSince === null ? 0 : t - this.validSince;
  }

  private clear(state: 'no_face' | 'too_small' | 'skin_filter' | 'lighting'): void {
    this.t = []; this.rgb = []; this.mot = [];
    this.validSince = null; this.state = state;
  }

  private brightnessBad(t: number, b: number): boolean {
    const p = this.p;
    const dq = this.bright;
    dq.push([t, b]);
    let drop = 0;
    while (dq.length - drop >= 2 && dq[drop + 1]![0] <= t - p.brightness_lag_s) drop++;
    if (drop) dq.splice(0, drop);
    let bad = !(p.min_brightness <= b && b <= p.max_brightness);
    const [t0, b0] = dq[0]!;
    if (t - t0 >= 0.5 * p.brightness_lag_s) {
      if (Math.abs(b - b0) / Math.max(b0, 1) / (t - t0) > p.max_brightness_rate) bad = true;
    }
    return bad;
  }

  update(t: number, roiRgb: Rgb | null, roiPx: number, faceMotion: number, brightness: number, skinFrac = 1): RppgUpdate | null {
    const p = this.p;
    if (this.lastT !== null && !(t > this.lastT)) return null;
    this.lastT = t;
    if (this.nextOut === null) this.nextOut = t + p.step_s;
    if (roiRgb === null) { this.bright = []; this.clear('no_face'); }
    else if (roiPx < p.min_roi_px) { this.bright = []; this.clear('too_small'); }
    else if (skinFrac < p.min_skin_frac) { this.bright = []; this.clear('skin_filter'); }
    else {
      if (this.brightnessBad(t, brightness)) this.lightUntil = t + p.lighting_hold_s;
      if (t < this.lightUntil) this.clear('lighting');
      else {
        this.state = 'ok';
        if (this.validSince === null) { this.validSince = t; this.epoch++; }
        this.t.push(t); this.rgb.push(roiRgb); this.mot.push(faceMotion);
        let drop = 0;
        while (this.t.length - drop >= 2 && this.t[drop + 1]! < t - p.window_s) drop++;
        if (drop) { this.t.splice(0, drop); this.rgb.splice(0, drop); this.mot.splice(0, drop); }
      }
    }
    if (t < this.nextOut) return null;
    while (this.nextOut <= t) this.nextOut += p.step_s;
    return this.emit(t);
  }

  private emit(t: number): RppgUpdate {
    const p = this.p;
    const out: RppgUpdate = { t, bpm: null, bpm_chrom: null, quality: 0, reported: false, reason: 'ok', snr: null, span_s: null, track_epoch: this.epoch };
    if (this.state !== 'ok') { out.reason = this.state; return out; }
    if (this.validSince === null || t - this.validSince < p.settle_s) { out.reason = 'settling'; return out; }
    const a = analyzeWindow(this.t, this.rgb, this.mot, p);
    if (a.gaps) out.reason = 'gaps';
    else if (a.motion) { out.reason = 'motion'; out.span_s = a.span_s; }
    else if (![a.bpm, a.bpm_chrom, a.snr, a.quality].every(Number.isFinite)) { out.reason = 'low_signal'; out.span_s = a.span_s; }
    else {
      out.span_s = a.span_s;
      out.bpm = a.bpm; out.bpm_chrom = a.bpm_chrom; out.quality = a.quality; out.snr = a.snr;
      if (a.snr < p.min_snr) out.reason = 'low_signal';
      else if (Math.abs(a.delta) > p.agree_bpm) out.reason = 'disagree';
      else { out.reason = 'ok'; out.reported = true; }
    }
    return out;
  }
}

/** Fixture runner: mirrors carelens.rppg.run_face_rppg. */
export function runFaceRppg(
  inp: { t: number[]; roi_rgb: Array<Rgb | null>; roi_px: number[]; face_motion: number[]; brightness: number[]; skin_frac?: number[] },
  p: RppgParams,
): { updates: RppgUpdate[] } {
  const fr = new FaceRPPG(p);
  const updates: RppgUpdate[] = [];
  inp.t.forEach((t, i) => {
    const u = fr.update(t, inp.roi_rgb[i]!, inp.roi_px[i]!, inp.face_motion[i]!, inp.brightness[i]!, inp.skin_frac ? inp.skin_frac[i]! : 1);
    if (u) updates.push(u);
  });
  return { updates };
}
