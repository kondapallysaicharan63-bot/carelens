// Contact PPG (fingertip on the phone's back camera + torch). DOM-free TypeScript port of
// carelens/ppg.py: same grid, FFT size, taper, peak rules, gates and quality formula, driven by
// shared/params.json -> contact_ppg. Read the Python docstring for the full signal design.
// Wellbeing demonstration only. Not a medical device.

import {
  EPS, bandBins, bandpass, clamp, detrendRelative, fftSize, gridTimes, interp, mean, parabolicLin, percentile,
  powerSpectrum, spectralPeak, spectralSnr,
} from './dsp';

// W1 tests and callers import the shared building blocks from here; they now live in ./dsp.
export {
  bandBins, bandpass, detrendRelative, fft, fftSize, gridTimes, hann, ifft, interp, parabolicLin, parabolicLog, percentile,
  powerSpectrum, spectralPeak, spectralSnr, taper,
} from './dsp';

import type { HrvParams } from './hrv';

export interface ContactPpgParams {
  window_s: number; step_s: number; fs_hz: number; max_gap_s: number; min_rate_hz: number;
  band_lo_hz: number; band_hi_hz: number; taper_stop_lo_hz: number; taper_stop_hi_hz: number;
  zero_pad_factor: number; snr_halfwidth_hz: number; harm_halfwidth_hz: number; harm_w2: number; harm_w3: number;
  pi_lo_pct: number; pi_hi_pct: number; beat_max_bpm: number;
  min_prom: number; min_ibis: number; finger_r_min: number; finger_ratio_min: number;
  finger_dr_max: number; max_clip_frac: number; settle_s: number; min_snr: number; agree_bpm: number;
  pi_min: number; pi_max: number; q_snr_full_db: number; q_pi_full_lo: number; q_pi_full_hi: number;
  q_w_snr: number; q_w_agree: number; q_w_pi: number;
}

export type PpgReason = 'ok' | 'no_finger' | 'saturated' | 'settling' | 'low_signal' | 'disagree' | 'gaps';

export interface PpgUpdate {
  t: number;
  bpm: number | null;
  bpm_ibi: number | null;
  quality: number;
  reported: boolean;
  reason: PpgReason;
  snr: number | null;
  pi: number | null;
  channel: 'R' | 'G' | null;
  /** counts contact restarts; a measurement must restart when it changes */
  contact_epoch: number;
  /** analysed window length (s); null when no window was analysed */
  span_s: number | null;
}


/** Consistency checks mirroring carelens.params.contact_ppg_errors; returns the problems found. */
export function contactPpgErrors(p: ContactPpgParams): string[] {
  const e: string[] = [];
  if (!(0 < p.taper_stop_lo_hz && p.taper_stop_lo_hz < p.band_lo_hz && p.band_lo_hz < p.band_hi_hz && p.band_hi_hz < p.taper_stop_hi_hz && p.taper_stop_hi_hz < p.fs_hz / 2)) {
    e.push('need 0 < taper_stop_lo < band_lo < band_hi < taper_stop_hi < Nyquist');
  }
  if (!(0 < p.pi_min && p.pi_min < p.q_pi_full_lo && p.q_pi_full_lo < p.q_pi_full_hi && p.q_pi_full_hi < p.pi_max)) {
    e.push('need 0 < pi_min < q_pi_full_lo < q_pi_full_hi < pi_max');
  }
  if (Math.min(p.q_w_snr, p.q_w_agree, p.q_w_pi) < 0 || Math.abs(p.q_w_snr + p.q_w_agree + p.q_w_pi - 1) > 1e-9) {
    e.push('quality weights must be >= 0 and sum to 1');
  }
  if (!(p.min_snr > 0 && p.q_snr_full_db > 10 * Math.log10(Math.max(p.min_snr, 1e-300)))) {
    e.push('need min_snr > 0 and q_snr_full_db above the gate in dB');
  }
  for (const k of ['window_s', 'step_s', 'fs_hz', 'max_gap_s', 'min_rate_hz', 'settle_s', 'agree_bpm', 'snr_halfwidth_hz',
    'harm_halfwidth_hz', 'beat_max_bpm', 'min_prom', 'finger_dr_max', 'finger_ratio_min'] as const) {
    if (!(p[k] > 0)) e.push(`${k} must be > 0`);
  }
  if (p.zero_pad_factor < 1 || p.min_ibis < 1) e.push('zero_pad_factor and min_ibis must be >= 1');
  if (!(0 <= p.pi_lo_pct && p.pi_lo_pct < p.pi_hi_pct && p.pi_hi_pct <= 100)) e.push('need 0 <= pi_lo_pct < pi_hi_pct <= 100');
  if (p.harm_w2 < 0 || p.harm_w3 < 0 || !(0 < p.max_clip_frac && p.max_clip_frac <= 1)) e.push('harm weights >= 0 and 0 < max_clip_frac <= 1');
  return e;
}

export function assertContactPpgParams(p: ContactPpgParams): ContactPpgParams {
  const errs = contactPpgErrors(p);
  if (errs.length) throw new Error('invalid contact_ppg params: ' + errs.join('; '));
  return p;
}

export const perfusionIndex = (rel: number[], p: ContactPpgParams): number =>
  (percentile(rel, p.pi_hi_pct) - percentile(rel, p.pi_lo_pct)) * 100;

function prominence(x: number[], i: number): number {
  const v = x[i]!;
  let lmin = v, rmin = v;
  for (let j = i - 1; j >= 0 && x[j]! < v; j--) lmin = Math.min(lmin, x[j]!);
  for (let j = i + 1; j < x.length && x[j]! < v; j++) rmin = Math.min(rmin, x[j]!);
  return v - Math.max(lmin, rmin);
}

/** Sub-sample beat positions (in samples), independent of the spectral estimate; see carelens/ppg.py find_beats. */
export function findBeats(x: number[], fs: number, spacingS: number, minPromAbs: number): number[] {
  const prom = new Map<number, number>();
  const cand: number[] = [];
  for (let i = 1; i < x.length - 1; i++) {
    if (x[i - 1]! < x[i]! && x[i]! >= x[i + 1]!) {
      const pr = prominence(x, i);
      prom.set(i, pr);
      if (pr >= minPromAbs) cand.push(i);
    }
  }
  cand.sort((a, b) => x[b]! - x[a]! || a - b);
  const minD = spacingS * fs;
  let kept: number[] = [];
  for (const i of cand) if (kept.every((j) => Math.abs(i - j) >= minD)) kept.push(i);
  kept.sort((a, b) => a - b);
  if (kept.length >= 3) {
    const d: number[] = [];
    for (let k = 0; k + 1 < kept.length; k++) d.push(kept[k + 1]! - kept[k]!);
    d.sort((a, b) => a - b);
    const m = Math.floor(d.length / 2);
    const med = d.length % 2 ? d[m]! : 0.5 * (d[m - 1]! + d[m]!);
    const keep2: number[] = [];
    for (const i of [...kept].sort((a, b) => prom.get(b)! - prom.get(a)! || a - b)) {
      if (keep2.every((j) => Math.abs(i - j) >= 0.5 * med)) keep2.push(i);
    }
    kept = keep2.sort((a, b) => a - b);
  }
  return kept.map((i) => i + parabolicLin(x[i - 1]!, x[i]!, x[i + 1]!));
}

const clip01 = (v: number): number => clamp(v, 0, 1);

export function qualityScore(snr: number, deltaBpm: number | null, pi: number, p: ContactPpgParams): number {
  const gateDb = 10 * Math.log10(p.min_snr);
  const qSnr = clip01((10 * Math.log10(Math.max(snr, EPS)) - gateDb) / (p.q_snr_full_db - gateDb));
  const qAgree = deltaBpm === null ? 0 : clip01(1 - Math.abs(deltaBpm) / p.agree_bpm);
  const qPi = Math.min(clip01((pi - p.pi_min) / (p.q_pi_full_lo - p.pi_min)), clip01((p.pi_max - pi) / (p.pi_max - p.q_pi_full_hi)));
  return p.q_w_snr * qSnr + p.q_w_agree * qAgree + p.q_w_pi * qPi;
}

// ------------------------------------------------------------------ window analysis

export type WindowAnalysis =
  | { gaps: true }
  | { gaps: false; span_s: number; channel: 'R' | 'G'; bpm: number; bpm_ibi: number | null; snr: number; pi: number; delta: number | null; ibis_ms: number[]; quality: number };

export function analyzeWindow(ts0: number[], rs0: number[], gs0: number[], p: ContactPpgParams): WindowAnalysis {
  const tEnd = ts0[ts0.length - 1]!;
  const w = Math.min(p.window_s, tEnd - ts0[0]!);
  const n = Math.floor(w * p.fs_hz + 1e-6);
  if (n < 16) return { gaps: true };
  let idx = 0; // searchsorted(ts, tEnd - w, 'left')
  while (idx < ts0.length && ts0[idx]! < tEnd - w) idx++;
  const lo = Math.max(idx - 1, 0);
  const ts = ts0.slice(lo), rs = rs0.slice(lo), gs = gs0.slice(lo);
  if (ts.length < 2) return { gaps: true };
  const rate = (ts.length - 1) / (ts[ts.length - 1]! - ts[0]!);
  let maxGap = 0;
  for (let i = 1; i < ts.length; i++) maxGap = Math.max(maxGap, ts[i]! - ts[i - 1]!);
  if (maxGap > p.max_gap_s || rate < p.min_rate_hz) return { gaps: true };
  const grid = gridTimes(tEnd, n, p.fs_hz);
  const nFft = fftSize(n, p.zero_pad_factor);
  let best: { channel: 'R' | 'G'; rel: number[]; fSpec: number; snr: number } | null = null;
  for (const [name, raw] of [['R', rs], ['G', gs]] as const) {
    const y = interp(grid, ts, raw);
    if (!(mean(y) > 1e-6)) continue;
    const { rel } = detrendRelative(y);
    const power = powerSpectrum(rel, nFft);
    const fSpec = spectralPeak(power, nFft, p.fs_hz, p);
    const snr = spectralSnr(power, nFft, p.fs_hz, fSpec, p);
    if (best === null || snr > best.snr) best = { channel: name, rel, fSpec, snr };
  }
  if (best === null) return { gaps: true };
  const pi = perfusionIndex(best.rel, p);
  const bp = bandpass(best.rel, nFft, p.fs_hz, p);
  const bm = mean(bp);
  let ss = 0;
  for (const v of bp) ss += (v - bm) * (v - bm);
  const std = Math.sqrt(ss / bp.length);
  const beats = findBeats(bp, p.fs_hz, 60 / p.beat_max_bpm, p.min_prom * std);
  const bt = beats.map((b) => grid[0]! + b / p.fs_hz);
  const ibis: number[] = [];
  for (let i = 0; i + 1 < bt.length; i++) ibis.push((bt[i + 1]! - bt[i]!) * 1000);
  let fIbi: number | null = null;
  if (ibis.length >= p.min_ibis) {
    const s = [...ibis].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    const med = s.length % 2 ? s[m]! : 0.5 * (s[m - 1]! + s[m]!);
    fIbi = 60000 / med;
  }
  const bpm = 60 * best.fSpec;
  const delta = fIbi === null ? null : bpm - fIbi;
  return { gaps: false, span_s: w, channel: best.channel, bpm, bpm_ibi: fIbi, snr: best.snr, pi, delta, ibis_ms: ibis, quality: qualityScore(best.snr, delta, pi, p) };
}

// ------------------------------------------------------------------ streaming

export class ContactPPG {
  /** IBIs (ms) of the most recent analysed window (W6 HRV will use them). */
  lastIbisMs: number[] = [];
  private t: number[] = [];
  private r: number[] = [];
  private g: number[] = [];
  private prevR: number | null = null;
  private contactSince: number | null = null;
  private epoch = 0;
  private state: 'ok' | 'no_finger' | 'saturated' = 'ok';
  private nextOut: number | null = null;
  private lastT: number | null = null;
  // raw samples of the current contact epoch AFTER settling (W6 HRV), frozen at hrv.max_duration_s; hrv undefined: not recorded
  private segT: number[] = [];
  private segR: number[] = [];
  private segG: number[] = [];
  private segUpd = 0; // 1 s updates since settling ended (current epoch)
  private segOk = 0; // ... of which steady: reported ok, or every W1 gate passed except SNR

  constructor(private readonly p: ContactPpgParams, private readonly hrv?: HrvParams) {
    assertContactPpgParams(p);
  }

  /** `[t, r, g, nUpdates, nOk]` of the current contact epoch (W6 HRV input). */
  segment(): { t: number[]; r: number[]; g: number[]; nUpd: number; nOk: number } {
    return { t: this.segT, r: this.segR, g: this.segG, nUpd: this.segUpd, nOk: this.segOk };
  }

  /** Length (s) of the buffered post-settling segment; 0 before it has two samples. */
  segmentSeconds(): number {
    return this.segT.length >= 2 ? this.segT[this.segT.length - 1]! - this.segT[0]! : 0;
  }

  private resetSegment(): void {
    this.segT = []; this.segR = []; this.segG = []; this.segUpd = 0; this.segOk = 0;
  }

  reset(): void {
    this.t = []; this.r = []; this.g = [];
    this.prevR = null; this.contactSince = null; this.epoch = 0; this.state = 'ok';
    this.nextOut = null; this.lastT = null; this.lastIbisMs = [];
    this.resetSegment();
  }

  /** Seconds of continuous contact so far (0 if none). */
  contactSeconds(t: number): number {
    return this.contactSince === null ? 0 : t - this.contactSince;
  }

  private clear(r: number, state: 'no_finger' | 'saturated'): void {
    this.t = []; this.r = []; this.g = [];
    this.contactSince = null; this.prevR = r; this.state = state;
    this.resetSegment();
  }

  update(t: number, r: number, g: number, _b = 0, clipFrac = 0): PpgUpdate | null {
    const p = this.p;
    if (this.lastT !== null && !(t > this.lastT)) return null;
    this.lastT = t;
    if (this.nextOut === null) this.nextOut = t + p.step_s;
    const jolt = this.prevR !== null && Math.abs(r - this.prevR) > p.finger_dr_max;
    if (r < p.finger_r_min || r / (g + 1) < p.finger_ratio_min || jolt) this.clear(r, 'no_finger');
    else if (clipFrac > p.max_clip_frac) this.clear(r, 'saturated');
    else {
      this.state = 'ok';
      this.prevR = r;
      if (this.contactSince === null) { this.contactSince = t; this.epoch++; }
      this.t.push(t); this.r.push(r); this.g.push(g);
      if (this.hrv && t - this.contactSince >= p.settle_s && (this.segT.length === 0 || t - this.segT[0]! <= this.hrv.max_duration_s)) { this.segT.push(t); this.segR.push(r); this.segG.push(g); }
      let drop = 0;
      while (this.t.length - drop >= 2 && this.t[drop + 1]! < t - p.window_s) drop++;
      if (drop) { this.t.splice(0, drop); this.r.splice(0, drop); this.g.splice(0, drop); }
    }
    if (t < this.nextOut) return null;
    while (this.nextOut <= t) this.nextOut += p.step_s;
    return this.emit(t);
  }

  private emit(t: number): PpgUpdate {
    const p = this.p;
    const out: PpgUpdate = { t, bpm: null, bpm_ibi: null, quality: 0, reported: false, reason: 'ok', snr: null, pi: null, channel: null, contact_epoch: this.epoch, span_s: null };
    if (this.state !== 'ok') { out.reason = this.state; return out; }
    if (this.contactSince === null || t - this.contactSince < p.settle_s) { out.reason = 'settling'; return out; }
    const a = analyzeWindow(this.t, this.r, this.g, p);
    const counting = !!this.hrv && this.segT.length > 0 && t - this.segT[0]! <= this.hrv.max_duration_s;
    if (counting) this.segUpd++;
    if (a.gaps) { out.reason = 'gaps'; return out; }
    this.lastIbisMs = a.ibis_ms;
    out.bpm = a.bpm;
    out.bpm_ibi = a.bpm_ibi;
    out.quality = a.quality; out.snr = a.snr; out.pi = a.pi; out.channel = a.channel; out.span_s = a.span_s;
    const piOk = p.pi_min <= a.pi && a.pi <= p.pi_max;
    if (counting && a.snr < p.min_snr && piOk && a.delta !== null && Math.abs(a.delta) <= p.agree_bpm) this.segOk++; // steady for HRV
    if (a.snr < p.min_snr || !piOk || a.delta === null) out.reason = 'low_signal';
    else if (Math.abs(a.delta) > p.agree_bpm) out.reason = 'disagree';
    else { out.reason = 'ok'; out.reported = true; if (counting) this.segOk++; }
    return out;
  }
}

/** Fixture runner: mirrors carelens.ppg.run_contact_ppg. */
export function runContactPpg(
  inp: { t: number[]; r: number[]; g: number[]; b: number[]; clip_frac: number[] },
  p: ContactPpgParams,
): { updates: PpgUpdate[]; final_ibis_ms: number[] } {
  const cp = new ContactPPG(p); // no HRV segment: the W1 fixtures do not need it
  const updates: PpgUpdate[] = [];
  inp.t.forEach((t, i) => {
    const u = cp.update(t, inp.r[i]!, inp.g[i]!, inp.b[i]!, inp.clip_frac[i]!);
    if (u) updates.push(u);
  });
  return { updates, final_ibis_ms: [...cp.lastIbisMs] };
}
