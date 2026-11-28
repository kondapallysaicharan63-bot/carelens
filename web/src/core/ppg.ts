// Contact PPG (fingertip on the phone's back camera + torch). DOM-free TypeScript port of
// carelens/ppg.py: same grid, FFT size, taper, peak rules, gates and quality formula, driven by
// shared/params.json -> contact_ppg. Read the Python docstring for the full signal design.
// Wellbeing demonstration only. Not a medical device.

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
}

const EPS = 1e-30;

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

// ------------------------------------------------------------------ FFT (iterative radix-2, no deps)

/** In-place forward complex FFT; length must be a power of two. */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n & (n - 1)) throw new Error('fft length must be a power of two');
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]!; re[i] = re[j]!; re[j] = t;
      t = im[i]!; im[i] = im[j]!; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const wr = new Float64Array(half);
    const wi = new Float64Array(half);
    for (let k = 0; k < half; k++) {
      const a = (-2 * Math.PI * k) / len;
      wr[k] = Math.cos(a);
      wi[k] = Math.sin(a);
    }
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const xr = re[b]! * wr[k]! - im[b]! * wi[k]!;
        const xi = re[b]! * wi[k]! + im[b]! * wr[k]!;
        re[b] = re[a]! - xr; im[b] = im[a]! - xi;
        re[a] = re[a]! + xr; im[a] = im[a]! + xi;
      }
    }
  }
}

/** In-place inverse FFT (conjugate trick, 1/N scaling). */
export function ifft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i]!;
  fft(re, im);
  for (let i = 0; i < n; i++) { re[i] = re[i]! / n; im[i] = -im[i]! / n; }
}

// ------------------------------------------------------------------ pure building blocks

export function gridTimes(tEnd: number, n: number, fs: number): number[] {
  const g = new Array<number>(n);
  for (let k = 0; k < n; k++) g[k] = tEnd - (n - 1 - k) / fs;
  return g;
}

/** Linear interpolation, holding the end values outside [xp[0], xp[last]] (like numpy.interp). */
export function interp(x: number[], xp: number[], fp: number[]): number[] {
  const out = new Array<number>(x.length);
  let j = 0;
  const last = xp.length - 1;
  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    if (v <= xp[0]!) { out[i] = fp[0]!; continue; }
    if (v >= xp[last]!) { out[i] = fp[last]!; continue; }
    while (xp[j + 1]! <= v) j++;
    const slope = (fp[j + 1]! - fp[j]!) / (xp[j + 1]! - xp[j]!);
    out[i] = slope * (v - xp[j]!) + fp[j]!;
  }
  return out;
}

const mean = (a: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]!;
  return s / a.length;
};

/** Subtract the closed-form least-squares line (against the sample index); divide by the raw mean. */
export function detrendRelative(y: number[]): { rel: number[]; mean: number } {
  const n = y.length;
  const ybar = mean(y);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const kk = i - (n - 1) / 2;
    num += kk * (y[i]! - ybar);
    den += kk * kk;
  }
  const slope = num / den;
  const rel = new Array<number>(n);
  for (let i = 0; i < n; i++) rel[i] = (y[i]! - (ybar + slope * (i - (n - 1) / 2))) / ybar;
  return { rel, mean: ybar };
}

/** Linear-interpolation percentile (numpy default), q in [0, 100]. */
export function percentile(x: number[], q: number): number {
  const s = Float64Array.from(x).sort();
  const h = ((s.length - 1) * q) / 100;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo]! + (s[hi]! - s[lo]!) * (h - lo);
}

export const perfusionIndex = (rel: number[], p: ContactPpgParams): number =>
  (percentile(rel, p.pi_hi_pct) - percentile(rel, p.pi_lo_pct)) * 100;

export function hann(n: number): number[] {
  const w = new Array<number>(n);
  for (let k = 0; k < n; k++) w[k] = 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (n - 1));
  return w;
}

export function fftSize(n: number, factor: number): number {
  const need = factor * n;
  let m = 1;
  while (m < need) m *= 2;
  return m;
}

/** |fft(hann * rel, nFft)|^2 for bins 0..nFft/2. */
export function powerSpectrum(rel: number[], nFft: number): Float64Array {
  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  const w = hann(rel.length);
  for (let i = 0; i < rel.length; i++) re[i] = rel[i]! * w[i]!;
  fft(re, im);
  const p = new Float64Array(nFft / 2 + 1);
  for (let k = 0; k < p.length; k++) p[k] = re[k]! * re[k]! + im[k]! * im[k]!;
  return p;
}

export function bandBins(nFft: number, fs: number, lo: number, hi: number): [number, number] {
  return [Math.ceil((lo * nFft) / fs - 1e-9), Math.floor((hi * nFft) / fs + 1e-9)];
}

function argmax(a: ArrayLike<number>, from: number, to: number): number {
  let best = from;
  for (let k = from + 1; k <= to; k++) if (a[k]! > a[best]!) best = k;
  return best;
}

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

/** Peak offset in [-0.5, 0.5] bins from three linear samples. */
export function parabolicLin(a: number, b: number, c: number): number {
  const den = a - 2 * b + c;
  return den === 0 ? 0 : clamp((0.5 * (a - c)) / den, -0.5, 0.5);
}

/** Peak offset in [-0.5, 0.5] bins at k, parabola through log power. */
export function parabolicLog(power: ArrayLike<number>, k: number): number {
  return parabolicLin(Math.log(power[k - 1]! + EPS), Math.log(power[k]! + EPS), Math.log(power[k + 1]! + EPS));
}

function peakBin(power: ArrayLike<number>, nFft: number, fs: number, f: number, hw: number, kLo: number, kHi: number): number {
  const lo = Math.max(kLo, Math.ceil(((f - hw) * nFft) / fs - 1e-9));
  const hi = Math.min(kHi, Math.floor(((f + hw) * nFft) / fs + 1e-9));
  return hi < lo ? -1 : argmax(power, lo, hi);
}

/** S(f) = P(f) + w2 P(2f) + w3 P(3f); P = peak power within +-harm_halfwidth; terms at/above Nyquist are 0. */
function harmonicScore(power: ArrayLike<number>, nFft: number, fs: number, f: number, p: ContactPpgParams): number {
  let sc = 0;
  for (const [mult, w] of [[1, 1], [2, p.harm_w2], [3, p.harm_w3]] as const) {
    if (mult * f < fs / 2) {
      const k = peakBin(power, nFft, fs, mult * f, p.harm_halfwidth_hz, 0, nFft / 2);
      if (k >= 0) sc += w * power[k]!;
    }
  }
  return sc;
}

/** Refined in-band fundamental (Hz): candidates f0, f0/2 and 2 f0 scored by the harmonic sum; ties go to the higher f. */
export function spectralPeak(power: Float64Array, nFft: number, fs: number, p: ContactPpgParams): number {
  const [kLo, kHi] = bandBins(nFft, fs, p.band_lo_hz, p.band_hi_hz);
  const k0 = argmax(power, kLo, kHi);
  const f0 = (k0 * fs) / nFft;
  let bestK = k0, bestS = harmonicScore(power, nFft, fs, f0, p);
  for (const fc of [f0 / 2, f0 * 2]) {
    if (!(p.band_lo_hz <= fc && fc <= p.band_hi_hz)) continue;
    const kc = peakBin(power, nFft, fs, fc, p.harm_halfwidth_hz, kLo, kHi);
    if (kc < 0) continue;
    const sc = harmonicScore(power, nFft, fs, (kc * fs) / nFft, p);
    if (sc > bestS || (sc === bestS && kc > bestK)) { bestK = kc; bestS = sc; }
  }
  return ((bestK + parabolicLog(power, bestK)) * fs) / nFft;
}

export function spectralSnr(power: Float64Array, nFft: number, fs: number, fStar: number, p: ContactPpgParams): number {
  const [kLo, kHi] = bandBins(nFft, fs, p.band_lo_hz, p.band_hi_hz);
  let sig = 0, noise = 0;
  const two = 2 * fStar < fs / 2;
  for (let k = kLo; k <= kHi; k++) {
    const f = (k * fs) / nFft;
    const inSig = Math.abs(f - fStar) <= p.snr_halfwidth_hz || (two && Math.abs(f - 2 * fStar) <= p.snr_halfwidth_hz);
    if (inSig) sig += power[k]!; else noise += power[k]!;
  }
  return sig / (noise + EPS);
}

/** Raised-cosine band-pass gain at frequency f (Hz). */
export function taper(f: number, p: ContactPpgParams): number {
  if (f >= p.taper_stop_lo_hz && f < p.band_lo_hz) {
    return 0.5 - 0.5 * Math.cos((Math.PI * (f - p.taper_stop_lo_hz)) / (p.band_lo_hz - p.taper_stop_lo_hz));
  }
  if (f >= p.band_lo_hz && f <= p.band_hi_hz) return 1;
  if (f > p.band_hi_hz && f < p.taper_stop_hi_hz) {
    return 0.5 + 0.5 * Math.cos((Math.PI * (f - p.band_hi_hz)) / (p.taper_stop_hi_hz - p.band_hi_hz));
  }
  return 0;
}

/** FFT-domain band-pass of the zero-padded signal, truncated back to the window length. */
export function bandpass(rel: number[], nFft: number, fs: number, p: ContactPpgParams): number[] {
  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  for (let i = 0; i < rel.length; i++) re[i] = rel[i]!;
  fft(re, im);
  for (let k = 0; k < nFft; k++) {
    const g = taper(((k <= nFft / 2 ? k : nFft - k) * fs) / nFft, p);
    re[k] = re[k]! * g; im[k] = im[k]! * g;
  }
  ifft(re, im);
  return Array.from(re.subarray(0, rel.length));
}

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
  | { gaps: false; channel: 'R' | 'G'; bpm: number; bpm_ibi: number | null; snr: number; pi: number; delta: number | null; ibis_ms: number[]; quality: number };

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
  return { gaps: false, channel: best.channel, bpm, bpm_ibi: fIbi, snr: best.snr, pi, delta, ibis_ms: ibis, quality: qualityScore(best.snr, delta, pi, p) };
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

  constructor(private readonly p: ContactPpgParams) {
    assertContactPpgParams(p);
  }

  reset(): void {
    this.t = []; this.r = []; this.g = [];
    this.prevR = null; this.contactSince = null; this.epoch = 0; this.state = 'ok';
    this.nextOut = null; this.lastT = null; this.lastIbisMs = [];
  }

  /** Seconds of continuous contact so far (0 if none). */
  contactSeconds(t: number): number {
    return this.contactSince === null ? 0 : t - this.contactSince;
  }

  private clear(r: number, state: 'no_finger' | 'saturated'): void {
    this.t = []; this.r = []; this.g = [];
    this.contactSince = null; this.prevR = r; this.state = state;
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
    const out: PpgUpdate = { t, bpm: null, bpm_ibi: null, quality: 0, reported: false, reason: 'ok', snr: null, pi: null, channel: null, contact_epoch: this.epoch };
    if (this.state !== 'ok') { out.reason = this.state; return out; }
    if (this.contactSince === null || t - this.contactSince < p.settle_s) { out.reason = 'settling'; return out; }
    const a = analyzeWindow(this.t, this.r, this.g, p);
    if (a.gaps) { out.reason = 'gaps'; return out; }
    this.lastIbisMs = a.ibis_ms;
    out.bpm = a.bpm;
    out.bpm_ibi = a.bpm_ibi;
    out.quality = a.quality; out.snr = a.snr; out.pi = a.pi; out.channel = a.channel;
    if (a.snr < p.min_snr || !(p.pi_min <= a.pi && a.pi <= p.pi_max) || a.delta === null) out.reason = 'low_signal';
    else if (Math.abs(a.delta) > p.agree_bpm) out.reason = 'disagree';
    else { out.reason = 'ok'; out.reported = true; }
    return out;
  }
}

/** Fixture runner: mirrors carelens.ppg.run_contact_ppg. */
export function runContactPpg(
  inp: { t: number[]; r: number[]; g: number[]; b: number[]; clip_frac: number[] },
  p: ContactPpgParams,
): { updates: PpgUpdate[]; final_ibis_ms: number[] } {
  const cp = new ContactPPG(p);
  const updates: PpgUpdate[] = [];
  inp.t.forEach((t, i) => {
    const u = cp.update(t, inp.r[i]!, inp.g[i]!, inp.b[i]!, inp.clip_frac[i]!);
    if (u) updates.push(u);
  });
  return { updates, final_ibis_ms: [...cp.lastIbisMs] };
}
