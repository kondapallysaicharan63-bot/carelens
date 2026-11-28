// Audio Tier 1 (W4): DOM-free TypeScript port of carelens/audio.py (read its docstring for the design and limits):
// 50 Hz log band-power envelope from 16 kHz audio, breathing rate from breath sounds (with the double-rate guard),
// (never 2x or 1/2x: the doubled-rate rule needs bump evidence, see carelens/audio.py), and the generic hysteresis/merge/refractory label-event detector (Cough now, Tier 2 labels later).
// Audio is processed in memory only. Wellbeing trend only. Not a medical device.

import {
  bandBins, clamp, detrendLinear, fft, fftSize, hann, mean, peakBin, percentile, powerSpectrum,
  spectralPeak, spectralSnr, type PeakParams,
} from './dsp';

export interface AudioBreathingParams extends PeakParams {
  window_s: number; step_s: number; zero_pad_factor: number; settle_s: number; min_range_log10: number;
  clip_level: number; clip_frac_max: number; speech_thr: number; speech_frac_max: number; min_snr: number;
  q_snr_full_db: number; sub_ratio: number; smooth_s: number; bump_prom_frac: number; min_bump_s: number;
  min_bumps: number; min_alt_bumps: number; rate_tol: number; alt_min: number; width_min: number;
}
export interface LabelEventParams {
  label: string; on_thr: number; off_thr: number; min_frames: number; merge_gap_s: number; refractory_s: number;
}
/** W10 per-label detectors (shared/params.json -> audio_events); validated by audioEventsErrors in sleepSounds.ts */
export interface AudioEventsParams {
  corroborate_s: number; cue_labels: string[];
  snoring: LabelEventParams; screaming: LabelEventParams; crying: LabelEventParams; glass: LabelEventParams; thump: LabelEventParams; speech: LabelEventParams;
}
export interface AudioParams {
  classifier: string; labels: string[]; fs_hz: number;
  envelope: { frame_len: number; hop: number; band_lo_hz: number; band_hi_hz: number; eps: number };
  yamnet: { window_samples: number; hop_samples: number; speech_label: string };
  breathing: AudioBreathingParams;
  cough: LabelEventParams;
}

export type BreathSoundReason = 'ok' | 'no_mic' | 'too_quiet' | 'clipping' | 'talking' | 'settling' | 'low_signal' | 'ambiguous';
export const YAMNET_WINDOW_S = 0.975;

export interface BreathSoundUpdate {
  t: number;
  bpm: number | null;
  quality: number;
  reported: boolean;
  reason: BreathSoundReason;
  snr: number | null;
  /** number of complete envelope bumps found (null when not evaluated) */
  bumps: number | null;
  /** odd/even bump alternation measure (null when not evaluated) */
  alt: number | null;
  range_log10: number | null;
  clip_frac: number | null;
  speech_frac: number | null;
  /** analysed window length (s); null before the window is analysed */
  span_s: number | null;
}

/** Consistency checks mirroring carelens.params.audio_errors. */
export function audioErrors(p: AudioParams): string[] {
  const e: string[] = [];
  const env = p.envelope, b = p.breathing, n = env.frame_len;
  if (n < 2 || (n & (n - 1)) !== 0 || env.hop < 1 || env.hop > n) e.push('envelope.frame_len must be a power of two >= 2 and 1 <= hop <= frame_len');
  if (!(0 <= env.band_lo_hz && env.band_lo_hz < env.band_hi_hz && env.band_hi_hz <= p.fs_hz / 2) || !(env.eps > 0)) e.push('need 0 <= envelope.band_lo < band_hi <= Nyquist and eps > 0');
  const envFs = p.fs_hz / Math.max(env.hop, 1);
  if (!(0 < b.band_lo_hz && b.band_lo_hz < b.band_hi_hz && b.band_hi_hz < envFs / 2)) e.push('need 0 < breathing.band_lo < band_hi < envelope Nyquist');
  if (!(0 < b.settle_s && b.settle_s <= b.window_s) || !(b.step_s > 0) || b.zero_pad_factor < 1) e.push('need 0 < settle_s <= window_s, step_s > 0, zero_pad_factor >= 1');
  for (const k of ['snr_halfwidth_hz', 'harm_halfwidth_hz', 'min_range_log10', 'min_snr', 'sub_ratio', 'smooth_s', 'min_bump_s', 'rate_tol', 'alt_min', 'width_min'] as const) if (!(b[k] > 0)) e.push(`breathing.${k} must be > 0`);
  if (!(0 < b.clip_level && b.clip_level <= 1 && 0 <= b.clip_frac_max && b.clip_frac_max < 1 && 0 <= b.speech_thr && b.speech_thr <= 1 && 0 <= b.speech_frac_max && b.speech_frac_max < 1)) e.push('clip_level/speech_thr in (0, 1], clip_frac_max/speech_frac_max in [0, 1)');
  if (!(0 < b.bump_prom_frac && b.bump_prom_frac < 1 && b.min_bumps >= 3 && b.min_alt_bumps >= 4 && b.rate_tol < 0.5)) e.push('need 0 < bump_prom_frac < 1, min_bumps >= 3, min_alt_bumps >= 4, rate_tol < 0.5');
  if (b.harm_w2 < 0 || b.harm_w3 < 0 || !(b.q_snr_full_db > 10 * Math.log10(Math.max(b.min_snr, 1e-300)))) e.push('harm weights >= 0 and q_snr_full_db above the gate in dB');
  const y = p.yamnet;
  if (y.window_samples < 1 || !(1 <= y.hop_samples && y.hop_samples <= y.window_samples)) e.push('need 1 <= yamnet.hop_samples <= window_samples');
  const c = p.cough;
  if (!(0 < c.off_thr && c.off_thr <= c.on_thr && c.on_thr <= 1) || c.min_frames < 1 || c.merge_gap_s < 0 || c.refractory_s < 0) e.push('cough: need 0 < off_thr <= on_thr <= 1, min_frames >= 1, merge_gap_s/refractory_s >= 0');
  return e;
}

export function assertAudioParams(p: AudioParams): AudioParams {
  const errs = audioErrors(p);
  if (errs.length) throw new Error('invalid audio params: ' + errs.join('; '));
  return p;
}

// ------------------------------------------------------------------ audio -> envelope

export function nFrames(nSamples: number, p: AudioParams): number {
  const e = p.envelope;
  return nSamples < e.frame_len ? 0 : 1 + Math.floor((nSamples - e.frame_len) / e.hop);
}

/** One envelope frame from `x[start .. start+frame_len)`: log10 band power and clip fraction. */
export function envelopeFrame(x: ArrayLike<number>, start: number, p: AudioParams, win: number[] = hann(p.envelope.frame_len)): { env: number; clip: number } {
  const e = p.envelope, n = e.frame_len;
  const re = new Float64Array(n), im = new Float64Array(n);
  let clipped = 0;
  for (let i = 0; i < n; i++) {
    const v = x[start + i]!;
    re[i] = v * win[i]!;
    if (Math.abs(v) >= p.breathing.clip_level) clipped++;
  }
  fft(re, im);
  const [kLo, kHi] = bandBins(n, p.fs_hz, e.band_lo_hz, e.band_hi_hz);
  let power = 0;
  for (let k = kLo; k <= kHi; k++) power += re[k]! * re[k]! + im[k]! * im[k]!;
  return { env: Math.log10(power + e.eps), clip: clipped / n };
}

export function envelope(x: ArrayLike<number>, p: AudioParams): { envelope: number[]; clip_frac: number[] } {
  const win = hann(p.envelope.frame_len);
  const env: number[] = [], clip: number[] = [];
  for (let k = 0, n = nFrames(x.length, p); k < n; k++) {
    const f = envelopeFrame(x, k * p.envelope.hop, p, win);
    env.push(f.env); clip.push(f.clip);
  }
  return { envelope: env, clip_frac: clip };
}

/** Incremental envelope: push(samples) returns the {t, env, clip} of every frame completed (t = frame end, stream seconds). */
export class EnvelopeStream {
  private buf = new Float64Array(0);
  private k = 0;
  private readonly win: number[];
  constructor(private readonly p: AudioParams) { this.win = hann(p.envelope.frame_len); }
  reset(): void { this.buf = new Float64Array(0); this.k = 0; }
  push(samples: ArrayLike<number>): { t: number; env: number; clip: number }[] {
    const e = this.p.envelope;
    const merged = new Float64Array(this.buf.length + samples.length);
    merged.set(this.buf);
    for (let i = 0; i < samples.length; i++) merged[this.buf.length + i] = samples[i]!;
    const out: { t: number; env: number; clip: number }[] = [];
    let pos = 0;
    while (merged.length - pos >= e.frame_len) {
      const f = envelopeFrame(merged, pos, this.p, this.win);
      out.push({ t: (this.k * e.hop + e.frame_len) / this.p.fs_hz, env: f.env, clip: f.clip });
      pos += e.hop; this.k++;
    }
    this.buf = merged.slice(pos);
    return out;
  }
}

// ------------------------------------------------------------------ breathing from the envelope

export interface Bump { t: number; prom: number; width: number }

/** Prominent local maxima of the boxcar-smoothed envelope (see carelens.audio.find_bumps). */
export function findBumps(y: ArrayLike<number>, fs: number, p: AudioBreathingParams): Bump[] {
  const m = Math.max(1, Math.round(p.smooth_s * fs));
  const c = new Float64Array(y.length + 1);
  for (let i = 0; i < y.length; i++) c[i + 1] = c[i]! + y[i]!;
  const n = y.length - m + 1;
  const s = new Float64Array(n);
  for (let i = 0; i < n; i++) s[i] = (c[i + m]! - c[i]!) / m;
  const minProm = p.bump_prom_frac * (percentile(Array.from(s), 95) - percentile(Array.from(s), 5));
  const out: Bump[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (!(s[i]! > s[i - 1]! && s[i]! >= s[i + 1]!)) continue;
    let loL = s[i]!, j = i - 1;
    while (j >= 0 && s[j]! <= s[i]!) { loL = Math.min(loL, s[j]!); j--; }
    let loR = s[i]!, k = i + 1;
    while (k < n && s[k]! <= s[i]!) { loR = Math.min(loR, s[k]!); k++; }
    const prom = s[i]! - Math.max(loL, loR);
    if (prom < minProm) continue;
    const level = s[i]! - 0.5 * prom;
    let l = i;
    while (l > 0 && s[l - 1]! > level) l--;
    let r = i;
    while (r < n - 1 && s[r + 1]! > level) r++;
    if (l > 0 && r < n - 1 && (r - l + 1) / fs >= p.min_bump_s) out.push({ t: i / fs, prom, width: (r - l + 1) / fs });
  }
  return out;
}

const altOf = (v: number[]): number => {
  let a = 0, na = 0, b = 0, nb = 0;
  v.forEach((x, i) => { if (i % 2 === 0) { a += x; na++; } else { b += x; nb++; } });
  return Math.abs(a / na - b / nb) / mean(v);
};

export function hasSubLine(power: Float64Array, nFft: number, fs: number, f: number, p: AudioBreathingParams): boolean {
  if (f / 2 < p.band_lo_hz + p.harm_halfwidth_hz) return false;
  const [kLo, kHi] = bandBins(nFft, fs, p.band_lo_hz, p.band_hi_hz);
  const kf = peakBin(power, nFft, fs, f, p.harm_halfwidth_hz, kLo, kHi);
  const kh = peakBin(power, nFft, fs, f / 2, p.harm_halfwidth_hz, kLo, kHi);
  return kf >= 0 && kLo < kh && kh < kHi && power[kh]! >= power[kh - 1]! && power[kh]! >= power[kh + 1]! && power[kh]! >= p.sub_ratio * power[kf]!;
}

/** Doubled-rate rule (carelens.audio docstring): the rate in Hz, or null when ambiguous. */
export function resolveRate(y: number[], power: Float64Array, nFft: number, fs: number, f: number, p: AudioBreathingParams): { rate: number | null; bumps: number; alt: number | null } {
  const bumps = findBumps(y, fs, p);
  const nb = bumps.length;
  if (nb < p.min_bumps) return { rate: null, bumps: nb, alt: null };
  const amps = bumps.map((b) => b.prom), widths = bumps.map((b) => b.width);
  const interval = (bumps[nb - 1]!.t - bumps[0]!.t) / (nb - 1);
  const fB = 1 / interval;
  let alt: number | null = null, alternating = false;
  if (nb >= p.min_alt_bumps) {
    const v = altOf(amps) >= altOf(widths) ? amps : widths;
    alt = altOf(v);
    let sumAbs = 0;
    const d: number[] = [];
    for (let i = 1; i < v.length; i++) { d.push(v[i]! - v[i - 1]!); sumAbs += Math.abs(v[i]! - v[i - 1]!); }
    const asym = sumAbs / d.length / mean(v);
    let flips = true;
    for (let i = 1; i < d.length; i++) if (!(d[i]! * d[i - 1]! < 0)) flips = false;
    alternating = alt >= p.alt_min && asym >= p.alt_min && flips;
  }
  if (Math.abs(f - fB) <= p.rate_tol * fB) {
    if (f / 2 < p.band_lo_hz * (1 - p.rate_tol)) return { rate: f, bumps: nb, alt };
    if (alternating) return { rate: f / 2, bumps: nb, alt };
    let wsum = 0;
    for (const w of widths) wsum += w;
    const single = wsum / nb / interval >= p.width_min && !hasSubLine(power, nFft, fs, f, p);
    return { rate: single ? f : null, bumps: nb, alt };
  }
  if (Math.abs(f - fB / 2) <= (p.rate_tol * fB) / 2) return { rate: alternating ? f : null, bumps: nb, alt };
  return { rate: null, bumps: nb, alt };
}

export function qualityScore(snr: number, p: AudioBreathingParams): number {
  const gateDb = 10 * Math.log10(p.min_snr);
  return clamp((10 * Math.log10(Math.max(snr, 1e-30)) - gateDb) / (p.q_snr_full_db - gateDb), 0, 1);
}

export class BreathSoundRate {
  private readonly p: AudioBreathingParams;
  private readonly envFs: number;
  private readonly windowN: number;
  private readonly settleN: number;
  private env: number[] = [];
  private clip: number[] = [];
  private speech: [number, number][] = [];
  private nextOut: number | null = null;
  private lastT: number | null = null;
  private noMic = false;

  constructor(params: AudioParams) {
    assertAudioParams(params);
    this.p = params.breathing;
    this.envFs = params.fs_hz / params.envelope.hop;
    this.windowN = Math.floor(this.p.window_s * this.envFs + 1e-6);
    this.settleN = Math.floor(this.p.settle_s * this.envFs + 1e-6);
  }

  reset(): void {
    this.env = []; this.clip = []; this.speech = []; this.nextOut = null; this.lastT = null; this.noMic = false;
  }

  pushSpeech(t: number, score: number): void { this.speech.push([t, score]); }

  /** `env` null/non-finite = no microphone signal. Returns an update once per step_s. */
  push(t: number, env: number | null, clipFrac = 0): BreathSoundUpdate | null {
    const p = this.p;
    if (this.lastT !== null && !(t > this.lastT)) return null;
    this.lastT = t;
    if (this.nextOut === null) this.nextOut = t + p.step_s;
    if (env === null || !Number.isFinite(env)) {
      this.env = []; this.clip = []; this.speech = []; this.noMic = true;
    } else {
      this.noMic = false;
      this.env.push(env); this.clip.push(clipFrac);
      if (this.env.length > this.windowN) { this.env.shift(); this.clip.shift(); }
    }
    if (t < this.nextOut) return null;
    while (this.nextOut <= t) this.nextOut += p.step_s;
    return this.emit(t);
  }

  private emit(t: number): BreathSoundUpdate {
    const p = this.p;
    const out: BreathSoundUpdate = { t, bpm: null, quality: 0, reported: false, reason: 'ok', snr: null, bumps: null, alt: null, range_log10: null, clip_frac: null, speech_frac: null, span_s: null };
    if (this.noMic) { out.reason = 'no_mic'; return out; }
    const n = this.env.length;
    if (n < this.settleN) { out.reason = 'settling'; return out; }
    out.span_s = n / this.envFs; // analysed window length (s)
    let cs = 0;
    for (const c of this.clip) cs += c;
    out.clip_frac = cs / n;
    out.range_log10 = percentile(this.env, 95) - percentile(this.env, 5);
    while (this.speech.length && this.speech[0]![0] <= t - p.window_s) this.speech.shift();
    if (this.speech.length) out.speech_frac = this.speech.filter((s) => s[1] >= p.speech_thr).length / this.speech.length;
    if (out.clip_frac > p.clip_frac_max) out.reason = 'clipping';
    else if (out.range_log10 < p.min_range_log10) out.reason = 'too_quiet';
    else if (out.speech_frac !== null && out.speech_frac >= p.speech_frac_max) out.reason = 'talking';
    else this.analyse(out);
    return out;
  }

  private analyse(out: BreathSoundUpdate): void {
    const p = this.p;
    const y = detrendLinear(this.env);
    const nFft = fftSize(y.length, p.zero_pad_factor);
    const power = powerSpectrum(y, nFft);
    const f = spectralPeak(power, nFft, this.envFs, p);
    let snr = spectralSnr(power, nFft, this.envFs, f, p);
    if (!(Number.isFinite(f) && Number.isFinite(snr)) || snr < p.min_snr) {
      out.reason = 'low_signal';
      out.snr = Number.isFinite(snr) ? snr : null;
      return;
    }
    const r = resolveRate(y, power, nFft, this.envFs, f, p);
    out.bumps = r.bumps; out.alt = r.alt;
    if (r.rate === null) { out.reason = 'ambiguous'; out.snr = snr; return; }
    snr = spectralSnr(power, nFft, this.envFs, r.rate, p);
    out.snr = snr; out.quality = qualityScore(snr, p); out.bpm = 60 * r.rate;
    out.reason = 'ok'; out.reported = true;
  }
}

/** Fixture runner: mirrors carelens.audio.run_breath_env. */
export function runBreathEnv(inp: { fs_env: number; env: Array<number | null>; clip?: number[]; speech?: [number, number][] }, params: AudioParams): { updates: BreathSoundUpdate[] } {
  const est = new BreathSoundRate(params);
  const speech = [...(inp.speech ?? [])].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let j = 0;
  const updates: BreathSoundUpdate[] = [];
  inp.env.forEach((v, i) => {
    const t = i / inp.fs_env;
    while (j < speech.length && speech[j]![0] <= t) { est.pushSpeech(speech[j]![0], speech[j]![1]); j++; }
    const u = est.push(t, v, inp.clip ? inp.clip[i]! : 0);
    if (u) updates.push(u);
  });
  return { updates };
}

// ------------------------------------------------------------------ label events

export interface LabelEvent { t_start: number; t_end: number; peak_score: number }

export function detectLabelEvents(frames: ReadonlyArray<readonly [number, number]>, p: LabelEventParams): { events: LabelEvent[]; count: number } {
  const segs: [number, number, number, number][] = [];
  let cur: [number, number, number, number] | null = null;
  for (const [t, s] of frames) {
    if (cur === null) { if (s >= p.on_thr) cur = [t, t, s, 1]; }
    else if (s >= p.off_thr) { cur[1] = t; cur[2] = Math.max(cur[2], s); cur[3]++; }
    else { segs.push(cur); cur = null; }
  }
  if (cur !== null) segs.push(cur);
  const events: [number, number, number][] = [];
  for (const [t0, t1, peak, nf] of segs) {
    if (nf < p.min_frames) continue;
    const prev = events[events.length - 1];
    if (prev && (t0 - prev[1] < p.merge_gap_s - 1e-9 || t0 - prev[0] < p.refractory_s - 1e-9)) { prev[1] = t1; prev[2] = Math.max(prev[2], peak); }
    else events.push([t0, t1, peak]);
  }
  return { events: events.map(([a, b, c]) => ({ t_start: a, t_end: b, peak_score: c })), count: events.length };
}

export const MAX_HOURS = 72; // per-hour buckets kept (session hours); later events fall into the last bucket

/** The time one YAMNet frame stands for (its native hop): an event's duration is `t_end - t_start + frameSeconds`. */
export const frameSeconds = (p: AudioParams): number => p.yamnet.hop_samples / p.fs_hz;

const hourOf = (tStart: number): number => Math.min(MAX_HOURS - 1, Math.max(0, Math.floor(tStart / 3600)));
function bump(hours: number[], hoursS: number[], h: number, d: number): void {
  while (hours.length < h + 1) hours.push(0);
  while (hoursS.length < h + 1) hoursS.push(0);
  hours[h]!++;
  hoursS[h]! += d;
}

/** `{count, total_s, per_hour}` (carelens.audio.summarize_events): total event time and the dense per-SESSION-hour counts. */
export function summarizeEvents(events: ReadonlyArray<LabelEvent>, frameS: number): { count: number; total_s: number; per_hour: number[]; per_hour_s: number[] } {
  let total = 0;
  const per: number[] = [], perS: number[] = [];
  for (const e of events) { const d = e.t_end - e.t_start + frameS; total += d; bump(per, perS, hourOf(e.t_start), d); }
  return { count: events.length, total_s: total, per_hour: per, per_hour_s: perS };
}

export interface LabelView { count: number; last: LabelEvent | null; ticks: number[]; total_s: number; per_hour: number[]; per_hour_s: number[] }

/**
 * Incremental `detectLabelEvents`: constant memory (open segment, last event, running totals, a capped list of event starts,
 * per-hour counts). `view()` equals the batch result on the frames pushed so far (a test guards it); `count` only grows.
 */
export class LabelEventTracker {
  private cur: [number, number, number, number] | null = null;
  private last: [number, number, number] | null = null; // last finalized event [t_start, t_end, peak] (a merge may still extend it)
  private count = 0;
  private doneS = 0; // duration of the events before `last`
  private hours: number[] = []; // per-hour counts of the events before `last`
  private hoursS: number[] = []; // per-hour seconds of the events before `last`
  private starts: number[] = [];
  constructor(private readonly p: LabelEventParams, private readonly maxTicks = 500, private readonly frameS = 0.48) {}

  reset(): void { this.cur = null; this.last = null; this.count = 0; this.doneS = 0; this.hours = []; this.hoursS = []; this.starts = []; }

  private absorb(seg: [number, number, number, number], last: [number, number, number] | null, count: number): { last: [number, number, number] | null; count: number; added: boolean } {
    const p = this.p;
    if (seg[3] < p.min_frames) return { last, count, added: false };
    if (last && (seg[0] - last[1] < p.merge_gap_s - 1e-9 || seg[0] - last[0] < p.refractory_s - 1e-9)) {
      return { last: [last[0], seg[1], Math.max(last[2], seg[2])], count, added: false };
    }
    return { last: [seg[0], seg[1], seg[2]], count: count + 1, added: true };
  }

  push(t: number, s: number): void {
    const p = this.p;
    if (this.cur === null) { if (s >= p.on_thr) this.cur = [t, t, s, 1]; return; }
    if (s >= p.off_thr) { this.cur[1] = t; this.cur[2] = Math.max(this.cur[2], s); this.cur[3]++; return; }
    const prev = this.last;
    const r = this.absorb(this.cur, this.last, this.count);
    if (r.added && prev) { const d = prev[1] - prev[0] + this.frameS; this.doneS += d; bump(this.hours, this.hoursS, hourOf(prev[0]), d); } // `prev` can no longer change
    this.last = r.last; this.count = r.count; this.cur = null;
    if (r.added) { this.starts.push(r.last![0]); if (this.starts.length > this.maxTicks) this.starts.shift(); }
  }

  /** Events as the batch detector would report them for the frames so far (an open segment counts). */
  view(): LabelView {
    let last = this.last, count = this.count, doneS = this.doneS;
    const prev = this.last;
    const ticks = this.starts.slice();
    const hours = this.hours.slice(), hoursS = this.hoursS.slice();
    if (this.cur) {
      const r = this.absorb(this.cur, last, count);
      last = r.last; count = r.count;
      if (r.added) {
        ticks.push(last![0]);
        if (prev) { const d = prev[1] - prev[0] + this.frameS; doneS += d; bump(hours, hoursS, hourOf(prev[0]), d); }
      }
    }
    let total = doneS;
    if (last) { const d = last[1] - last[0] + this.frameS; total += d; bump(hours, hoursS, hourOf(last[0]), d); }
    return { count, last: last ? { t_start: last[0], t_end: last[1], peak_score: last[2] } : null, ticks, total_s: total, per_hour: hours, per_hour_s: hoursS };
  }
}

/** Label -> detector parameters: Cough (W4) plus the W10 labels of `audio_events` (carelens.audio.label_sections). */
export function labelSections(params: { audio: AudioParams; audio_events: AudioEventsParams }): Record<string, LabelEventParams> {
  const e = params.audio_events;
  const out: Record<string, LabelEventParams> = { Cough: params.audio.cough };
  for (const s of [e.snoring, e.screaming, e.crying, e.glass, e.thump, e.speech]) out[s.label] = s;
  return out;
}

export function runLabelEvents(inp: { label: string; frames: [number, number][] }, params: { audio: AudioParams; audio_events: AudioEventsParams }): { events: LabelEvent[]; count: number; total_s: number; per_hour: number[]; per_hour_s: number[] } {
  const res = detectLabelEvents(inp.frames, labelSections(params)[inp.label]!);
  const sm = summarizeEvents(res.events, frameSeconds(params.audio));
  return { ...res, total_s: sm.total_s, per_hour: sm.per_hour, per_hour_s: sm.per_hour_s };
}

/** Fixture runner for audio_envelope: int16 PCM (base64) -> envelope + clip fraction. */
export function runAudioEnvelope(inp: { pcm_b64: string }, params: AudioParams): { envelope: number[]; clip_frac: number[] } {
  const bin = atob(inp.pcm_b64);
  const x = new Float64Array(bin.length / 2);
  for (let i = 0; i < x.length; i++) {
    let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    if (v >= 32768) v -= 65536;
    x[i] = v / 32768;
  }
  return envelope(x, params);
}

// ------------------------------------------------------------------ resampling (browser capture -> 16 kHz)

/** Streaming windowed-sinc resampler (Hann, cutoff 0.45 x min(in, out) rate). Mono Float32/64 in, Float32 out. */
export class Resampler {
  private buf: number[] = [];
  private pos = 0; // next output position in input samples, relative to buf[0]
  private readonly step: number;
  private readonly half: number;
  private readonly cut: number;
  constructor(private readonly inRate: number, private readonly outRate: number, taps = 12) {
    this.step = inRate / outRate;
    this.cut = (0.45 * Math.min(inRate, outRate)) / inRate; // normalised (cycles/input sample)
    this.half = Math.ceil(taps * Math.max(1, this.step));
  }
  push(x: ArrayLike<number>): Float32Array {
    for (let i = 0; i < x.length; i++) this.buf.push(x[i]!);
    const out: number[] = [];
    while (this.pos + this.half < this.buf.length) {
      const c = this.pos, i0 = Math.ceil(c - this.half), i1 = Math.floor(c + this.half);
      let acc = 0, wsum = 0;
      for (let i = Math.max(i0, 0); i <= i1; i++) {
        const d = i - c;
        const w = 0.5 + 0.5 * Math.cos((Math.PI * d) / (this.half + 1));
        const s = d === 0 ? 2 * this.cut : Math.sin(2 * Math.PI * this.cut * d) / (Math.PI * d);
        acc += this.buf[i]! * s * w; wsum += s * w;
      }
      out.push(wsum !== 0 ? acc / wsum : 0);
      this.pos += this.step;
    }
    const drop = Math.max(0, Math.floor(this.pos - this.half) - 1);
    if (drop > 0) { this.buf.splice(0, drop); this.pos -= drop; }
    return Float32Array.from(out);
  }
}
