// W10 (Tier 2, Experimental): DOM-free port of carelens/sleep_sounds.py (read its docstring for the design and limits):
// possible pauses in breathing sounds (breath bumps from the 50 Hz envelope, regularity + speech/spike gates), and the
// sound cue for a possible fall. Audio is processed in memory only. Wellbeing trend only. NOT a medical device and NOT a sleep test.

import { YAMNET_WINDOW_S, type AudioEventsParams, type LabelEventParams } from './audio';

export interface SleepBumpParams {
  smooth_s: number; floor_win_s: number; rise_on_log10: number; rise_off_frac: number; min_bump_s: number; max_bump_s: number; merge_s: number; silence_log10: number;
  spike_ratio: number; typ_bumps: number;
}
export interface SleepSoundsParams {
  pause_min_s: number; pause_max_s: number; pre_regular_s: number; pre_min_bumps: number; pre_cv_max: number;
  rate_min_bpm: number; rate_max_bpm: number; min_valid_regular_s: number; speech_thr: number; overlap_margin_s: number;
  bump: SleepBumpParams;
}

export const MAX_KEEP = 500; // pauses kept in the view (the count itself is unbounded)
export const MAX_RECENT = 600; // bumps kept for the regularity window

/** Consistency checks mirroring carelens.params.audio_events_errors. */
export function audioEventsErrors(p: AudioEventsParams): string[] {
  const e: string[] = [];
  const keys = ['snoring', 'screaming', 'crying', 'glass', 'thump', 'speech'] as const;
  const secs: LabelEventParams[] = keys.map((k) => p[k]);
  keys.forEach((k, i) => {
    const s = secs[i]!;
    if (!(0 < s.off_thr && s.off_thr <= s.on_thr && s.on_thr <= 1) || s.min_frames < 1 || s.merge_gap_s < 0 || s.refractory_s < 0) e.push(`${k}: need 0 < off_thr <= on_thr <= 1, min_frames >= 1, merge_gap_s/refractory_s >= 0`);
  });
  const labels = secs.map((s) => s.label);
  if (labels.some((l) => !l) || new Set(labels).size !== labels.length) e.push('section labels must be non-empty and distinct');
  if (!(p.corroborate_s > 0)) e.push('corroborate_s must be > 0');
  if (!p.cue_labels.length || p.cue_labels.some((l) => !labels.includes(l))) e.push('cue_labels must be a non-empty subset of the section labels');
  return e;
}

/** Consistency checks mirroring carelens.params.sleep_sounds_errors. */
export function sleepSoundsErrors(p: SleepSoundsParams): string[] {
  const e: string[] = [];
  const b = p.bump;
  if (!(0 < p.pause_min_s && p.pause_min_s < p.pause_max_s)) e.push('need 0 < pause_min_s < pause_max_s');
  if (!(p.pre_regular_s > 0 && p.pre_min_bumps >= 3 && 0 < p.pre_cv_max && p.pre_cv_max < 1 && p.min_valid_regular_s > 0)) e.push('need pre_regular_s > 0, pre_min_bumps >= 3, 0 < pre_cv_max < 1, min_valid_regular_s > 0');
  if (!(0 < p.rate_min_bpm && p.rate_min_bpm < p.rate_max_bpm)) e.push('need 0 < rate_min_bpm < rate_max_bpm');
  if (!(0 <= p.speech_thr && p.speech_thr <= 1 && p.overlap_margin_s >= 0)) e.push('need 0 <= speech_thr <= 1 and overlap_margin_s >= 0');
  if (!(b.smooth_s > 0 && b.rise_on_log10 > 0 && 0 < b.rise_off_frac && b.rise_off_frac < 1 && 0 < b.min_bump_s && b.min_bump_s < b.max_bump_s && b.spike_ratio > 1 && b.typ_bumps >= 1)) e.push('bump: need smooth_s > 0, rise_on_log10 > 0, 0 < rise_off_frac < 1, 0 < min_bump_s < max_bump_s, spike_ratio > 1, typ_bumps >= 1');
  if (!(0 <= b.merge_s && b.merge_s <= 60 / Math.max(p.rate_max_bpm, 1e-9))) e.push('bump.merge_s must be in [0, 60 / rate_max_bpm] (in-band breaths are never merged)');
  if (p.rate_min_bpm > 0 && !(b.floor_win_s >= 60 / p.rate_min_bpm)) e.push('bump.floor_win_s must cover at least one slowest breath (>= 60 / rate_min_bpm)');
  return e;
}

export function assertSleepParams(audioEvents: AudioEventsParams, sleep: SleepSoundsParams): void {
  const errs = [...audioEventsErrors(audioEvents), ...sleepSoundsErrors(sleep)];
  if (errs.length) throw new Error('invalid sleep-sound params: ' + errs.join('; '));
}

const roundHalfUp = (x: number): number => Math.floor(x + 0.5 + 1e-9);
function median(v: ArrayLike<number>): number {
  const s = Array.from(v).sort((a, b) => a - b), n = s.length;
  return n % 2 ? s[(n - 1) / 2]! : 0.5 * (s[n / 2 - 1]! + s[n / 2]!);
}

export type BumpRecord = { kind: 'bump'; t: number; rise: number } | { kind: 'spike'; t0: number; t1: number };

/** Envelope frames -> breath bumps and loud non-breath "spikes" (see the Python docstring). */
export class BumpStream {
  private readonly b: SleepBumpParams;
  private readonly m: number;
  private readonly floorN: number;
  private ring: number[] = [];
  private monoK: number[] = [];
  private monoV: number[] = [];
  private k = 0;
  private seg: [number, number, number, number] | null = null; // [t0, peak, t_peak, peak_rise]
  private rises: number[] = [];

  constructor(p: SleepSoundsParams, private readonly fs: number) {
    this.b = p.bump;
    this.m = Math.max(1, roundHalfUp(this.b.smooth_s * fs));
    this.floorN = Math.max(1, Math.floor(this.b.floor_win_s * fs + 1e-6));
  }

  reset(): void { this.ring = []; this.monoK = []; this.monoV = []; this.k = 0; this.seg = null; this.rises = []; }

  push(t: number, env: number): BumpRecord[] {
    const b = this.b;
    this.ring.push(env);
    if (this.ring.length > this.m) this.ring.shift();
    if (this.ring.length < this.m) return [];
    let acc = 0;
    for (const v of this.ring) acc += v;
    const s = acc / this.m;
    const ts = t - ((this.m - 1) / 2) / this.fs;
    const k = this.k++;
    while (this.monoV.length && this.monoV[this.monoV.length - 1]! >= s) { this.monoV.pop(); this.monoK.pop(); }
    this.monoK.push(k); this.monoV.push(s);
    while (this.monoK[0]! <= k - this.floorN) { this.monoK.shift(); this.monoV.shift(); }
    const rise = s - this.monoV[0]!;
    const out: BumpRecord[] = [];
    const seg = this.seg;
    if (seg === null) {
      if (rise >= b.rise_on_log10) this.seg = [ts, s, ts, rise];
      return out;
    }
    if (s > seg[1]) { seg[1] = s; seg[2] = ts; seg[3] = rise; }
    if (rise >= b.rise_on_log10 * b.rise_off_frac) return out;
    this.seg = null;
    const dur = ts - seg[0];
    if (dur > b.max_bump_s) out.push({ kind: 'spike', t0: seg[0], t1: ts });
    else if (dur >= b.min_bump_s) {
      if (this.rises.length >= b.typ_bumps && seg[3] > b.spike_ratio * median(this.rises)) out.push({ kind: 'spike', t0: seg[0], t1: ts });
      else {
        this.rises.push(seg[3]);
        if (this.rises.length > 16) this.rises.shift();
        out.push({ kind: 'bump', t: seg[2], rise: seg[3] });
      }
    }
    return out;
  }

  /** buffer sizes (for the bounded-memory test) */
  get sizes(): { mono: number; floorN: number; rises: number } { return { mono: this.monoK.length, floorN: this.floorN, rises: this.rises.length }; }
}

export interface PauseEvent { t_start: number; t_end: number; duration_s: number }
export interface PauseView { count: number; pauses: PauseEvent[]; regular_s: number; rate_per_h: number | null; no_sound_gaps: number }

/** Possible pauses in breathing sounds, incrementally and in bounded memory. */
export class PauseTracker {
  private readonly stream: BumpStream;
  count = 0;
  regularS = 0;
  noSoundGaps = 0;
  private pauses: [number, number, number][] = [];
  private speech: [number, number][] = [];
  private spikes: [number, number][] = [];
  private recent: number[] = [];
  private lastRise: number | null = null;
  private now = 0;

  constructor(private readonly p: SleepSoundsParams, fsEnv: number) { this.stream = new BumpStream(p, fsEnv); }

  reset(): void {
    this.stream.reset();
    this.count = 0; this.regularS = 0; this.noSoundGaps = 0;
    this.pauses = []; this.speech = []; this.spikes = []; this.recent = []; this.lastRise = null; this.now = 0;
  }

  /** The signal broke (microphone lost): forget the running bump / regularity state, keep the totals. */
  breakStream(): void { this.stream.reset(); this.recent = []; this.lastRise = null; this.speech = []; this.spikes = []; }

  private trim(t: number): void {
    this.now = Math.max(this.now, t);
    const horizon = this.now - this.p.pause_max_s - 10;
    while (this.speech.length && this.speech[0]![0] < horizon) this.speech.shift();
    while (this.spikes.length && this.spikes[0]![1] < horizon) this.spikes.shift();
  }

  pushSpeech(t: number, score: number): void { this.speech.push([t, score]); this.trim(t); }

  /** `env` null / non-finite = no signal. */
  pushFrame(t: number, env: number | null): void {
    if (env === null || !Number.isFinite(env) || env < this.p.bump.silence_log10) { this.breakStream(); return; } // no signal, or digital silence
    this.trim(t);
    for (const r of this.stream.push(t, env)) {
      if (r.kind === 'bump') this.pushBump(r.t, r.rise); else this.pushSpike(r.t0, r.t1);
    }
  }

  pushSpike(t0: number, t1: number): void { this.spikes.push([t0, t1]); this.trim(t1); }

  private overlaps(a: number, b: number): boolean {
    const p = this.p, lo = a - p.overlap_margin_s, hi = b + p.overlap_margin_s;
    for (const [t, s] of this.speech) if (s >= p.speech_thr && t > lo && t - YAMNET_WINDOW_S < hi) return true;
    for (const [t0, t1] of this.spikes) if (t1 > lo && t0 < hi) return true;
    return false;
  }

  private regular(tEnd: number): boolean {
    const p = this.p;
    const w = this.recent.filter((x) => x >= tEnd - p.pre_regular_s);
    if (w.length < p.pre_min_bumps) return false;
    const iv: number[] = [];
    for (let i = 0; i + 1 < w.length; i++) iv.push(w[i + 1]! - w[i]!);
    let sum = 0;
    for (const x of iv) sum += x;
    const mean = sum / iv.length;
    if (!(mean > 0)) return false;
    let ss = 0;
    for (const x of iv) ss += (x - mean) ** 2;
    const sd = Math.sqrt(ss / iv.length), rate = 60 / mean;
    const covers = w[0]! <= tEnd - p.pre_regular_s + 1.5 * mean;
    return covers && sd / mean <= p.pre_cv_max && p.rate_min_bpm <= rate && rate <= p.rate_max_bpm;
  }

  /** A breath bump at `t` (`rise` = its height, to pick the louder of two bumps of one breath). */
  pushBump(t: number, rise: number | null = null): void {
    const p = this.p;
    this.trim(t);
    if (this.recent.length) {
      const tp = this.recent[this.recent.length - 1]!;
      const iv = t - tp;
      if (iv > 0 && iv < p.bump.merge_s) { // the second bump of the same breath: keep the louder one's time
        if (rise !== null && (this.lastRise === null || rise > this.lastRise)) { this.recent[this.recent.length - 1] = t; this.lastRise = rise; }
        return;
      }
      if (iv >= p.pause_min_s) {
        if (iv <= p.pause_max_s) {
          if (this.regular(tp) && !this.overlaps(tp, t)) {
            this.count++; this.regularS += iv;
            this.pauses.push([tp, t, iv]);
            if (this.pauses.length > MAX_KEEP) this.pauses.shift();
          }
        } else this.noSoundGaps++;
      } else if (iv > 0) {
        this.addRecent(t);
        this.lastRise = rise;
        if (this.regular(t) && !this.overlaps(tp, t)) this.regularS += iv;
        this.trimRecent(t);
        return;
      } else return; // non-monotonic: ignore
    }
    this.addRecent(t);
    this.lastRise = rise;
    this.trimRecent(t);
  }

  private addRecent(t: number): void { this.recent.push(t); if (this.recent.length > MAX_RECENT) this.recent.shift(); }
  private trimRecent(t: number): void {
    const horizon = t - this.p.pre_regular_s - this.p.pause_max_s;
    while (this.recent.length && this.recent[0]! < horizon) this.recent.shift();
  }

  view(): PauseView {
    const p = this.p;
    return {
      count: this.count,
      pauses: this.pauses.map(([a, b, d]) => ({ t_start: a, t_end: b, duration_s: d })),
      regular_s: this.regularS,
      rate_per_h: this.regularS >= p.min_valid_regular_s ? this.count / (this.regularS / 3600) : null,
      no_sound_gaps: this.noSoundGaps,
    };
  }

  /** buffer sizes (for the bounded-memory test) */
  get sizes(): { recent: number; speech: number; spikes: number; pauses: number } { return { recent: this.recent.length, speech: this.speech.length, spikes: this.spikes.length, pauses: this.pauses.length }; }
}

// ------------------------------------------------------------------ sound cue for a possible fall

export interface SoundEvent { label: string; t_start: number; t_end: number }

/** Pure: is there a cue-label sound within `corroborate_s` of the fall's [t_onset, t_confirmed]? (carelens.sleep_sounds.sound_cue) */
export function soundCue(fall: { t_onset: number; t_confirmed: number }, sounds: ReadonlyArray<SoundEvent>, p: Pick<AudioEventsParams, 'cue_labels' | 'corroborate_s'>): boolean {
  const lo = fall.t_onset - p.corroborate_s, hi = fall.t_confirmed + p.corroborate_s;
  return sounds.some((s) => p.cue_labels.includes(s.label) && s.t_end >= lo && s.t_start <= hi);
}

// ------------------------------------------------------------------ fixture runners (parity)

export function runSleepBumps(inp: { bumps: number[]; spikes?: [number, number][]; speech?: [number, number][] }, params: { audio: { fs_hz: number; envelope: { hop: number } }; sleep_sounds: SleepSoundsParams }): PauseView {
  const items: [number, number, number[]][] = [
    ...(inp.speech ?? []).map((v): [number, number, number[]] => [v[0], 0, v]),
    ...(inp.spikes ?? []).map((v): [number, number, number[]] => [v[1], 1, v]),
    ...inp.bumps.map((t): [number, number, number[]] => [t, 2, [t]]),
  ];
  items.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const tr = new PauseTracker(params.sleep_sounds, params.audio.fs_hz / params.audio.envelope.hop);
  for (const [, kind, v] of items) {
    if (kind === 0) tr.pushSpeech(v[0]!, v[1]!); else if (kind === 1) tr.pushSpike(v[0]!, v[1]!); else tr.pushBump(v[0]!);
  }
  return tr.view();
}

export function runSleepEnv(inp: { fs_env: number; env: number[]; speech?: [number, number][] }, params: { sleep_sounds: SleepSoundsParams }): PauseView {
  const tr = new PauseTracker(params.sleep_sounds, inp.fs_env);
  const speech = [...(inp.speech ?? [])].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let j = 0;
  inp.env.forEach((v, i) => {
    const t = i / inp.fs_env;
    while (j < speech.length && speech[j]![0] <= t) { tr.pushSpeech(speech[j]![0], speech[j]![1]); j++; }
    tr.pushFrame(t, v);
  });
  return tr.view();
}

export function runSoundCue(inp: { falls: { t_onset: number; t_confirmed: number }[]; sounds: SoundEvent[] }, params: { audio_events: AudioEventsParams }): { cues: boolean[] } {
  return { cues: inp.falls.map((f) => soundCue(f, inp.sounds, params.audio_events)) };
}
