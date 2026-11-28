// Guided mobility tests from pose landmarks (W7, Experimental): 30 s chair stand, five-times sit-to-stand, Timed Up and Go,
// walking speed. DOM-free TypeScript port of carelens/mobility.py: read its docstring for the design and conventions.
// Operation for operation; thresholds from shared/params.json -> mobility. Not a medical device: a wellbeing trend to compare
// with your own earlier results, never a norm, a cut-off, a pass/fail or a "fall risk" score.

import type { Landmarks } from './features';
import { expandLandmarks } from './pipeline';
import { PresenceTracker, type TrackingParams } from './tracking';

export interface MobilityParams {
  min_visibility: number; leg_vis_margin: number; min_quality: number;
  sit_deg: number; rise_deg: number; stand_deg: number; stand_hip_frac: number; stand_dwell_s: number; sit_dwell_s: number;
  chair_stand_s: number; sts_reps: number; sts_early_s: number; sts_max_s: number; tug_max_s: number; walk_window_s: number;
  tug: { vel_win_s: number; turn_speed_frac: number; min_excursion_bodies: number; min_return_frac: number };
  walk: {
    height_default_m: number; height_min_m: number; height_max_m: number; nose_ankle_frac: number; min_speed_bodies_s: number;
    vel_win_s: number; gap_s: number; height_tol_frac: number; steady_frac: number; min_dist_m: number; min_dur_s: number;
    min_samples: number; cadence_hyst_frac: number; cadence_min_crossings: number;
  };
}

/** Mirrors carelens.params.mobility_errors. */
export function mobilityErrors(p: MobilityParams): string[] {
  const e: string[] = [];
  if (!(0 < p.sit_deg && p.sit_deg < p.rise_deg && p.rise_deg < p.stand_deg && p.stand_deg <= 180)) e.push('need 0 < sit_deg < rise_deg < stand_deg <= 180');
  for (const k of ['min_visibility', 'min_quality', 'stand_hip_frac'] as const) if (!(p[k] > 0 && p[k] <= 1)) e.push(`${k} must be in (0, 1]`);
  if (!(p.leg_vis_margin >= 0 && p.leg_vis_margin < 1)) e.push('leg_vis_margin must be in [0, 1)');
  for (const k of ['stand_dwell_s', 'sit_dwell_s', 'chair_stand_s', 'sts_early_s', 'sts_max_s', 'tug_max_s', 'walk_window_s'] as const) if (!(p[k] > 0)) e.push(`${k} must be > 0`);
  if (p.sts_reps < 2 || p.sts_reps > 50) e.push('sts_reps must be in [2, 50]');
  const t = p.tug, w = p.walk;
  if (!(t.vel_win_s > 0 && t.turn_speed_frac > 0 && t.turn_speed_frac < 1 && t.min_excursion_bodies > 0 && t.min_return_frac > 0 && t.min_return_frac <= 1)) {
    e.push('tug: need vel_win_s > 0, 0 < turn_speed_frac < 1, min_excursion_bodies > 0, 0 < min_return_frac <= 1');
  }
  if (!(0 < w.height_min_m && w.height_min_m <= w.height_default_m && w.height_default_m <= w.height_max_m)) e.push('walk: need 0 < height_min_m <= height_default_m <= height_max_m');
  for (const k of ['nose_ankle_frac', 'height_tol_frac', 'steady_frac'] as const) if (!(w[k] > 0 && w[k] < 1)) e.push(`walk.${k} must be in (0, 1)`);
  for (const k of ['min_speed_bodies_s', 'vel_win_s', 'gap_s', 'min_dist_m', 'min_dur_s', 'cadence_hyst_frac'] as const) if (!(w[k] > 0)) e.push(`walk.${k} must be > 0`);
  if (w.min_samples < 3 || w.cadence_min_crossings < 3) e.push('walk: min_samples and cadence_min_crossings must be >= 3');
  return e;
}

export function assertMobilityParams(p: MobilityParams): MobilityParams {
  const errs = mobilityErrors(p);
  if (errs.length) throw new Error('invalid mobility params: ' + errs.join('; '));
  return p;
}

// MediaPipe Pose indices (carelens/pose.py)
const NOSE = 0, L_HIP = 23, R_HIP = 24, L_KNEE = 25, R_KNEE = 26, L_ANKLE = 27, R_ANKLE = 28;
export const MOBILITY_INDICES = [0, 11, 12, 23, 24, 25, 26, 27, 28];
export const TESTS = ['chair_stand_30s', 'sit_to_stand_5x', 'tug', 'walking_speed'] as const;
export type MobilityTest = (typeof TESTS)[number];
export const PHASES = ['rise', 'walk_out', 'turn_1', 'walk_back', 'turn_2', 'sit'] as const;
export type Phase = (typeof PHASES)[number];

export interface Sample {
  t: number;
  /** a measurement with at least one visible leg (never true for held frames) */
  ok: boolean;
  /** tracker repeated the last pose (a dropout): not a measurement */
  held: boolean;
  kneeDeg: number; hipH: number; x: number; height: number | null; ankleDx: number | null;
}
export const blankSample = (t: number, held = false): Sample => ({ t, ok: false, held, kneeDeg: 0, hipH: 0, x: 0, height: null, ankleDx: null });

function leg(lm: Landmarks, hip: number, knee: number, ank: number, aspect: number): [number, number, number] | null {
  const vis = Math.min(lm[hip]![2]!, lm[knee]![2]!, lm[ank]![2]!);
  const ax = lm[hip]![0]! * aspect - lm[knee]![0]! * aspect, ay = lm[hip]![1]! - lm[knee]![1]!;
  const bx = lm[ank]![0]! * aspect - lm[knee]![0]! * aspect, by = lm[ank]![1]! - lm[knee]![1]!;
  const na = Math.hypot(ax, ay), nb = Math.hypot(bx, by);
  if (na < 1e-6 || nb < 1e-6) return null;
  const c = Math.max(-1, Math.min(1, (ax * bx + ay * by) / (na * nb)));
  return [vis, (Math.acos(c) * 180) / Math.PI, (lm[ank]![1]! - lm[hip]![1]!) / (na + nb)];
}

/** One frame's signals from 33 normalised landmark rows [x, y, visibility] (null = no person). `aspect` = frame width / height
 *  (1 for landmarks that are already in frame-height units, e.g. BodyState.landmarks). */
export function sampleFromLandmarks(t: number, lm: Landmarks | null, aspect: number, mp: MobilityParams): Sample {
  if (lm === null) return blankSample(t);
  const legs: [number, number, number][] = [];
  for (const [h, k, a] of [[L_HIP, L_KNEE, L_ANKLE], [R_HIP, R_KNEE, R_ANKLE]] as const) {
    const g = leg(lm, h, k, a, aspect);
    if (g !== null && g[0] >= mp.min_visibility) legs.push(g);
  }
  if (!legs.length) return blankSample(t);
  let use = legs;
  if (legs.length === 2 && Math.abs(legs[0]![0] - legs[1]![0]) > mp.leg_vis_margin) use = [legs[0]![0] > legs[1]![0] ? legs[0]! : legs[1]!];
  const kneeDeg = use.reduce((s, g) => s + g[1], 0) / use.length;
  const hipH = use.reduce((s, g) => s + g[2], 0) / use.length;
  const x = ((lm[L_HIP]![0]! + lm[R_HIP]![0]!) / 2) * aspect;
  const ank = [L_ANKLE, R_ANKLE].map((k) => lm[k]!).filter((r) => r[2]! >= mp.min_visibility);
  let height: number | null = null;
  if (ank.length && lm[NOSE]![2]! >= mp.min_visibility) height = ank.reduce((s, r) => s + r[1]!, 0) / ank.length - lm[NOSE]![1]!;
  let ankleDx: number | null = null;
  if (ank.length === 2) ankleDx = (lm[L_ANKLE]![0]! - lm[R_ANKLE]![0]!) * aspect;
  return { t, ok: true, held: false, kneeDeg, hipH, x, height, ankleDx };
}

/** landmarks -> PresenceTracker -> Samples (dropouts bridged by the tracker come out as `held`, not ok). Incremental, for live use. */
export class SampleTracker {
  private readonly tracker: PresenceTracker;
  constructor(tracking: TrackingParams, private readonly mp: MobilityParams) { this.tracker = new PresenceTracker(tracking); }
  step(t: number, lm: Landmarks | null, aspect: number): Sample {
    const [o, held] = this.tracker.update(t, lm, aspect);
    if (o === null || held) return blankSample(t, held);
    return sampleFromLandmarks(t, o, aspect, this.mp);
  }
}

export function trackSamples(ts: number[], frames: (Landmarks | null)[], aspect: number, tracking: TrackingParams, mp: MobilityParams): Sample[] {
  const st = new SampleTracker(tracking, mp);
  return frames.map((lm, i) => st.step(ts[i]!, lm, aspect));
}

// ------------------------------------------------------------------ helpers

export const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m]! : 0.5 * (s[m - 1]! + s[m]!);
};
/** Linear-interpolated percentile (numpy default), q in [0, 1]. */
const pct = (v: number[], q: number): number => {
  const s = [...v].sort((a, b) => a - b);
  const pos = q * (s.length - 1), lo = Math.floor(pos), hi = Math.min(lo + 1, s.length - 1);
  return s[lo]! + (s[hi]! - s[lo]!) * (pos - lo);
};

export function smoothedVelocity(ts: number[], xs: number[], winS: number): number[] {
  const n = ts.length, out: number[] = [];
  for (let i = 0; i < n; i++) {
    let lo = i;
    while (lo > 0 && ts[lo - 1]! >= ts[i]! - winS / 2) lo--;
    let hi = i;
    while (hi < n - 1 && ts[hi + 1]! <= ts[i]! + winS / 2) hi++;
    out.push(hi > lo && ts[hi]! > ts[lo]! ? (xs[hi]! - xs[lo]!) / (ts[hi]! - ts[lo]!) : 0);
  }
  return out;
}

const quality = (win: Sample[]): number => (win.length ? win.filter((s) => s.ok).length / win.length : 0);

export type MobilityReason =
  | 'ok' | 'out_of_view' | 'not_seated_at_start' | 'no_rise' | 'no_turn' | 'not_seated_at_end' | 'incomplete'
  | 'not_side_on' | 'no_walk' | 'too_short' | 'bad_height';

// ------------------------------------------------------------------ sit / stand state machine

export class RepMachine {
  state: 'init' | 'seated' | 'rising' | 'standing' = 'init';
  startSeated = false;
  stands: number[] = [];
  riseOnsets: number[] = [];
  seats: number[] = [];
  sitOnsets: number[] = [];
  partialRises = 0;
  private riseT = 0;
  private candStand: number | null = null;
  private candSit: number | null = null;
  private lastStand = 0;
  private armed = false;
  private prevT: number | null = null; // time of the previous ok sample

  constructor(private readonly mp: MobilityParams) {}

  update(s: Sample): void {
    if (!s.ok) return;
    const mp = this.mp, t = s.t;
    // Dwell rule (mirrors RepMachine in mobility.py): frames further apart than the dwell cannot show a hold, so at low or
    // irregular pose rates ONE frame of the posture is enough; at normal rates the two-frame check applies.
    const dt = this.prevT === null ? 0 : t - this.prevT;
    this.prevT = t;
    const isSit = s.kneeDeg < mp.sit_deg;
    const isStand = s.kneeDeg >= mp.stand_deg && s.hipH >= mp.stand_hip_frac;
    if (this.state === 'init') {
      this.startSeated = isSit;
      this.state = isSit ? 'seated' : 'standing';
      this.lastStand = t;
      return;
    }
    if (this.state === 'seated') {
      if (s.kneeDeg > mp.rise_deg) { this.state = 'rising'; this.riseT = t; this.armed = true; this.candStand = this.candSit = null; }
    }
    if (this.state === 'rising') { // (also on the frame that started the rise: at low rates it may already be standing)
      if (isStand) {
        if (this.candStand === null) this.candStand = t;
        if (t - this.candStand >= mp.stand_dwell_s - 1e-9 || dt >= mp.stand_dwell_s - 1e-9) {
          this.state = 'standing';
          this.stands.push(this.candStand);
          this.riseOnsets.push(this.riseT);
          this.lastStand = t; this.candSit = null;
        }
        return;
      }
      this.candStand = null;
      if (isSit) {
        if (this.candSit === null) this.candSit = t;
        if (t - this.candSit >= mp.sit_dwell_s - 1e-9 || dt >= mp.sit_dwell_s - 1e-9) { this.state = 'seated'; this.candSit = null; this.partialRises++; }
      } else this.candSit = null;
    } else if (this.state === 'standing') {
      if (isStand) this.lastStand = t;
      if (isSit) {
        if (this.candSit === null) this.candSit = t;
        if (t - this.candSit >= mp.sit_dwell_s - 1e-9 || dt >= mp.sit_dwell_s - 1e-9) {
          this.state = 'seated';
          if (this.armed) { this.seats.push(this.candSit); this.sitOnsets.push(this.lastStand); }
          this.armed = false; this.candSit = null;
        }
      } else this.candSit = null;
    }
  }

  get inProgress(): boolean { return (this.state === 'rising' || this.state === 'standing') && this.armed; }
}

// ------------------------------------------------------------------ the tests

export interface ChairStandResult {
  test: 'chair_stand_30s'; reported: boolean; reason: MobilityReason; reps: number | null; rep_times_s: number[] | null;
  partial_rises: number | null; rep_in_progress: boolean | null; quality: number;
}
export interface StsResult { test: 'sit_to_stand_5x'; reported: boolean; reason: MobilityReason; time_s: number | null; rep_times_s: number[] | null; stands: number; quality: number }
export interface TugResult { test: 'tug'; reported: boolean; reason: MobilityReason; time_s: number | null; phases_s: Record<Phase, number> | null; quality: number }
export interface WalkSegment { t_start: number; duration_s: number; distance_m: number; speed_m_s: number }
export interface WalkResult {
  test: 'walking_speed'; reported: boolean; reason: MobilityReason; speed_m_s: number | null; cadence_spm: number | null;
  distance_m: number | null; duration_s: number | null; quality: number; n_segments: number; n_not_side_on: number; segments: WalkSegment[];
}
export type MobilityResult = ChairStandResult | StsResult | TugResult | WalkResult;

export function chairStand30s(samples: Sample[], goT: number, mp: MobilityParams): ChairStandResult {
  const win = samples.filter((s) => goT <= s.t && s.t <= goT + mp.chair_stand_s);
  const q = quality(win);
  const base = { test: 'chair_stand_30s' as const, reps: null, rep_times_s: null, partial_rises: null, rep_in_progress: null, quality: q };
  if (q < mp.min_quality) return { ...base, reported: false, reason: 'out_of_view' };
  const m = new RepMachine(mp);
  for (const s of win) m.update(s);
  if (!m.startSeated) return { ...base, reported: false, reason: 'not_seated_at_start' };
  return { ...base, reported: true, reason: 'ok', reps: m.seats.length, rep_times_s: m.seats.map((t) => t - goT), partial_rises: m.partialRises, rep_in_progress: m.inProgress };
}

const clockStart = (m: RepMachine, goT: number): number => (m.riseOnsets.length ? Math.min(goT, m.riseOnsets[0]!) : goT);

export function sitToStand5x(samples: Sample[], goT: number, mp: MobilityParams): StsResult {
  const m = new RepMachine(mp);
  let endT: number | null = null;
  for (const s of samples) {
    if (s.t < goT - mp.sts_early_s) continue;
    m.update(s);
    if (m.stands.length >= mp.sts_reps) { endT = m.stands[mp.sts_reps - 1]!; break; }
  }
  const lastT = endT ?? (samples.length ? samples[samples.length - 1]!.t : goT);
  const q = quality(samples.filter((s) => goT <= s.t && s.t <= lastT));
  const nul = { test: 'sit_to_stand_5x' as const, time_s: null, rep_times_s: null, stands: m.stands.length, quality: q };
  if (q < mp.min_quality) return { ...nul, reported: false, reason: 'out_of_view' };
  if (!m.startSeated) return { ...nul, reported: false, reason: 'not_seated_at_start' };
  if (endT === null) return { ...nul, reported: false, reason: 'incomplete' };
  const start = clockStart(m, goT);
  const st = m.stands.slice(0, mp.sts_reps);
  const reps = [st[0]! - start];
  for (let i = 1; i < st.length; i++) reps.push(st[i]! - st[i - 1]!);
  return { test: 'sit_to_stand_5x', reported: true, reason: 'ok', time_s: endT - start, rep_times_s: reps, stands: st.length, quality: q };
}

export function tug(samples: Sample[], goT: number, mp: MobilityParams): TugResult {
  const tp = mp.tug;
  const m = new RepMachine(mp);
  for (const s of samples) if (s.t >= goT - mp.sts_early_s) m.update(s);
  const endT = m.seats.length ? m.seats[m.seats.length - 1]! : samples.length ? samples[samples.length - 1]!.t : goT;
  const q = quality(samples.filter((s) => goT <= s.t && s.t <= endT));
  const no = (reason: MobilityReason): TugResult => ({ test: 'tug', reported: false, reason, time_s: null, phases_s: null, quality: q });
  if (q < mp.min_quality) return no('out_of_view');
  if (!m.startSeated) return no('not_seated_at_start');
  if (!m.stands.length) return no('no_rise');
  const start = clockStart(m, goT);
  const use = samples.filter((s) => s.ok && s.t >= goT - mp.sts_early_s);
  const xSeat = use[0]!.x;
  // the stand that starts the walk: the last stand at or before the farthest point reached after the first stand
  const first = use.filter((s) => s.t >= m.stands[0]!);
  let far = 0;
  for (let i = 1; i < first.length; i++) if (Math.abs(first[i]!.x - xSeat) > Math.abs(first[far]!.x - xSeat)) far = i;
  const standT = Math.max(...m.stands.filter((st) => st <= first[far]!.t));
  const after = use.filter((s) => s.t >= standT);
  const hs = after.filter((s) => s.height !== null).map((s) => s.height as number);
  if (!hs.length) return no('out_of_view');
  const body = median(hs);
  const ts = after.map((s) => s.t), xs = after.map((s) => s.x);
  const exc = xs.map((x) => Math.abs(x - xSeat));
  let ext = 0;
  for (let i = 1; i < exc.length; i++) if (exc[i]! > exc[ext]!) ext = i; // first maximum
  if (exc[ext]! / body < tp.min_excursion_bodies || ext === exc.length - 1) return no('no_turn');
  if (exc[ext]! - Math.min(...exc.slice(ext + 1)) < tp.min_return_frac * exc[ext]!) return no('no_turn');
  const v = smoothedVelocity(ts, xs, tp.vel_win_s);
  const outSpeeds: number[] = [];
  for (let i = 0; i <= ext; i++) if (Math.abs(v[i]!) >= mp.walk.min_speed_bodies_s * body) outSpeeds.push(Math.abs(v[i]!));
  if (!outSpeeds.length) return no('no_turn');
  const thr = tp.turn_speed_frac * median(outSpeeds);
  let s0 = ext;
  while (s0 > 0 && Math.abs(v[s0 - 1]!) < thr) s0--;
  let e0 = ext;
  while (e0 < v.length - 1 && Math.abs(v[e0 + 1]!) < thr) e0++;
  const k = m.seats.findIndex((st) => st > ts[e0]!);
  if (k < 0) return no('not_seated_at_end');
  const seatedT = m.seats[k]!, sitOnset = m.sitOnsets[k]!;
  let arrival = sitOnset;
  for (let i = e0 + 1; i < v.length; i++) {
    if (ts[i]! > sitOnset) break;
    if (Math.abs(v[i]!) < thr && Math.abs(v[i - 1]!) >= thr) { arrival = ts[i]!; break; }
  }
  const a = standT;
  const b = Math.min(Math.max(ts[s0]!, a), sitOnset);
  const c = Math.min(Math.max(ts[e0]!, b), sitOnset);
  const d = Math.min(Math.max(arrival, c), sitOnset);
  const cuts = [start, a, b, c, d, sitOnset, seatedT];
  const phases = {} as Record<Phase, number>;
  PHASES.forEach((p, i) => { phases[p] = cuts[i + 1]! - cuts[i]!; });
  return { test: 'tug', reported: true, reason: 'ok', time_s: seatedT - start, phases_s: phases, quality: q };
}

export function walkingSpeed(samples: Sample[], heightM: number, mp: MobilityParams): WalkResult {
  const w = mp.walk;
  const nul = { speed_m_s: null, cadence_spm: null, distance_m: null, duration_s: null, n_segments: 0, n_not_side_on: 0, segments: [] as WalkSegment[] };
  const oks = samples.reduce<number[]>((a, s, i) => (s.ok ? (a.push(i), a) : a), []);
  const q = oks.length ? quality(samples.slice(oks[0]!, oks[oks.length - 1]! + 1)) : 0; // in-view span only
  const fail = (reason: MobilityReason, extra: Partial<WalkResult> = {}): WalkResult => ({ test: 'walking_speed', reported: false, reason, quality: q, ...nul, ...extra });
  if (!(w.height_min_m <= heightM && heightM <= w.height_max_m)) return fail('bad_height');
  const use = samples.filter((s) => s.ok);
  const hsAll = use.filter((s) => s.height !== null).map((s) => s.height as number);
  if (q < mp.min_quality || use.length < w.min_samples || !hsAll.length) return fail('out_of_view');
  const hmed = median(hsAll);
  const ts = use.map((s) => s.t), xs = use.map((s) => s.x);
  const v = smoothedVelocity(ts, xs, w.vel_win_s);
  const thr = w.min_speed_bodies_s * hmed;
  const segs: number[][] = [];
  let cur: number[] = [];
  for (let i = 0; i < use.length; i++) {
    const walking = Math.abs(v[i]!) >= thr;
    if (cur.length && (!walking || ts[i]! - ts[cur[cur.length - 1]!]! > w.gap_s || v[i]! > 0 !== v[cur[cur.length - 1]!]! > 0)) { segs.push(cur); cur = []; }
    if (walking) cur.push(i);
  }
  if (cur.length) segs.push(cur);
  const out: WalkSegment[] = [];
  let nCand = 0, nNotSide = 0, totD = 0, totT = 0, cadN = 0, cadT = 0;
  for (const seg of segs) {
    if (seg.length < w.min_samples) continue;
    const vm = median(seg.map((i) => Math.abs(v[i]!)));
    const steady = seg.filter((i) => Math.abs(v[i]!) >= w.steady_frac * vm);
    const run = seg.filter((i) => steady[0]! <= i && i <= steady[steady.length - 1]!);
    if (run.length < w.min_samples) continue;
    nCand++;
    const hr = run.filter((i) => use[i]!.height !== null).map((i) => use[i]!.height as number);
    if (!hr.length) continue;
    const hm = median(hr);
    if (pct(hr, 0.05) < (1 - w.height_tol_frac) * hm || pct(hr, 0.95) > (1 + w.height_tol_frac) * hm) { nNotSide++; continue; }
    const tr = run.map((i) => ts[i]!), xr = run.map((i) => xs[i]!);
    const n = run.length;
    const mt = tr.reduce((s, x) => s + x, 0) / n, mx = xr.reduce((s, x) => s + x, 0) / n;
    let sxx = 0, sxy = 0;
    for (let j = 0; j < n; j++) { sxx += (tr[j]! - mt) * (tr[j]! - mt); sxy += (tr[j]! - mt) * (xr[j]! - mx); }
    const slope = sxx > 0 ? sxy / sxx : 0;
    const dur = tr[n - 1]! - tr[0]!;
    const speed = Math.abs(slope) * ((w.nose_ankle_frac * heightM) / hm);
    const dist = speed * dur;
    if (dur < w.min_dur_s && dist < w.min_dist_m) continue;
    out.push({ t_start: tr[0]!, duration_s: dur, distance_m: dist, speed_m_s: speed });
    totD += dist; totT += dur;
    const dd = run.filter((i) => use[i]!.ankleDx !== null).map((i) => [use[i]!.t, use[i]!.ankleDx as number] as [number, number]);
    if (dd.length >= 8) {
      const mean = dd.reduce((s, r) => s + r[1], 0) / dd.length;
      const amp = pct(dd.map((r) => Math.abs(r[1] - mean)), 0.9);
      const hy = w.cadence_hyst_frac * amp;
      let state = 0;
      const cross: number[] = [];
      for (const [t, d] of dd) {
        const dv = d - mean;
        if (dv > hy) { if (state === -1) cross.push(t); state = 1; }
        else if (dv < -hy) { if (state === 1) cross.push(t); state = -1; }
      }
      if (cross.length >= w.cadence_min_crossings && cross[cross.length - 1]! > cross[0]!) { cadN += cross.length - 1; cadT += cross[cross.length - 1]! - cross[0]!; }
    }
  }
  const base = { n_segments: out.length, n_not_side_on: nNotSide, segments: out };
  if (!out.length) {
    let reason: MobilityReason;
    if (nNotSide) reason = 'not_side_on';
    else if (nCand) reason = 'too_short';
    else reason = pct(hsAll, 0.05) < (1 - w.height_tol_frac) * hmed || pct(hsAll, 0.95) > (1 + w.height_tol_frac) * hmed ? 'not_side_on' : 'no_walk';
    return fail(reason, base);
  }
  return { test: 'walking_speed', reported: true, reason: 'ok', speed_m_s: totD / totT, cadence_spm: cadT > 0 ? (cadN / cadT) * 60 : null, distance_m: totD, duration_s: totT, quality: q, ...base };
}

// ------------------------------------------------------------------ fixture replay (parity)

/** Replays `mobility` fixtures (carelens.mobility.run_mobility). */
export function runMobilityFixture(input: any, params: { tracking: TrackingParams; mobility: MobilityParams }): any {
  const mp = params.mobility;
  const frames = (input.landmarks as (number[][] | null)[]).map((f) => (f ? expandLandmarks(f, input.landmark_indices) : null));
  const ts: number[] = input.t ?? frames.map((_, i) => i / input.fps);
  const samples = trackSamples(ts, frames, input.aspect, params.tracking, mp);
  switch (input.test as MobilityTest) {
    case 'chair_stand_30s': return chairStand30s(samples, input.go_t, mp);
    case 'sit_to_stand_5x': return sitToStand5x(samples, input.go_t, mp);
    case 'tug': return tug(samples, input.go_t, mp);
    case 'walking_speed': return walkingSpeed(samples, input.height_m ?? mp.walk.height_default_m, mp);
    default: throw new Error(`unknown mobility test ${input.test}`);
  }
}
