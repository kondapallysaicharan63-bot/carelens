// Blink rate, eyes-closed time and long closures from MediaPipe FaceLandmarker eye points (W9, Experimental): DOM-free TypeScript
// port of carelens/blink.py (read its docstring for the design: EAR, per-person 90th-percentile baseline, hysteresis, gating).
// Not a medical device: wellbeing trends to compare with your own earlier results. Driven by shared/params.json -> blink.

export interface BlinkParams {
  eye_landmarks: { left: number[]; right: number[] };
  pose_landmarks: { nose: number; forehead: number; chin: number };
  baseline_s: number; baseline_pct: number; closed_frac: number; reopen_frac: number; perclos_frac: number;
  blink_min_ms: number; blink_max_ms: number; rate_window_s: number; min_window_s: number; min_rate_hz: number;
  fps_window_s: number; max_dt_s: number; min_eye_width: number; max_ear: number;
  yaw_max_ratio: number; pitch_neutral: number; pitch_tol: number;
}

export type BlinkReason = 'ok' | 'no_face' | 'low_fps' | 'head_turned' | 'calibrating';

export interface BlinkState {
  reason: BlinkReason; reported: boolean; ear: number | null; baseline: number | null; closed: boolean;
  blinks_per_min: number; perclos: number; long_closures: number; window_s: number; coverage: number; fps: number; n_blinks: number;
}
export interface ClosureEvent { t_start: number; t_end: number; long: boolean }
/** x / y normalised by frame width / height (the FaceLandmarker output); only the indices in params are read */
export type FacePt = { x: number; y: number };

/** Mirrors carelens.params.blink_errors. */
export function blinkErrors(p: BlinkParams): string[] {
  const e: string[] = [];
  for (const side of ['left', 'right'] as const) {
    const pts = p.eye_landmarks[side];
    if (pts.length !== 6 || new Set(pts).size !== 6 || Math.min(...pts) < 0) e.push(`eye_landmarks.${side} must be 6 distinct landmark indices (p1..p6: corner, 2 upper lid, corner, 2 lower lid)`);
  }
  if (Object.keys(p.pose_landmarks).sort().join() !== 'chin,forehead,nose') e.push('pose_landmarks needs exactly nose, forehead, chin');
  if (!(0 < p.perclos_frac && p.perclos_frac < p.closed_frac && p.closed_frac < p.reopen_frac && p.reopen_frac < 1)) e.push('need 0 < perclos_frac < closed_frac < reopen_frac < 1');
  if (!(0 < p.blink_min_ms && p.blink_min_ms < p.blink_max_ms)) e.push('need 0 < blink_min_ms < blink_max_ms');
  if (!(0 < p.min_window_s && p.min_window_s <= p.rate_window_s)) e.push('need 0 < min_window_s <= rate_window_s');
  if (!(0 < p.baseline_pct && p.baseline_pct <= 100)) e.push('baseline_pct must be in (0, 100]');
  if (!(p.min_rate_hz >= 1000 / p.blink_min_ms)) e.push('min_rate_hz is too low to resolve blink_min_ms');
  for (const k of ['baseline_s', 'fps_window_s', 'max_dt_s', 'min_eye_width', 'max_ear', 'yaw_max_ratio', 'pitch_tol', 'pitch_neutral', 'min_rate_hz'] as const) if (!(p[k] > 0)) e.push(`${k} must be > 0`);
  if (!(p.max_dt_s < p.fps_window_s)) e.push('need max_dt_s < fps_window_s');
  return e;
}

export function assertBlinkParams(p: BlinkParams): BlinkParams {
  const errs = blinkErrors(p);
  if (errs.length) throw new Error('invalid blink params: ' + errs.join('; '));
  return p;
}

/** Every landmark index the blink core reads (what a fixture or a scripted provider must supply). */
export function blinkIndices(p: BlinkParams): number[] {
  return [...new Set([...p.eye_landmarks.left, ...p.eye_landmarks.right, p.pose_landmarks.nose, p.pose_landmarks.forehead, p.pose_landmarks.chin])].sort((a, b) => a - b);
}

/** Linear-interpolation percentile (numpy's default). */
export function percentile(values: number[], pct: number): number {
  const s = [...values].sort((a, b) => a - b);
  const k = ((s.length - 1) * pct) / 100;
  const lo = Math.floor(k);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo]! + (s[hi]! - s[lo]!) * (k - lo);
}

const dist = (a: FacePt, b: FacePt, aspect: number): number => Math.hypot((a.x - b.x) * aspect, a.y - b.y);

export function eyeEar(lm: ArrayLike<FacePt>, idx: number[], aspect: number, p: BlinkParams): number | null {
  const q = idx.map((i) => lm[i]!);
  const width = dist(q[0]!, q[3]!, aspect);
  if (!(width >= p.min_eye_width)) return null;
  const ear = (dist(q[1]!, q[5]!, aspect) + dist(q[2]!, q[4]!, aspect)) / (2 * width);
  return ear >= 0 && ear <= p.max_ear ? ear : null;
}

export function frameEar(lm: ArrayLike<FacePt>, aspect: number, p: BlinkParams): number | null {
  const l = eyeEar(lm, p.eye_landmarks.left, aspect, p);
  const r = eyeEar(lm, p.eye_landmarks.right, aspect, p);
  return l === null || r === null ? null : 0.5 * (l + r);
}

export function headTurned(lm: ArrayLike<FacePt>, aspect: number, p: BlinkParams): boolean {
  const a = lm[p.eye_landmarks.left[0]!]!, b = lm[p.eye_landmarks.right[3]!]!;
  const iod = dist(a, b, aspect);
  const nose = lm[p.pose_landmarks.nose]!, fore = lm[p.pose_landmarks.forehead]!, chin = lm[p.pose_landmarks.chin]!;
  const height = chin.y - fore.y;
  if (!(iod > 0) || !(height > 0)) return true;
  const yaw = ((nose.x - 0.5 * (a.x + b.x)) * aspect) / iod;
  const pitch = (nose.y - fore.y) / height;
  return Math.abs(yaw) > p.yaw_max_ratio || Math.abs(pitch - p.pitch_neutral) > p.pitch_tol;
}

/** Streaming tracker: update(t, lm, aspect) per DISTINCT camera frame (lm null = no face). */
export class BlinkTracker {
  private lastT: number | null = null;
  private times: number[] = [];
  private base: [number, number][] = []; // [dt, ear] of valid frames in the baseline window
  private baseSum = 0;
  validTotal = 0;
  private closed = false;
  private needOpen = true; // a closure already under way when counting (re)starts has an unknown start: skip it
  private runStart = 0;
  private runLong = false;
  private longEv: ClosureEvent | null = null; // the open long-closure event: its t_end is set when the eyes reopen or the run is cut off
  readonly events: ClosureEvent[] = [];
  private win: [number, number, boolean][] = []; // [t, dt, deep] of counted frames in the rate window
  private firstCounted: number | null = null;
  last: BlinkState;

  constructor(private readonly p: BlinkParams) {
    this.last = this.state('calibrating', null, null, 0, 0, 0, 0, 0, 0);
  }

  private state(reason: BlinkReason, ear: number | null, base: number | null, bpm: number, perclos: number, longN: number, winS: number, cov: number, rate: number): BlinkState {
    return {
      reason, reported: reason === 'ok', ear, baseline: base, closed: this.closed, blinks_per_min: bpm, perclos, long_closures: longN,
      window_s: winS, coverage: cov, fps: rate, n_blinks: this.events.reduce((n, e) => n + (e.long ? 0 : 1), 0),
    };
  }

  private distinctRate(t: number, gap: boolean): number | null {
    const p = this.p;
    if (gap) this.times = [];
    this.times.push(t);
    while (this.times.length && this.times[0]! < t - p.fps_window_s) this.times.shift();
    const span = this.times[this.times.length - 1]! - this.times[0]!;
    if (this.times.length < 3 || span < 0.5 * p.fps_window_s) return null;
    return (this.times.length - 1) / span;
  }

  private endRun(t: number, reopened: boolean): void {
    const p = this.p;
    if (this.longEv !== null) { this.longEv.t_end = t; this.longEv = null; }
    if (this.closed && reopened) {
      const durMs = (t - this.runStart) * 1000;
      if (durMs >= p.blink_min_ms && durMs <= p.blink_max_ms) this.events.push({ t_start: this.runStart, t_end: t, long: false });
      else if (durMs > p.blink_max_ms && !this.runLong) this.events.push({ t_start: this.runStart, t_end: t, long: true });
    }
    this.closed = false;
    this.runLong = false;
    if (!reopened) this.needOpen = true;
  }

  update(t: number, lm: ArrayLike<FacePt> | null, aspect: number): BlinkState {
    const p = this.p;
    let dt = 0;
    let gap = false;
    if (this.lastT !== null) {
      const raw = t - this.lastT;
      gap = raw > p.max_dt_s;
      dt = gap ? 0 : Math.max(raw, 0);
    }
    this.lastT = t;
    if (gap) this.endRun(t, false);
    const rate = this.distinctRate(t, gap);

    const ear = lm !== null ? frameEar(lm, aspect, p) : null;
    if (ear === null) {
      this.endRun(t, false);
      return (this.last = this.finish('no_face', null, rate, null));
    }
    if (rate === null) { // rate not yet verified (start, or after a frame gap): never count at an unverified rate
      this.endRun(t, false);
      return (this.last = this.finish('calibrating', ear, rate, null));
    }
    if (rate < p.min_rate_hz) {
      this.endRun(t, false);
      return (this.last = this.finish('low_fps', ear, rate, null));
    }
    if (headTurned(lm!, aspect, p)) {
      this.endRun(t, false);
      return (this.last = this.finish('head_turned', ear, rate, null));
    }

    this.base.push([dt, ear]);
    this.baseSum += dt;
    while (this.base.length > 1 && this.baseSum - this.base[0]![0] >= p.baseline_s) this.baseSum -= this.base.shift()![0];
    this.validTotal += dt;
    const calibrated = this.validTotal >= p.baseline_s;
    const base = percentile(this.base.map((e) => e[1]), p.baseline_pct);
    if (!calibrated || !(base > 0)) {
      this.needOpen = true;
      return (this.last = this.finish('calibrating', ear, rate, base > 0 ? base : null));
    }

    if (this.firstCounted === null) this.firstCounted = t;
    const deep = ear < p.perclos_frac * base;
    this.win.push([t, dt, deep]);
    if (this.needOpen) {
      if (ear >= p.closed_frac * base) this.needOpen = false;
    } else if (!this.closed) {
      if (ear < p.closed_frac * base) { this.closed = true; this.runStart = t; this.runLong = false; }
    } else if (ear > p.reopen_frac * base) {
      this.endRun(t, true);
    } else if (!this.runLong && (t - this.runStart) * 1000 > p.blink_max_ms) {
      this.runLong = true;
      this.longEv = { t_start: this.runStart, t_end: t, long: true };
      this.events.push(this.longEv);
    }
    return (this.last = this.finish('ok', ear, rate, base));
  }

  private finish(reasonIn: BlinkReason, ear: number | null, rate: number | null, base: number | null): BlinkState {
    const p = this.p;
    const t = this.lastT!;
    let reason = reasonIn;
    while (this.win.length && this.win[0]![0] < t - p.rate_window_s) this.win.shift();
    let counted = 0, deepS = 0;
    for (const [, d, k] of this.win) { counted += d; if (k) deepS += d; }
    const lo = t - p.rate_window_s;
    let nBlink = 0, nLong = 0;
    for (const e of this.events) if (e.t_start >= lo) { if (e.long) nLong++; else nBlink++; }
    const span = this.firstCounted !== null ? Math.min(p.rate_window_s, t - this.firstCounted) : 0;
    const cov = span > 0 ? Math.min(1, counted / span) : 0;
    if (reason === 'ok' && counted < p.min_window_s) reason = 'calibrating';
    const bpm = counted > 0 ? (nBlink / counted) * 60 : 0;
    const perclos = counted > 0 ? deepS / counted : 0;
    return this.state(reason, ear, base, bpm, perclos, nLong, counted, cov, rate ?? 0);
  }
}

/** Replays `blink` fixtures (carelens.blink.run_blink). */
export function runBlinkFixture(input: any, params: { blink: BlinkParams }): any {
  const idx = input.landmark_indices as number[];
  const aspect = input.w / input.h;
  const tr = new BlinkTracker(params.blink);
  const runs: { t: number; reason: string }[] = [];
  const updates: any[] = [];
  let lastSec: number | null = null;
  const round3 = (x: number): number => Math.round(x * 1000) / 1000;
  (input.landmarks as (number[][] | null)[]).forEach((frame, i) => {
    const t: number = input.t ? input.t[i] : i / input.fps;
    let lm: FacePt[] | null = null;
    if (frame) { lm = []; idx.forEach((k, j) => { lm![k] = { x: frame[j]![0]!, y: frame[j]![1]! }; }); }
    const u = tr.update(t, lm, aspect);
    if (!runs.length || runs[runs.length - 1]!.reason !== u.reason) runs.push({ t: round3(t), reason: u.reason });
    if (lastSec !== Math.floor(t)) {
      lastSec = Math.floor(t);
      updates.push({ t: round3(t), reason: u.reason, blinks_per_min: u.blinks_per_min, perclos: u.perclos, long_closures: u.long_closures, ear: u.ear, baseline: u.baseline });
    }
  });
  const { closed: _closed, ...final } = tr.last;
  return {
    blinks: tr.events.filter((e) => !e.long).map((e) => ({ t_start: e.t_start, t_end: e.t_end })),
    long_closures: tr.events.filter((e) => e.long).map((e) => ({ t_start: e.t_start, t_end: e.t_end })),
    reason_runs: runs, updates_1hz: updates, final,
  };
}
