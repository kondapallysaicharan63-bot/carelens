// Fall detection and inactivity: DOM-free TypeScript port of carelens/events.py (FallDetector incl. the camera-gap
// pause and held-frame rules, InactivityDetector). Operation for operation; thresholds from shared/params.json -> events.
// Wellness wording only: these raise "check-in suggested" events, never a diagnosis. Not a medical device.

import type { BodyState } from './features';

export interface FallParams {
  drop_window_s: number; upright_deg: number; lying_deg: number; drop_frac: number; min_angle_rate_deg_s: number;
  hip_drop_min_rate_frac_s: number; still_seconds: number; pending_timeout_s: number;
}
export interface InactivityParams { seconds: number }

export interface PoseEvent {
  kind: 'fall' | 'inactivity';
  tOnset: number;
  tConfirmed: number;
  message: string;
  details: Record<string, unknown> & { trigger?: string };
}

const round = (x: number, dp: number): number => { const m = 10 ** dp; return Math.round(x * m) / m; };

interface Pending {
  tOnset: number; stillStart: number; info: Record<string, unknown> & { trigger?: string };
  gap: number; gapTotal: number; lastT: number; postGap: boolean;
}

/** Sudden drop in torso angle or hip height, then little movement for `still_seconds`. Standing back up cancels a pending fall. */
export class FallDetector {
  private hist: [number, number, number, number][] = []; // t, torso angle, hip y, extent
  private pending: Pending | null = null;
  private armed = true; // re-armed once the person is upright again

  constructor(private readonly p: FallParams, private readonly stillMotion: number, private readonly stillSeconds: number = p.still_seconds) {}

  private suddenDrop(s: BodyState): (Record<string, unknown> & { trigger: string }) | null {
    const p = this.p;
    for (const [t0, ang0, hip0, size0] of this.hist) {
      if (ang0 > p.upright_deg) continue;
      const dt = Math.max(s.t - t0, 1e-3);
      if (s.torsoAngle > p.lying_deg && (s.torsoAngle - ang0) / dt >= p.min_angle_rate_deg_s) {
        return { trigger: 'torso_angle', from_deg: round(ang0, 1), to_deg: round(s.torsoAngle, 1), drop_time_s: round(dt, 2),
          angle_rate_deg_s: round((s.torsoAngle - ang0) / dt, 1) };
      }
      if (s.hipY - hip0 > p.drop_frac * size0 && (s.hipY - hip0) / size0 / dt >= p.hip_drop_min_rate_frac_s) {
        return { trigger: 'hip_drop', drop_body_frac: round((s.hipY - hip0) / size0, 2), drop_time_s: round(dt, 2) };
      }
    }
    return null;
  }

  /** The camera was unobserved for gapS seconds (timestamps jumped by that much). Unobserved time neither counts as
   *  stillness nor erases a pending fall: the pending state is kept and the gap is excluded from both clocks. */
  pause(gapS: number): void {
    this.hist = [];
    const pd = this.pending;
    if (pd !== null) {
      pd.gap += gapS;
      pd.gapTotal += gapS;
      pd.lastT += gapS;
      pd.postGap = true; // from now on only PRESENT frames count as still
    }
  }

  /** held=true: `s` is a carried-forward state (pose dropout bridged by the tracker). It is not a measurement, so it
   *  never enters the drop history or triggers a drop; its carried motion still feeds the stillness timer. */
  update(s: BodyState, held = false): PoseEvent | null {
    const p = this.p;
    if (s.present && !held) this.hist.push([s.t, s.torsoAngle, s.hipY, s.extent]);
    while (this.hist.length && this.hist[0]![0] < s.t - p.drop_window_s) this.hist.shift();

    if (s.present && !held && s.torsoAngle < p.upright_deg && (s.posture === 'standing' || s.posture === 'walking')) {
      if (this.pending !== null) this.pending = null; // recovered
      this.armed = true;
    }

    if (this.pending === null) {
      if (this.armed && s.present && !held) {
        const info = this.suddenDrop(s);
        if (info) {
          this.pending = { tOnset: s.t, stillStart: s.t, info, gap: 0, gapTotal: 0, lastT: s.t, postGap: false };
          this.armed = false;
        }
      }
      return null;
    }

    const pd = this.pending;
    if (s.t - pd.tOnset - pd.gapTotal > p.pending_timeout_s) { this.pending = null; return null; }
    const dt = s.t - pd.lastT;
    pd.lastT = s.t;
    if (pd.postGap && !s.present) pd.gap += dt; // after an outage an empty room neither confirms nor counts
    if (s.present && s.motion > this.stillMotion) { pd.stillStart = s.t; pd.gap = 0; } // moving: restart stillness timer
    const still = s.t - pd.stillStart - pd.gap; // observed stillness only
    if (still >= this.stillSeconds) {
      this.pending = null;
      return {
        kind: 'fall', tOnset: round(pd.tOnset, 2), tConfirmed: round(s.t, 2),
        message: 'Possible fall: sudden drop followed by little movement. Check-in suggested.',
        details: { ...pd.info, still_for_s: round(still, 1) },
      };
    }
    return null;
  }
}

/** No meaningful movement for `seconds` while the person is present. */
export class InactivityDetector {
  private start: number | null = null;
  private gap = 0;
  private fired = false;

  constructor(readonly seconds: number, private readonly stillMotion: number) {}

  /** Camera unobserved for gapS: that time does not count toward inactivity. */
  pause(gapS: number): void { if (this.start !== null) this.gap += gapS; }

  update(s: BodyState): PoseEvent | null {
    if (!s.present || s.motion > this.stillMotion) { this.start = null; this.gap = 0; this.fired = false; return null; }
    if (this.start === null) this.start = s.t;
    if (!this.fired && s.t - this.start - this.gap >= this.seconds) {
      this.fired = true;
      return {
        kind: 'inactivity', tOnset: round(this.start, 2), tConfirmed: round(s.t, 2),
        message: `No meaningful movement for ${(this.seconds / 60).toFixed(1)} min while present (${s.posture}). Check-in suggested.`,
        details: { posture: s.posture },
      };
    }
    return null;
  }
}

// ------------------------------------------------------------------ activity (W9, Experimental)

export const POSTURES = ['sitting', 'standing', 'walking', 'lying', 'absent'] as const;

const p2 = (n: number): string => String(n).padStart(2, '0');

/** A wall-clock time as "naive" milliseconds: the local calendar fields read as if they were UTC (Python's naive datetime). */
export function naiveWallMs(d: Date): number {
  return Date.UTC(d.getFullYear(), d.getMonth(), d.getDate(), d.getHours(), d.getMinutes(), d.getSeconds(), d.getMilliseconds());
}

/** Seconds spent in each posture, bucketed by wall-clock hour (port of carelens.events.ActivityTimeline). `startNaiveMs` is the
 *  session start as naive wall-clock milliseconds (see naiveWallMs). */
export class ActivityTimeline {
  readonly buckets = new Map<string, Record<string, number>>();
  private lastT: number | null = null;

  constructor(public startNaiveMs: number, private readonly maxStepS: number) {}

  update(s: BodyState): void {
    if (this.lastT !== null) {
      const dt = Math.min(s.t - this.lastT, this.maxStepS);
      const d = new Date(this.startNaiveMs + Math.trunc(s.t * 1000));
      const hour = `${d.getUTCFullYear()}-${p2(d.getUTCMonth() + 1)}-${p2(d.getUTCDate())} ${p2(d.getUTCHours())}:00`;
      let b = this.buckets.get(hour);
      if (!b) { b = {}; this.buckets.set(hour, b); }
      b[s.posture] = (b[s.posture] ?? 0) + dt;
    }
    this.lastT = s.t;
  }

  /** hour -> posture -> seconds, rounded to 0.1 s like the Python reference */
  toDict(): Record<string, Record<string, number>> {
    const out: Record<string, Record<string, number>> = {};
    for (const h of [...this.buckets.keys()].sort()) {
      const v = this.buckets.get(h)!;
      out[h] = Object.fromEntries(POSTURES.map((p) => [p, round(v[p] ?? 0, 1)]));
    }
    return out;
  }

  /** browser only: the frame clock started over (a new camera stream), so the next step is not measured against the old time */
  breakStream(): void { this.lastT = null; }

  /** seconds per posture over the whole session */
  totals(): Record<string, number> {
    const out: Record<string, number> = Object.fromEntries(POSTURES.map((p) => [p, 0]));
    for (const v of this.buckets.values()) for (const p of POSTURES) out[p]! += v[p] ?? 0;
    return out;
  }
}

export interface ActivitySummaryResult {
  reported: boolean; reason: 'ok' | 'too_short'; active_s: number; still_s: number; absent_s: number; total_s: number; longest_still_s: number; active_min: number;
}

/** Time active vs still vs away over a session (port of carelens.events.ActivitySummary). Same per-frame motion signal and
 *  still-motion threshold as the inactivity detector; a step above the cap is an outage: it credits 0 s and ends a still stretch. A wellbeing trend, not an assessment. */
export class ActivitySummary {
  activeS = 0;
  stillS = 0;
  absentS = 0;
  longestStillS = 0;
  private run = 0;
  private lastT: number | null = null;

  constructor(private readonly stillMotion: number, private readonly maxStepS: number, private readonly minReportS: number) {}

  update(s: BodyState): void {
    if (this.lastT !== null) {
      const raw = s.t - this.lastT;
      const outage = raw > this.maxStepS;
      if (outage) this.run = 0; // unobserved time breaks a still stretch and is credited to nothing
      const dt = outage ? 0 : raw;
      if (!s.present) { this.absentS += dt; this.run = 0; }
      else if (s.motion > this.stillMotion) { this.activeS += dt; this.run = 0; }
      else { this.stillS += dt; this.run += dt; this.longestStillS = Math.max(this.longestStillS, this.run); }
    }
    this.lastT = s.t;
  }

  /** browser only: see ActivityTimeline.breakStream; also ends a still stretch */
  breakStream(): void { this.lastT = null; this.run = 0; }

  get totalS(): number { return this.activeS + this.stillS + this.absentS; }

  toDict(): ActivitySummaryResult {
    const ok = this.totalS >= this.minReportS;
    return {
      reported: ok, reason: ok ? 'ok' : 'too_short', active_s: this.activeS, still_s: this.stillS, absent_s: this.absentS, total_s: this.totalS,
      longest_still_s: this.longestStillS, active_min: this.activeS / 60,
    };
  }
}
