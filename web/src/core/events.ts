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
