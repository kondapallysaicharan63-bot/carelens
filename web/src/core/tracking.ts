// Presence debouncing and short-term single-person tracking for pose dropouts: DOM-free TypeScript port of
// carelens/tracking.py (One Euro filter + PresenceTracker), thresholds from shared/params.json -> tracking.
// Single-person heuristic only; NOT identity tracking. Not a medical device.

import type { Landmarks } from './features';

export interface TrackingParams {
  hold_s: number; enter_s: number; max_jump_frame_heights: number; one_euro_min_cutoff_hz: number;
  one_euro_beta: number; one_euro_d_cutoff_hz: number; one_euro_filter_default_beta: number;
}

const CENTROID_KPS = [11, 12, 23, 24];

/** One Euro filter (Casiez, Roussel, Vogel, CHI 2012) over a flat vector, time-based. */
export class OneEuroFilter {
  private t: number | null = null;
  private x: Float64Array | null = null;
  private dx: Float64Array | null = null;

  constructor(private readonly minCutoff: number, private readonly beta: number, private readonly dCutoff: number) {}

  reset(): void { this.t = null; this.x = null; this.dx = null; }

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(t: number, x: ArrayLike<number>): Float64Array {
    const n = x.length;
    if (this.t === null || this.x === null || this.dx === null) {
      this.t = t; this.x = Float64Array.from(x); this.dx = new Float64Array(n);
      return Float64Array.from(x);
    }
    const dt = Math.max(t - this.t, 1e-3);
    const aD = OneEuroFilter.alpha(this.dCutoff, dt);
    for (let i = 0; i < n; i++) {
      const dx = (x[i]! - this.x[i]!) / dt;
      this.dx[i] = aD * dx + (1 - aD) * this.dx[i]!;
      const cutoff = this.minCutoff + this.beta * Math.abs(this.dx[i]!);
      const a = OneEuroFilter.alpha(cutoff, dt);
      this.x[i] = a * x[i]! + (1 - a) * this.x[i]!;
    }
    this.t = t;
    return Float64Array.from(this.x);
  }
}

const copy = (lm: Landmarks): Landmarks => lm.map((r) => r.slice());

/** update(t, lm, aspect) -> [landmarks | null, held]. `held` frames repeat the last smoothed pose and are NOT new
 *  measurements. Jumps never cause absence (a fast fall at low fps passes raw landmarks through); only a gap longer
 *  than hold_s drops presence. */
export class PresenceTracker {
  present = false;
  private readonly filt: OneEuroFilter;
  private firstT: number | null = null;
  private lastT: number | null = null;
  private lastOut: Landmarks | null = null;
  private core: number[][] | null = null;

  constructor(private readonly p: TrackingParams) {
    this.filt = new OneEuroFilter(p.one_euro_min_cutoff_hz, p.one_euro_beta, p.one_euro_d_cutoff_hz);
  }

  private resetState(): void {
    this.filt.reset();
    this.present = false;
    this.firstT = null;
    this.lastOut = null;
    this.core = null;
  }

  /** Camera unobserved for gapS: reset the smoothing filters but keep presence state, as if no time had passed. */
  skipGap(gapS: number): void {
    this.filt.reset();
    this.core = null;
    if (this.lastT !== null) this.lastT += gapS;
    if (this.firstT !== null) this.firstT += gapS;
  }

  update(t: number, lm: Landmarks | null, aspect = 1): [Landmarks | null, boolean] {
    const p = this.p;
    if (lm === null) {
      if (this.present && this.lastT !== null && this.lastOut !== null && t - this.lastT <= p.hold_s) return [copy(this.lastOut), true];
      this.resetState();
      return [null, false];
    }
    const core = CENTROID_KPS.map((k) => [lm[k]![0]! * aspect, lm[k]![1]!]);
    let dt = p.enter_s;
    if (this.lastT !== null) {
      dt = t - this.lastT;
      if (dt > p.hold_s) this.resetState();
    }
    if (this.core !== null) {
      let maxD = 0;
      for (let i = 0; i < core.length; i++) maxD = Math.max(maxD, Math.hypot(core[i]![0]! - this.core[i]![0]!, core[i]![1]! - this.core[i]![1]!));
      if (maxD > p.max_jump_frame_heights) this.filt.reset(); // fast move: no smoothing, keep presence
    }
    this.core = core;
    this.lastT = t;
    if (this.firstT === null) this.firstT = t;

    const flat = new Float64Array(lm.length * 2);
    lm.forEach((r, i) => { flat[2 * i] = r[0]!; flat[2 * i + 1] = r[1]!; });
    const sm = this.filt.filter(t, flat);
    const out: Landmarks = lm.map((r, i) => [sm[2 * i]!, sm[2 * i + 1]!, r[2]!]); // visibility passes through
    this.lastOut = out;
    if (!this.present) {
      if (t - this.firstT + dt < p.enter_s - 1e-6 || t === this.firstT) return [null, false];
      this.present = true;
    }
    return [copy(out), false];
  }
}
