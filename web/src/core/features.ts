// Per-frame body features from pose landmarks: DOM-free TypeScript port of carelens/features.py
// (height, torso angle, posture, motion). Operation for operation; thresholds from shared/params.json -> features.
// Landmarks are rows [x, y, visibility] in MediaPipe Pose order (33), x and y normalised by frame width / height.
// Not a medical device.

export type Landmarks = number[][];

export interface FeatureParams {
  min_visibility: number; walk_speed_bodysizes_s: number; smooth_s: number; aspect_w_over_h: number;
  lying_torso_angle_deg: number; standing_leg_ratio: number; default_leg_ratio: number;
}

export type Posture = 'standing' | 'sitting' | 'lying' | 'walking' | 'absent';

export interface BodyState {
  t: number;
  present: boolean;
  height: number;
  extent: number;
  torsoAngle: number; // degrees from vertical (0 upright, 90 horizontal)
  hipY: number; // hip midpoint y (0 top .. 1 bottom), frame-height units
  center: [number, number];
  motion: number; // landmark speed, body sizes per second (smoothed)
  speedX: number;
  posture: Posture;
  /** aspect-corrected landmarks (x multiplied by the frame aspect); null when absent */
  landmarks: Landmarks | null;
}

// MediaPipe Pose indices (carelens/pose.py)
export const L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24, L_ANKLE = 27, R_ANKLE = 28;
export const NOSE = 0, L_EYE = 2, R_EYE = 5;
const CORE = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP];

const mid = (lm: Landmarks, a: number, b: number): [number, number] => [(lm[a]![0]! + lm[b]![0]!) / 2, (lm[a]![1]! + lm[b]![1]!) / 2];

export function absentState(t: number): BodyState {
  return { t, present: false, height: 0, extent: 0, torsoAngle: 0, hipY: 0, center: [0, 0], motion: 0, speedX: 0, posture: 'absent', landmarks: null };
}

export class FeatureExtractor {
  private prev: [number, Landmarks] | null = null;
  private motionHist: [number, number][] = [];
  private xHist: [number, number][] = [];

  /** aspect = frame width / height. x is rescaled so both axes are in "frame-height units". */
  constructor(private readonly p: FeatureParams, private readonly aspect: number = p.aspect_w_over_h) {}

  /** Forget motion history (camera outage): the next frame has no previous pose. */
  reset(): void {
    this.prev = null;
    this.xHist = [];
    this.motionHist = [];
  }

  update(t: number, lmIn: Landmarks | null): BodyState {
    const p = this.p;
    if (lmIn === null) { this.prev = null; this.xHist = []; return absentState(t); }
    let coreVis = 0;
    for (const k of CORE) coreVis += lmIn[k]![2]!;
    if (coreVis / CORE.length < p.min_visibility) { this.prev = null; this.xHist = []; return absentState(t); }
    const lm: Landmarks = lmIn.map((r) => [r[0]! * this.aspect, r[1]!, r[2]!]);

    const vis = lm.map((r) => r[2]! >= p.min_visibility);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < lm.length; i++) {
      if (!vis[i]) continue;
      x0 = Math.min(x0, lm[i]![0]!); x1 = Math.max(x1, lm[i]![0]!);
      y0 = Math.min(y0, lm[i]![1]!); y1 = Math.max(y1, lm[i]![1]!);
    }
    const w = x1 - x0, h = y1 - y0;
    const extent = Math.max(w, h, 1e-3);

    const sh = mid(lm, L_SHOULDER, R_SHOULDER);
    const hip = mid(lm, L_HIP, R_HIP);
    const vx = sh[0] - hip[0], vy = sh[1] - hip[1];
    const torsoLen = Math.hypot(vx, vy) + 1e-6;
    const angle = (Math.atan2(Math.abs(vx), Math.abs(vy)) * 180) / Math.PI;

    let inst = 0;
    if (this.prev !== null) {
      const [pt, plm] = this.prev;
      const dt = Math.max(t - pt, 1e-3);
      let sum = 0, cnt = 0;
      for (let i = 0; i < lm.length; i++) {
        if (vis[i] && plm[i]![2]! >= p.min_visibility) { sum += Math.hypot(lm[i]![0]! - plm[i]![0]!, lm[i]![1]! - plm[i]![1]!); cnt++; }
      }
      if (cnt > 0) inst = sum / cnt / extent / dt;
    }
    this.prev = [t, lm.map((r) => r.slice())];
    this.motionHist.push([t, inst]);
    while (this.motionHist.length && this.motionHist[0]![0] < t - p.smooth_s) this.motionHist.shift();
    let ms = 0;
    for (const [, m] of this.motionHist) ms += m;
    const motion = ms / this.motionHist.length;

    this.xHist.push([t, hip[0]]);
    while (this.xHist.length && this.xHist[0]![0] < t - p.smooth_s) this.xHist.shift();
    let speedX = 0;
    if (this.xHist.length >= 2 && this.xHist[this.xHist.length - 1]![0] > this.xHist[0]![0]) {
      const [ta, xa] = this.xHist[0]!, [tb, xb] = this.xHist[this.xHist.length - 1]!;
      speedX = Math.abs(xb - xa) / (tb - ta) / extent;
    }

    let posture: Posture;
    if (angle > p.lying_torso_angle_deg) {
      posture = 'lying';
    } else {
      const ank = mid(lm, L_ANKLE, R_ANKLE);
      const ankOk = (lm[L_ANKLE]![2]! + lm[R_ANKLE]![2]!) / 2 >= p.min_visibility;
      const legRatio = ankOk ? (ank[1] - hip[1]) / torsoLen : p.default_leg_ratio;
      posture = legRatio > p.standing_leg_ratio ? 'standing' : 'sitting';
      if (posture === 'standing' && speedX > p.walk_speed_bodysizes_s) posture = 'walking';
    }
    return { t, present: true, height: h, extent, torsoAngle: angle, hipY: hip[1], center: [hip[0], hip[1]], motion, speedX, posture, landmarks: lm };
  }
}
