// The pose chain the browser runs, DOM-free: landmarks -> PresenceTracker -> FeatureExtractor -> FallDetector +
// InactivityDetector. Mirrors the wiring in carelens/pipeline.py run(): camera gaps (skip_gap / reset / pause), held
// frames (carried state, never a measurement), per-frame ordering. The fixtures in shared/fixtures replay exactly this.
// Not a medical device.

import { FeatureExtractor, type BodyState, type FeatureParams, type Landmarks } from './features';
import { ActivitySummary, ActivityTimeline, FallDetector, InactivityDetector, type FallParams, type PoseEvent } from './events';
import { PresenceTracker, type TrackingParams } from './tracking';

export interface PoseChainParams {
  common: { still_motion_bodysizes_s: number };
  features: FeatureParams;
  events: { fall: FallParams; inactivity: { seconds: number }; timeline: { max_step_s: number } };
  activity: { min_report_s: number };
  tracking: TrackingParams;
}

export interface PoseChainOptions {
  /** inactivity threshold in seconds (default: params.events.inactivity.seconds) */
  inactivitySeconds?: number;
  /** false replays raw landmarks without the tracker (the `landmarks` fixtures) */
  tracker?: boolean;
}

export interface ChainStep { state: BodyState; held: boolean; events: PoseEvent[]; /** the frame aspect changed (e.g. phone rotated): treated like a gap */ aspectChanged: boolean }

export class PosePipeline {
  private feats: FeatureExtractor | null = null;
  private tracker: PresenceTracker | null;
  private fall: FallDetector;
  private inact: InactivityDetector;
  private s: BodyState | null = null;
  private aspect: number | null = null;

  constructor(private readonly params: PoseChainParams, private readonly opts: PoseChainOptions = {}) {
    const still = params.common.still_motion_bodysizes_s;
    this.tracker = opts.tracker === false ? null : new PresenceTracker(params.tracking);
    this.fall = new FallDetector(params.events.fall, still);
    this.inact = new InactivityDetector(opts.inactivitySeconds ?? params.events.inactivity.seconds, still);
  }

  get inactivitySeconds(): number { return this.inact.seconds; }

  /** Change the inactivity threshold (the demo selector): same detector, another number; its stillness timer restarts. */
  setInactivitySeconds(seconds: number): void {
    this.inact = new InactivityDetector(seconds, this.params.common.still_motion_bodysizes_s);
  }

  /** One pose frame at time t (s). `lm` null = no person detected. gapBefore > 0: the camera was unobserved for that
   *  many seconds before this frame (timestamps already include the jump). */
  step(t: number, lm: Landmarks | null, aspect: number, gapBefore = 0): ChainStep {
    // Python fixes the aspect per run. The browser treats an aspect change (rotation) like a gap of zero length:
    // smoothing and motion history restart with the new aspect and the detectors drop their drop-history
    // (presence and any pending fall are kept; no time is credited as stillness).
    const aspectChanged = this.aspect !== null && Math.abs(aspect - this.aspect) > 1e-3;
    this.aspect = aspect;
    if (gapBefore > 0 || aspectChanged) {
      this.tracker?.skipGap(gapBefore);
      this.feats?.reset();
      this.fall.pause(gapBefore);
      this.inact.pause(gapBefore);
    }
    if (aspectChanged) this.feats = null;
    if (this.feats === null) this.feats = new FeatureExtractor(this.params.features, aspect);
    let held = false;
    let m = lm;
    if (this.tracker) [m, held] = this.tracker.update(t, lm, aspect);
    let s: BodyState;
    if (held && this.s !== null) s = { ...this.s, t }; // carry forward the last measured state incl. its motion
    else s = this.feats.update(t, m);
    this.s = s;
    const events: PoseEvent[] = [];
    const ef = this.fall.update(s, held);
    if (ef) events.push(ef);
    const ei = this.inact.update(s);
    if (ei) events.push(ei);
    return { state: s, held, events, aspectChanged };
  }
}

// ------------------------------------------------------------------ fixture replay (parity)

const N_LANDMARKS = 33;
const FILL_VISIBILITY = 0.2; // carelens.fixtures.LM_FILL_VISIBILITY: unlisted landmarks are sub-threshold

/** Fixture `landmarks[frame]` rows (for the listed MediaPipe indices) -> 33 landmark rows. */
export function expandLandmarks(frame: number[][], indices: number[]): Landmarks {
  const lm: Landmarks = Array.from({ length: N_LANDMARKS }, () => [0, 0, FILL_VISIBILITY]);
  indices.forEach((k, j) => { lm[k] = frame[j]!.slice(0, 3); });
  return lm;
}

const round = (x: number, dp: number): number => { const m = 10 ** dp; return Math.round(x * m) / m; };

/** Replays `landmarks` / `pose_chain` fixtures (see carelens.fixtures.run_landmarks / run_pose_chain). */
export function runPoseFixture(input: any, params: PoseChainParams, withTracker: boolean): any {
  const fps: number = input.fps;
  const cfg = input.config ?? {};
  const gaps = new Map<number, number>((input.gaps ?? []).map((g: [number, number]) => [g[0], g[1]]));
  const pipe = new PosePipeline(params, { tracker: withTracker, inactivitySeconds: cfg.inactivity_seconds });
  const events: any[] = [], features: any[] = [], postures: any[] = [], presence: any[] = [];
  let lastSec: number | null = null;
  (input.landmarks as (number[][] | null)[]).forEach((frame, i) => {
    const t: number = input.t ? input.t[i] : i / fps;
    const lm = frame ? expandLandmarks(frame, input.landmark_indices) : null;
    const { state: s, held, events: evs } = pipe.step(t, lm, input.aspect, gaps.get(i) ?? 0);
    if (withTracker) {
      const st = held ? 'held' : s.present ? 'present' : 'absent';
      if (!presence.length || presence[presence.length - 1].state !== st) presence.push({ t: round(t, 3), state: st });
      if (!postures.length || postures[postures.length - 1].posture !== s.posture) postures.push({ t: round(t, 3), posture: s.posture });
    }
    if (lastSec !== Math.floor(t)) {
      lastSec = Math.floor(t);
      features.push({ t: round(t, 3), present: s.present, posture: s.posture, torso_angle_deg: s.torsoAngle, motion: s.motion });
    }
    for (const e of evs) events.push({ kind: e.kind, t_onset: e.tOnset, t_confirmed: e.tConfirmed, trigger: e.details.trigger ?? null });
  });
  return withTracker
    ? { events, features_1hz: features, posture_runs: postures, presence_runs: presence }
    : { events, features_1hz: features };
}

/** Replays `activity` fixtures (carelens.fixtures.run_activity): the pose chain (tracker, held frames, gaps) feeding the activity
 *  timeline and summary. `input.start` is the naive wall-clock start "YYYY-MM-DD HH:MM:SS". */
export function runActivityFixture(input: any, params: PoseChainParams): any {
  const fps: number = input.fps;
  const gaps = new Map<number, number>((input.gaps ?? []).map((g: [number, number]) => [g[0], g[1]]));
  const pipe = new PosePipeline(params, { tracker: true });
  const m = /^(\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)$/.exec(input.start)!;
  const start = Date.UTC(+m[1]!, +m[2]! - 1, +m[3]!, +m[4]!, +m[5]!, +m[6]!);
  const timeline = new ActivityTimeline(start, params.events.timeline.max_step_s);
  const summary = new ActivitySummary(params.common.still_motion_bodysizes_s, params.events.timeline.max_step_s, params.activity.min_report_s);
  (input.landmarks as (number[][] | null)[]).forEach((frame, i) => {
    const t: number = input.t ? input.t[i] : i / fps;
    const lm = frame ? expandLandmarks(frame, input.landmark_indices) : null;
    const { state } = pipe.step(t, lm, input.aspect, gaps.get(i) ?? 0);
    timeline.update(state);
    summary.update(state);
  });
  return { timeline: timeline.toDict(), summary: summary.toDict() };
}
