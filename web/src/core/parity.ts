// Parity registry + comparison for shared/fixtures (DOM-free, no node imports).
// Each TS implementation of a Python-reference algorithm is imported HERE and registered by fixture
// name, so every vitest file sees the same map (a registry inside a test file is per-file).
// Not a medical device.

import { runContactPpg } from './ppg';
import { runFaceRppg } from './rppg';
import { runPoseFixture } from './pipeline';
import { BreathingRateEstimator } from './breathing';

export interface Fixture {
  name: string;
  description: string;
  input_level: string;
  params_version: number;
  input: any;
  expected: any;
  tolerance: Record<string, number | string>;
}

export type Implementation = (fx: Fixture, params: any) => any;

const ppgContact: Implementation = (fx, params) => runContactPpg(fx.input, params.contact_ppg);

const faceRppg: Implementation = (fx, params) => runFaceRppg(fx.input, params.rppg);

const landmarks: Implementation = (fx, params) => runPoseFixture(fx.input, params, false);

const poseChain: Implementation = (fx, params) => runPoseFixture(fx.input, params, true);

/** chest_trace_breathing: cumulative chest displacement per frame -> BreathingRateEstimator. */
const chestBreathing: Implementation = (fx, params) => {
  const est = new BreathingRateEstimator(params.vitals, params.common.min_quality);
  const fps: number = fx.input.fps;
  (fx.input.chest_y_px as number[]).forEach((y, i) => est.updateDisplacement(i / fps, y));
  return { estimates: est.estimates };
};

export const IMPLEMENTATIONS: Record<string, Implementation> = {
  breathing_10bpm: chestBreathing,
  breathing_15bpm: chestBreathing,
  breathing_20bpm: chestBreathing,
  fall_sudden: landmarks,
  fall_slow_lie_down: landmarks,
  inactivity_sit_still: landmarks,
  track_dropout_fill: poseChain,
  track_enter_hysteresis: poseChain,
  track_fast_fall_5fps: poseChain,
  posture_sequence: poseChain,
  fall_camera_gap: poseChain,
  ppg_clean_60bpm: ppgContact,
  ppg_clean_90bpm: ppgContact,
  ppg_clean_150bpm: ppgContact,
  ppg_jitter_drops: ppgContact,
  ppg_motion_burst: ppgContact,
  ppg_saturation: ppgContact,
  ppg_finger_lift: ppgContact,
  ppg_harmonic_trap: ppgContact,
  rppg_clean_light_72bpm: faceRppg,
  rppg_clean_medium_90bpm: faceRppg,
  rppg_clean_dark_60bpm: faceRppg,
  rppg_motion_burst: faceRppg,
  rppg_lighting_step: faceRppg,
  rppg_face_lost: faceRppg,
  rppg_small_face: faceRppg,
  rppg_skin_filter: faceRppg,
  rppg_disagree_flicker: faceRppg,
};

const TOL_KEYS: Record<string, [string, number]> = {
  value: ['value_bpm', 0.5], quality: ['quality', 0.05], t: ['t_s', 0.05],
  t_onset: ['event_time_s', 0.1], t_confirmed: ['event_time_s', 0.1],
  torso_angle_deg: ['torso_angle_deg', 0.05], motion: ['motion', 0.01],
  bpm: ['bpm', 0.3], bpm_ibi: ['bpm', 0.3], bpm_chrom: ['bpm', 0.3], final_ibis_ms: ['ibi_ms', 5], snr: ['snr', 0.05], pi: ['pi', 0.01],
};
const TOL_OTHER = 1e-4; // numeric fields without a named tolerance

/** Same semantics as carelens.fixtures.compare_outputs: numbers within tolerance by field name;
 *  strings/booleans/null (reported, kind, posture, ...), array lengths and key sets exactly. */
export function compareOutputs(fx: Fixture, actual: any, expected: any = fx.expected): string[] {
  const errs: string[] = [];
  const tolFor = (field: string): number => {
    const e = TOL_KEYS[field];
    if (!e) return TOL_OTHER;
    const v = fx.tolerance[e[0]];
    return typeof v === 'number' ? v : e[1];
  };
  const walk = (e: any, a: any, path: string, field: string): void => {
    if (typeof e === 'number') {
      if (typeof a !== 'number' || !(Math.abs(e - a) <= tolFor(field) + 1e-9)) {
        errs.push(`${path}: expected ${e} got ${a} (tol ${tolFor(field)})`);
      }
    } else if (Array.isArray(e)) {
      if (!Array.isArray(a) || a.length !== e.length) {
        errs.push(`${path}: length ${Array.isArray(a) ? a.length : a} != ${e.length}`);
      } else e.forEach((x, i) => walk(x, a[i], `${path}[${i}]`, field));
    } else if (e && typeof e === 'object') {
      const ka = a && typeof a === 'object' && !Array.isArray(a) ? Object.keys(a).sort() : null;
      const ke = Object.keys(e).sort();
      if (!ka || ka.join() !== ke.join()) errs.push(`${path}: keys differ`);
      else for (const k of ke) walk(e[k], a[k], `${path}.${k}`, k);
    } else if (e !== a) {
      errs.push(`${path}: expected ${JSON.stringify(e)} got ${JSON.stringify(a)}`);
    }
  };
  walk(expected, actual, 'expected', '');
  return errs;
}

export interface ParityReport {
  executed: string[];
  pending: string[];
  /** pending now but not in the committed pending list: an implementation was dropped (or a new fixture is unported) */
  newlyPending: string[];
  failures: string[];
}

export function runParity(
  fixtures: Fixture[], impls: Record<string, Implementation>, params: any, allowedPending: string[],
): ParityReport {
  const r: ParityReport = { executed: [], pending: [], newlyPending: [], failures: [] };
  for (const fx of fixtures) {
    const impl = impls[fx.name];
    if (!impl) {
      r.pending.push(fx.name);
      if (!allowedPending.includes(fx.name)) r.newlyPending.push(fx.name);
      continue;
    }
    r.executed.push(fx.name);
    let errs: string[];
    try { errs = compareOutputs(fx, impl(fx, params)); } catch (e) { errs = [`${fx.name}: threw ${e}`]; }
    r.failures.push(...errs.map((m) => `${fx.name}: ${m}`));
  }
  return r;
}
