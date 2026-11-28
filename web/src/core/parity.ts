// Parity registry + comparison for shared/fixtures (DOM-free, no node imports).
// Each TS implementation of a Python-reference algorithm is imported HERE and registered by fixture
// name, so every vitest file sees the same map (a registry inside a test file is per-file).
// Not a medical device.

import { runContactPpg } from './ppg';
import { runHrv } from './hrv';
import { runFaceRppg } from './rppg';
import { runActivityFixture, runPoseFixture } from './pipeline';
import { runBlinkFixture } from './blink';
import { runMobilityFixture } from './mobility';
import { runTremorFixture } from './tremor';
import { BreathingRateEstimator } from './breathing';
import { runAudioEnvelope, runBreathEnv, runLabelEvents } from './audio';

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

const ppgHrv: Implementation = (fx, params) => runHrv(fx.input, params.contact_ppg, params.hrv);

const faceRppg: Implementation = (fx, params) => runFaceRppg(fx.input, params.rppg);

const audioEnvelope: Implementation = (fx, params) => runAudioEnvelope(fx.input, params.audio);

const breathEnv: Implementation = (fx, params) => runBreathEnv(fx.input, params.audio);

const labelEvents: Implementation = (fx, params) => runLabelEvents(fx.input, params.audio);

const mobility: Implementation = (fx, params) => runMobilityFixture(fx.input, params);

const tremor: Implementation = (fx, params) => runTremorFixture(fx.input, params);

const blink: Implementation = (fx, params) => runBlinkFixture(fx.input, params);

const activity: Implementation = (fx, params) => runActivityFixture(fx.input, params);

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
  hrv_clean_30ms: ppgHrv,
  hrv_clean_80ms: ppgHrv,
  hrv_ectopics: ppgHrv,
  hrv_too_many_artifacts: ppgHrv,
  hrv_too_short: ppgHrv,
  rppg_clean_light_72bpm: faceRppg,
  rppg_clean_medium_90bpm: faceRppg,
  rppg_clean_dark_60bpm: faceRppg,
  rppg_motion_burst: faceRppg,
  rppg_lighting_step: faceRppg,
  rppg_face_lost: faceRppg,
  rppg_small_face: faceRppg,
  rppg_skin_filter: faceRppg,
  rppg_disagree_flicker: faceRppg,
  audio_envelope_shaped_noise: audioEnvelope,
  audio_breath_8bpm: breathEnv,
  audio_breath_12bpm: breathEnv,
  audio_breath_18bpm: breathEnv,
  audio_breath_30bpm: breathEnv,
  audio_breath_double_peaks_15bpm: breathEnv,
  audio_breath_double_equal_15bpm: breathEnv,
  audio_breath_talking: breathEnv,
  audio_breath_too_quiet: breathEnv,
  audio_breath_clipping: breathEnv,
  audio_cough_single: labelEvents,
  audio_cough_bout_merge: labelEvents,
  audio_cough_borderline_hysteresis: labelEvents,
  audio_cough_sustained_noise: labelEvents,
  audio_cough_speech_only: labelEvents,
  mobility_chair_stand_8: mobility,
  mobility_chair_stand_12: mobility,
  mobility_chair_stand_15: mobility,
  mobility_chair_stand_dropouts: mobility,
  mobility_chair_stand_low_rate: mobility,
  mobility_sts_5x: mobility,
  mobility_tug: mobility,
  mobility_tug_no_turn: mobility,
  mobility_walk_0p6: mobility,
  mobility_walk_1p0: mobility,
  mobility_walk_1p4: mobility,
  mobility_walk_crossing_3s: mobility,
  mobility_walk_toward_camera: mobility,
  tremor_4hz: tremor,
  tremor_9hz: tremor,
  tremor_null: tremor,
  tremor_low_fps: tremor,
  tremor_moving: tremor,
  blink_8pm: blink,
  blink_15pm: blink,
  blink_25pm: blink,
  blink_long_closures: blink,
  blink_head_turned: blink,
  blink_low_fps: blink,
  activity_session: activity,
  activity_gap_dropout: activity,
};

const TOL_KEYS: Record<string, [string, number]> = {
  value: ['value_bpm', 0.5], quality: ['quality', 0.05], t: ['t_s', 0.05],
  t_onset: ['event_time_s', 0.1], t_confirmed: ['event_time_s', 0.1],
  torso_angle_deg: ['torso_angle_deg', 0.05], motion: ['motion', 0.01],
  bpm: ['bpm', 0.3], rmssd_ms: ['hrv_ms', 0.5], sdnn_ms: ['hrv_ms', 0.5], mean_hr_bpm: ['bpm', 0.3], bpm_ibi: ['bpm', 0.3], bpm_chrom: ['bpm', 0.3], final_ibis_ms: ['ibi_ms', 5], snr: ['snr', 0.05], pi: ['pi', 0.01], envelope: ['env_log10', 1e-4],
};
/** mobility fixtures only (scoped by input_level so cough onsets, HRV durations etc. keep their exact comparison) */
const MOB_TOL_KEYS: Record<string, [string, number]> = {
  quality: ['quality', 0.05], time_s: ['mob_time_s', 0.1], rep_times_s: ['mob_time_s', 0.1], rise: ['mob_time_s', 0.1], walk_out: ['mob_time_s', 0.1], turn_1: ['mob_time_s', 0.1], walk_back: ['mob_time_s', 0.1], turn_2: ['mob_time_s', 0.1], sit: ['mob_time_s', 0.1], t_start: ['mob_time_s', 0.1], duration_s: ['mob_time_s', 0.1],
  speed_m_s: ['speed_m_s', 0.02], distance_m: ['distance_m', 0.1], cadence_spm: ['cadence_spm', 1],
};
/** tremor fixtures only: frequency in Hz; amplitudes compare RELATIVELY (tolerance = amp_rel * |expected|) */
const TREMOR_TOL_KEYS: Record<string, [string, number]> = {
  freq_hz: ['freq_hz', 0.05], amp_pct: ['amp_rel', 0.01], p2p_pct: ['amp_rel', 0.01], rms_pct: ['amp_rel', 0.01], motion_pct: ['amp_rel', 0.01],
  prominence: ['prominence', 0.01],
};
/** blink fixtures only */
const BLINK_TOL_KEYS: Record<string, [string, number]> = {
  t_start: ['blink_t_s', 0.05], t_end: ['blink_t_s', 0.05], t: ['blink_t_s', 0.05], blinks_per_min: ['blink_rate', 0.05], perclos: ['perclos', 0.005],
  ear: ['ear', 0.005], baseline: ['ear', 0.005], coverage: ['quality', 0.01], window_s: ['window_s', 0.1], fps: ['blink_rate', 0.05],
};
/** activity fixtures only: every seconds figure (posture buckets, active / still / absent / longest still) */
const ACTIVITY_TOL_KEYS: Record<string, [string, number]> = Object.fromEntries(
  ['sitting', 'standing', 'walking', 'lying', 'absent', 'active_s', 'still_s', 'absent_s', 'total_s', 'longest_still_s', 'active_min'].map((k) => [k, ['act_time_s', 0.1] as [string, number]]),
);
const TOL_RELATIVE = new Set(['amp_rel']);
const TOL_OTHER = 1e-4; // numeric fields without a named tolerance

/** Same semantics as carelens.fixtures.compare_outputs: numbers within tolerance by field name;
 *  strings/booleans/null (reported, kind, posture, ...), array lengths and key sets exactly. */
export function compareOutputs(fx: Fixture, actual: any, expected: any = fx.expected): string[] {
  const errs: string[] = [];
  const keys = ({ mobility: MOB_TOL_KEYS, tremor: TREMOR_TOL_KEYS, blink: BLINK_TOL_KEYS, activity: ACTIVITY_TOL_KEYS } as Record<string, Record<string, [string, number]>>)[fx.input_level] ?? TOL_KEYS;
  const tolFor = (field: string, expectedValue = 0): number => {
    const e = keys[field];
    if (!e) return TOL_OTHER;
    const v = fx.tolerance[e[0]];
    const t = typeof v === 'number' ? v : e[1];
    return TOL_RELATIVE.has(e[0]) ? t * Math.abs(expectedValue) : t;
  };
  const walk = (e: any, a: any, path: string, field: string): void => {
    if (typeof e === 'number') {
      if (typeof a !== 'number' || !(Math.abs(e - a) <= tolFor(field, e) + 1e-9)) {
        errs.push(`${path}: expected ${e} got ${a} (tol ${tolFor(field, e)})`);
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
