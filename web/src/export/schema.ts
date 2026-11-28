// Session export schema (shared/session.schema.json) and a hand-written validator (no dependency).
// carelens/websession.py implements the same checks. Wellbeing trends only. Not a medical device.
export const SCHEMA_VERSION = 1; // JSON 1 and 1.0 are the same number; booleans are not numbers (Number.isInteger(true) is false)
export const METRICS = ['hr_fingertip', 'hr_face', 'br_camera', 'br_sound'] as const;
export const SUMMARY_KINDS = ['hrv'] as const;
export const EVENT_KINDS = ['possible_fall', 'inactivity', 'cough'] as const;
export const MODES = ['camera', 'microphone', 'fingertip'] as const;
export const UA_FAMILIES = ['Chrome', 'Edge', 'Firefox', 'Safari', 'Other'] as const;
export type Metric = (typeof METRICS)[number];
export type EventKind = (typeof EVENT_KINDS)[number];
export type Mode = (typeof MODES)[number];
export type UaFamily = (typeof UA_FAMILIES)[number];

export interface EstimateRow {
  t: number; wall: string; metric: Metric; value: number | null; quality: number; reported: boolean; reason: string; span_s?: number;
}
export interface EventRow {
  t: number; wall: string; kind: EventKind; t_onset?: number; t_end?: number; peak_score?: number; trigger?: string;
}
/** Whole-measurement result (W6 HRV). `summaries` is an optional top-level array: absent in older exports. */
export interface SummaryRow {
  t: number; wall: string; kind: (typeof SUMMARY_KINDS)[number]; reported: boolean; reason: string;
  rmssd_ms?: number; sdnn_ms?: number; mean_hr_bpm?: number; n_beats?: number; n_kept?: number; kept_frac?: number; duration_s?: number;
}
export interface SessionExport {
  schema_version: number; app_version: string; params_version: number; started_at: string; tz_offset_min: number;
  device: { user_agent_family: UaFamily }; modes: Mode[]; estimates: EstimateRow[]; events: EventRow[]; summaries?: SummaryRow[]; disclaimer: string; truncated?: boolean;
}

const ISO_OFFSET = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;
const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null && !Array.isArray(x);
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const oneOf = (list: readonly string[], x: unknown): boolean => typeof x === 'string' && list.includes(x);

function noExtra(o: Record<string, unknown>, allowed: readonly string[], where: string, errs: string[]): void {
  for (const k of Object.keys(o)) if (!allowed.includes(k)) errs.push(`${where}: unexpected field "${k}"`);
}
function isoOk(x: unknown): boolean {
  return typeof x === 'string' && ISO_OFFSET.test(x) && !Number.isNaN(Date.parse(x));
}

/** Returns a list of problems (empty = valid). Never throws. */
export function validateSession(x: unknown): string[] {
  const errs: string[] = [];
  if (!isObj(x)) return ['session: must be a JSON object'];
  const top = ['schema_version', 'app_version', 'params_version', 'started_at', 'tz_offset_min', 'device', 'modes', 'estimates', 'events', 'disclaimer'];
  for (const k of top) if (!(k in x)) errs.push(`session: missing "${k}"`);
  noExtra(x, [...top, 'truncated', 'summaries'], 'session', errs);
  if ('truncated' in x && typeof x.truncated !== 'boolean') errs.push('truncated: expected a boolean');
  if ('schema_version' in x && !(Number.isInteger(x.schema_version) && x.schema_version === SCHEMA_VERSION)) errs.push(`schema_version: expected ${SCHEMA_VERSION}, got ${JSON.stringify(x.schema_version)}`);
  if ('app_version' in x && !(typeof x.app_version === 'string' && x.app_version.length >= 1 && x.app_version.length <= 32)) errs.push('app_version: expected a string of 1-32 characters');
  if ('params_version' in x && !(Number.isInteger(x.params_version) && (x.params_version as number) >= 0)) errs.push('params_version: expected a non-negative integer');
  if ('started_at' in x && !isoOk(x.started_at)) errs.push('started_at: expected ISO 8601 with a UTC offset');
  if ('tz_offset_min' in x && !(Number.isInteger(x.tz_offset_min) && Math.abs(x.tz_offset_min as number) <= 840)) errs.push('tz_offset_min: expected an integer within +-840');
  if ('disclaimer' in x && !(typeof x.disclaimer === 'string' && x.disclaimer.length > 0)) errs.push('disclaimer: expected a non-empty string');
  if ('device' in x) {
    if (!isObj(x.device)) errs.push('device: must be an object');
    else {
      noExtra(x.device, ['user_agent_family'], 'device', errs);
      if (!oneOf(UA_FAMILIES, x.device.user_agent_family)) errs.push(`device.user_agent_family: expected one of ${UA_FAMILIES.join('|')}`);
    }
  }
  if ('modes' in x) {
    if (!Array.isArray(x.modes) || !x.modes.every((m) => oneOf(MODES, m)) || new Set(x.modes).size !== x.modes.length) errs.push(`modes: expected unique items from ${MODES.join('|')}`);
  }
  if ('estimates' in x) {
    if (!Array.isArray(x.estimates)) errs.push('estimates: must be an array');
    else x.estimates.forEach((e, i) => estimateErrors(e, `estimates[${i}]`, errs));
  }
  if ('events' in x) {
    if (!Array.isArray(x.events)) errs.push('events: must be an array');
    else x.events.forEach((e, i) => eventErrors(e, `events[${i}]`, errs));
  }
  if ('summaries' in x) {
    if (!Array.isArray(x.summaries)) errs.push('summaries: must be an array');
    else x.summaries.forEach((e, i) => summaryErrors(e, `summaries[${i}]`, errs));
  }
  return errs.slice(0, 50);
}

function summaryErrors(e: unknown, w: string, errs: string[]): void {
  if (!isObj(e)) { errs.push(`${w}: must be an object`); return; }
  const req = ['t', 'wall', 'kind', 'reported', 'reason'];
  for (const k of req) if (!(k in e)) errs.push(`${w}: missing "${k}"`);
  noExtra(e, [...req, 'rmssd_ms', 'sdnn_ms', 'mean_hr_bpm', 'n_beats', 'n_kept', 'kept_frac', 'duration_s'], w, errs);
  if ('t' in e && !(isNum(e.t) && e.t >= 0)) errs.push(`${w}.t: expected a number >= 0`);
  if ('wall' in e && !isoOk(e.wall)) errs.push(`${w}.wall: expected ISO 8601 with a UTC offset`);
  if ('kind' in e && !oneOf(SUMMARY_KINDS, e.kind)) errs.push(`${w}.kind: expected one of ${SUMMARY_KINDS.join('|')}`);
  if ('reported' in e && typeof e.reported !== 'boolean') errs.push(`${w}.reported: expected a boolean`);
  if ('reason' in e && !(typeof e.reason === 'string' && e.reason.length <= 40)) errs.push(`${w}.reason: expected a string up to 40 characters`);
  for (const k of ['rmssd_ms', 'sdnn_ms', 'duration_s'] as const) if (k in e && !(isNum(e[k]) && (e[k] as number) >= 0)) errs.push(`${w}.${k}: expected a number >= 0`);
  if ('mean_hr_bpm' in e && !(isNum(e.mean_hr_bpm) && e.mean_hr_bpm > 0)) errs.push(`${w}.mean_hr_bpm: expected a number > 0`);
  for (const k of ['n_beats', 'n_kept'] as const) if (k in e && !(Number.isInteger(e[k]) && (e[k] as number) >= 0)) errs.push(`${w}.${k}: expected an integer >= 0`);
  if ('kept_frac' in e && !(isNum(e.kept_frac) && e.kept_frac >= 0 && e.kept_frac <= 1)) errs.push(`${w}.kept_frac: expected a number in [0, 1]`);
  if (e.reported === true && !(isNum(e.rmssd_ms) && isNum(e.sdnn_ms))) errs.push(`${w}: reported rows need rmssd_ms and sdnn_ms`);
}

function estimateErrors(e: unknown, w: string, errs: string[]): void {
  if (!isObj(e)) { errs.push(`${w}: must be an object`); return; }
  const req = ['t', 'wall', 'metric', 'value', 'quality', 'reported', 'reason'];
  for (const k of req) if (!(k in e)) errs.push(`${w}: missing "${k}"`);
  noExtra(e, [...req, 'span_s'], w, errs);
  if ('t' in e && !(isNum(e.t) && e.t >= 0)) errs.push(`${w}.t: expected a number >= 0`);
  if ('wall' in e && !isoOk(e.wall)) errs.push(`${w}.wall: expected ISO 8601 with a UTC offset`);
  if ('metric' in e && !oneOf(METRICS, e.metric)) errs.push(`${w}.metric: expected one of ${METRICS.join('|')}`);
  if ('value' in e && !(e.value === null || isNum(e.value))) errs.push(`${w}.value: expected a number or null`);
  if ('quality' in e && !(isNum(e.quality) && e.quality >= 0 && e.quality <= 1)) errs.push(`${w}.quality: expected a number in [0, 1]`);
  if ('reported' in e && typeof e.reported !== 'boolean') errs.push(`${w}.reported: expected a boolean`);
  if (e.reported === true && e.value === null) errs.push(`${w}: reported rows need a value`);
  if ('reason' in e && !(typeof e.reason === 'string' && e.reason.length <= 40)) errs.push(`${w}.reason: expected a string up to 40 characters`);
  if ('span_s' in e && !(isNum(e.span_s) && e.span_s > 0)) errs.push(`${w}.span_s: expected a number > 0`);
}

function eventErrors(e: unknown, w: string, errs: string[]): void {
  if (!isObj(e)) { errs.push(`${w}: must be an object`); return; }
  const req = ['t', 'wall', 'kind'];
  for (const k of req) if (!(k in e)) errs.push(`${w}: missing "${k}"`);
  noExtra(e, [...req, 't_onset', 't_end', 'peak_score', 'trigger'], w, errs);
  if ('t' in e && !(isNum(e.t) && e.t >= 0)) errs.push(`${w}.t: expected a number >= 0`);
  if ('wall' in e && !isoOk(e.wall)) errs.push(`${w}.wall: expected ISO 8601 with a UTC offset`);
  if ('kind' in e && !oneOf(EVENT_KINDS, e.kind)) errs.push(`${w}.kind: expected one of ${EVENT_KINDS.join('|')}`);
  for (const k of ['t_onset', 't_end'] as const) if (k in e && !(isNum(e[k]) && (e[k] as number) >= 0)) errs.push(`${w}.${k}: expected a number >= 0`);
  if ('peak_score' in e && !(isNum(e.peak_score) && e.peak_score >= 0 && e.peak_score <= 1)) errs.push(`${w}.peak_score: expected a number in [0, 1]`);
  if ('trigger' in e && !(typeof e.trigger === 'string' && e.trigger.length <= 40)) errs.push(`${w}.trigger: expected a string up to 40 characters`);
}
