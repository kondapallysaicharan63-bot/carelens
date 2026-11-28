// In-memory record of one session's estimates and check-in events, and its JSON / CSV export.
// No media, no device ids or labels. Nothing is sent anywhere: the page only offers file downloads.
// Wellbeing trends only. Not a medical device.
import params from '../../../shared/params.json';
import pkg from '../../package.json';
import { DISCLAIMER } from '../ui/consent';
import {
  SCHEMA_VERSION, type EstimateRow, type EventRow, type Metric, type Mode, type SessionExport, type UaFamily,
} from './schema';

export const APP_VERSION: string = pkg.version;
/** Row caps. 90 000 = 25 h at one estimate per second per metric; check-in events are rare, 10 000 is generous.
 *  When any cap is hit later rows are dropped, `truncated` is set and the UI says so. */
export const MAX_PER_METRIC = 90000;
export const MAX_EVENTS = 10000;
export const LIMIT_TEXT = 'Recording limit reached';

export function uaFamily(ua: string): UaFamily {
  if (/Edg(e|A|iOS)?\//.test(ua)) return 'Edge';
  if (/Firefox\/|FxiOS\//.test(ua)) return 'Firefox';
  if (/Chrome\/|Chromium\/|CriOS\//.test(ua)) return 'Chrome';
  if (/Safari\//.test(ua)) return 'Safari';
  return 'Other';
}

const p2 = (n: number, w = 2): string => String(Math.trunc(Math.abs(n))).padStart(w, '0');

/** ISO 8601 with the given offset (minutes east of UTC), e.g. 2026-11-20T10:00:00.250-08:00. */
export function isoWithOffset(ms: number, offsetMin: number): string {
  const d = new Date(ms + offsetMin * 60000);
  const sign = offsetMin < 0 ? '-' : '+';
  return `${d.getUTCFullYear()}-${p2(d.getUTCMonth() + 1)}-${p2(d.getUTCDate())}T${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}:${p2(d.getUTCSeconds())}.${p2(d.getUTCMilliseconds(), 3)}${sign}${p2(offsetMin / 60)}:${p2(offsetMin % 60)}`;
}

const r3 = (x: number): number => Math.round(x * 1000) / 1000;
const median = (v: number[]): number | null => {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m]! : 0.5 * (s[m - 1]! + s[m]!);
};

export interface SessionSummary {
  id: string;
  started_at: string;
  duration_s: number;
  /** median of REPORTED values per metric (null when none) */
  medians: Partial<Record<Metric, number | null>>;
  counts: { possible_fall: number; inactivity: number; cough: number };
}

export class SessionRecorder {
  readonly startedMs: number;
  readonly tzOffsetMin: number;
  readonly estimates: EstimateRow[] = [];
  readonly events: EventRow[] = [];
  private endT: number | null = null;
  private readonly perMetric = new Map<Metric, number>();
  /** true once a row cap was hit: later rows were dropped */
  truncated = false;

  constructor(readonly modes: Mode[], private readonly clock: () => number = () => Date.now(), private readonly ua = typeof navigator === 'undefined' ? '' : navigator.userAgent) {
    this.startedMs = clock();
    this.tzOffsetMin = -new Date(this.startedMs).getTimezoneOffset();
  }

  /** seconds since the session started */
  now(): number { return Math.max(0, (this.clock() - this.startedMs) / 1000); }
  private wall(t: number): string { return isoWithOffset(this.startedMs + t * 1000, this.tzOffsetMin); }

  addEstimate(e: { metric: Metric; value: number | null; quality: number; reported: boolean; reason: string; span_s?: number | null }): void {
    const have = this.perMetric.get(e.metric) ?? 0;
    if (have >= MAX_PER_METRIC) { this.truncated = true; return; }
    this.perMetric.set(e.metric, have + 1);
    const t = r3(this.now());
    const q = Number.isFinite(e.quality) ? Math.min(1, Math.max(0, e.quality)) : 0;
    const row: EstimateRow = { t, wall: this.wall(t), metric: e.metric, value: e.value !== null && Number.isFinite(e.value) ? r3(e.value) : null, quality: r3(q), reported: e.reported, reason: e.reason.slice(0, 40) };
    if (e.span_s !== undefined && e.span_s !== null && e.span_s > 0) row.span_s = r3(e.span_s);
    this.estimates.push(row);
  }

  /** `agoS`: the event began that many seconds before it was raised. */
  addEvent(e: { kind: EventRow['kind']; agoS?: number; peak_score?: number; trigger?: string; durationS?: number }): EventRow | null {
    if (this.events.length >= MAX_EVENTS) { this.truncated = true; return null; }
    const t = r3(this.now());
    const row: EventRow = { t, wall: this.wall(t), kind: e.kind };
    if (e.agoS !== undefined) row.t_onset = r3(Math.max(0, t - Math.max(0, e.agoS)));
    if (e.durationS !== undefined && row.t_onset !== undefined) row.t_end = r3(row.t_onset + Math.max(0, e.durationS));
    if (e.peak_score !== undefined) row.peak_score = r3(Math.min(1, Math.max(0, e.peak_score)));
    if (e.trigger) row.trigger = e.trigger.slice(0, 40);
    this.events.push(row);
    return row;
  }

  /** Update a stored event (a cough that is still going on / merged with the next sound). */
  updateEvent(row: EventRow, patch: { t_end?: number; peak_score?: number }): void {
    if (patch.t_end !== undefined) row.t_end = r3(Math.max(0, patch.t_end));
    if (patch.peak_score !== undefined) row.peak_score = r3(Math.min(1, Math.max(0, patch.peak_score)));
  }

  finish(): void { if (this.endT === null) this.endT = this.now(); }
  get finished(): boolean { return this.endT !== null; }
  get isEmpty(): boolean { return this.estimates.length === 0 && this.events.length === 0; }

  toJSON(): SessionExport {
    return {
      schema_version: SCHEMA_VERSION, app_version: APP_VERSION, params_version: params.version, started_at: isoWithOffset(this.startedMs, this.tzOffsetMin),
      tz_offset_min: this.tzOffsetMin, device: { user_agent_family: uaFamily(this.ua) }, modes: [...this.modes],
      estimates: this.estimates.map((e) => ({ ...e })), events: this.events.map((e) => ({ ...e })), disclaimer: DISCLAIMER, truncated: this.truncated,
    };
  }

  toCsv(): string { return estimatesCsv(this.estimates); }

  summary(): SessionSummary {
    const medians: SessionSummary['medians'] = {};
    for (const m of new Set(this.estimates.map((e) => e.metric))) {
      const med = median(this.estimates.filter((e) => e.metric === m && e.reported && e.value !== null).map((e) => e.value as number));
      medians[m] = med === null ? null : Math.round(med * 10) / 10;
    }
    const last = Math.max(0, ...this.estimates.map((e) => e.t), ...this.events.map((e) => e.t));
    const count = (k: EventRow['kind']): number => this.events.filter((e) => e.kind === k).length;
    return {
      id: isoWithOffset(this.startedMs, 0), started_at: isoWithOffset(this.startedMs, this.tzOffsetMin),
      duration_s: Math.round(this.endT ?? last), medians, counts: { possible_fall: count('possible_fall'), inactivity: count('inactivity'), cough: count('cough') },
    };
  }
}

export const CSV_HEADER = 't,wall,metric,value,quality,reported,reason,span_s';
export function estimatesCsv(rows: readonly EstimateRow[]): string {
  const lines = rows.map((e) => [e.t, e.wall, e.metric, e.value ?? '', e.quality, e.reported ? 'true' : 'false', e.reason, e.span_s ?? ''].join(','));
  return [CSV_HEADER, ...lines].join('\n') + '\n';
}
