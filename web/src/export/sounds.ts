// Maps the W10 sound results to session-export summary rows. Wellbeing trends only: counts, minutes and a withheld reason,
// no interpretation. Nothing about "sleep quality" or any condition.
import type { LabelView } from '../core/audio';
import type { PauseView } from '../core/sleepSounds';
import type { MobilitySummaryFields } from './mobility';
import { MAX_DURATIONS, MAX_PER_HOUR } from './schema';

const r3 = (x: number): number => Math.round(x * 1000) / 1000;

export function snoringSummary(v: LabelView): MobilitySummaryFields {
  return { kind: 'snoring', reported: true, reason: 'ok', count: v.count, minutes: r3(v.total_s / 60), per_hour: v.per_hour.slice(0, MAX_PER_HOUR) };
}

export function talkingSummary(v: LabelView): MobilitySummaryFields {
  return { kind: 'talking', reported: true, reason: 'ok', minutes: r3(v.total_s / 60), per_hour: v.per_hour_s.slice(0, MAX_PER_HOUR).map((s) => r3(s / 60)) };
}

/** `reported` once some regular breathing sounds were heard; the per-hour rate is only present once 30 min of them were. */
export function pausesSummary(v: PauseView): MobilitySummaryFields {
  const row: MobilitySummaryFields = { kind: 'pauses', reported: v.regular_s > 0, reason: v.regular_s > 0 ? 'ok' : 'no_regular_breathing', count: v.count, regular_min: r3(v.regular_s / 60) };
  if (v.pauses.length) row.durations_s = v.pauses.slice(-MAX_DURATIONS).map((p) => r3(p.duration_s));
  if (v.rate_per_h !== null) row.rate_per_h = r3(v.rate_per_h);
  return row;
}
