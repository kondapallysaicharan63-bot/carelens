// Maps the W9 blink / eyes-closed result and the activity summary to session-export summary rows. Wellbeing trends only:
// numbers or a withheld reason, no interpretation, no norms.
import type { ActivitySummaryResult } from '../core/events';
import type { BlinkState } from '../core/blink';
import type { MobilitySummaryFields } from './mobility';

const r1 = (x: number): number => Math.round(x * 10) / 10;
const r3 = (x: number): number => Math.round(x * 1000) / 1000;
const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

export function blinkSummary(s: Pick<BlinkState, 'reported' | 'reason' | 'blinks_per_min' | 'perclos' | 'long_closures' | 'window_s' | 'coverage'>): MobilitySummaryFields {
  const row: MobilitySummaryFields = { kind: 'blink', reported: s.reported, reason: s.reason.slice(0, 40), duration_s: r3(Math.max(0, s.window_s)) };
  if (s.reported) {
    row.blinks_per_min = r1(Math.max(0, s.blinks_per_min));
    row.eyes_closed_pct = r1(100 * clamp01(s.perclos));
    row.long_closures = Math.max(0, Math.trunc(s.long_closures));
    row.quality = r3(clamp01(s.coverage));
  }
  return row;
}

export function activitySummaryRow(r: ActivitySummaryResult): MobilitySummaryFields {
  const row: MobilitySummaryFields = { kind: 'activity', reported: r.reported, reason: r.reason, duration_s: r3(Math.max(0, r.total_s)) };
  if (r.reported) {
    row.active_min = r3(r.active_s / 60);
    row.still_min = r3(r.still_s / 60);
    row.absent_min = r3(r.absent_s / 60);
    row.longest_still_min = r3(r.longest_still_s / 60);
  }
  return row;
}
