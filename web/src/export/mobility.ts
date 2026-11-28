// Maps a W7 mobility result to a session-export summary row. Wellbeing trends only: numbers and withheld reasons, no
// interpretation, no norms. The entered height is never exported (it only scales the walking speed).
import type { MobilityResult } from '../core/mobility';
import type { SummaryRow } from './schema';

const r3 = (x: number): number => Math.round(x * 1000) / 1000;
const r1 = (x: number): number => Math.round(x * 10) / 10;

export type MobilitySummaryFields = Omit<SummaryRow, 't' | 'wall'>;

export function mobilitySummary(r: MobilityResult): MobilitySummaryFields {
  const row: MobilitySummaryFields = { kind: r.test, reported: r.reported, reason: r.reason.slice(0, 40), quality: r3(Math.min(1, Math.max(0, r.quality))) };
  if (!r.reported) return row;
  switch (r.test) {
    case 'chair_stand_30s':
      row.reps = r.reps ?? 0;
      row.rep_times_s = (r.rep_times_s ?? []).slice(0, 100).map(r3);
      break;
    case 'sit_to_stand_5x':
      row.time_s = r3(r.time_s ?? 0);
      row.rep_times_s = (r.rep_times_s ?? []).slice(0, 100).map(r3);
      break;
    case 'tug':
      row.time_s = r3(r.time_s ?? 0);
      if (r.phases_s) row.phases_s = Object.fromEntries(Object.entries(r.phases_s).map(([k, v]) => [k, r3(v)]));
      break;
    case 'walking_speed':
      row.speed_m_s = r3(r.speed_m_s ?? 0);
      if (r.cadence_spm !== null && r.cadence_spm > 0) row.cadence_spm = r1(r.cadence_spm);
      if (r.duration_s !== null) row.duration_s = r3(r.duration_s);
      break;
  }
  return row;
}
