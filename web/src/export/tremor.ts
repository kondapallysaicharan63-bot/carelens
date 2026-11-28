// Maps a W8 rhythmic-hand-movement result to a session-export summary row. Wellbeing trend only: numbers or a neutral
// withheld reason, no interpretation.
import type { TremorResult } from '../core/tremor';
import type { MobilitySummaryFields } from './mobility';

const r3 = (x: number): number => Math.round(x * 1000) / 1000;

export function tremorSummary(r: TremorResult): MobilitySummaryFields {
  const row: MobilitySummaryFields = { kind: 'hand_tremor', reported: r.reported, reason: r.reason, duration_s: r3(Math.max(0, r.duration_s)) };
  if (r.reported && r.freq_hz !== null && r.amp_pct !== null) {
    row.freq_hz = r3(r.freq_hz);
    row.amp_pct = r3(r.amp_pct);
    if (r.p2p_pct !== null) row.p2p_pct = r3(r.p2p_pct);
  }
  return row;
}
