// Keeps ONE exported HRV summary per contact epoch (a continuous fingertip measurement): created at the first evaluated
// result (>= min_duration_s of contact) and updated in place while the measurement continues. Wellbeing trends only.
import type { HrvResult } from '../core/hrv';
import type { SessionRecorder } from './recorder';
import type { SummaryRow } from './schema';

export class HrvRecorder {
  private epoch = -1;
  private row: SummaryRow | null = null;
  private bestDur = -1; // duration of the reported result stored for this epoch (-1: none yet)
  private lastR: HrvResult | null = null;

  constructor(private readonly rec: SessionRecorder, private readonly minDurationS: number) {}

  /** Once a result has been reported it is only replaced by another reported one that is at least as long. */
  update(epoch: number, r: HrvResult): void {
    if (this.rec.finished || r === this.lastR || r.duration_s < this.minDurationS) return; // frozen segment: same object, nothing changed
    this.lastR = r;
    if (epoch !== this.epoch) { this.epoch = epoch; this.row = null; this.bestDur = -1; }
    if (this.row && this.bestDur >= 0 && !(r.reported && r.duration_s >= this.bestDur)) return;
    const s = { kind: 'hrv' as const, ...r };
    if (this.row) this.rec.updateSummary(this.row, s);
    else this.row = this.rec.addSummary(s);
    if (r.reported) this.bestDur = r.duration_s;
  }
}
