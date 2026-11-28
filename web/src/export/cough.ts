// Keeps the exported cough events equal to the tracker's: a cough is stored as soon as it is seen and UPDATED (end time,
// peak) while the tracker extends or merges it, until the session ends. Wellbeing trends only. Not a medical device.
import type { LabelEvent } from '../core/audio';
import type { EventRow } from './schema';
import type { SessionRecorder } from './recorder';

export class CoughRecorder {
  private seen = 0;
  private cur: { row: EventRow; tStart: number } | null = null;
  private done = false;

  constructor(private readonly rec: SessionRecorder) {}

  /** `r` is LabelEventTracker.view() plus the stream time of the view. */
  update(r: { count: number; last: LabelEvent | null; t: number }): void {
    if (this.done || this.rec.finished) return; // the session ended: the stored events are final
    if (r.count < this.seen) this.cur = null; // a new audio session restarted the count
    const last = r.last;
    if (last) {
      const dur = last.t_end - last.t_start;
      if (r.count > this.seen) {
        const row = this.rec.addEvent({ kind: 'cough', agoS: r.t - last.t_start, durationS: dur, peak_score: last.peak_score });
        this.cur = row ? { row, tStart: last.t_start } : null;
      } else if (this.cur && this.cur.tStart === last.t_start) {
        const onset = this.cur.row.t_onset ?? this.cur.row.t;
        this.rec.updateEvent(this.cur.row, { t_end: onset + dur, peak_score: last.peak_score });
      }
    }
    this.seen = r.count;
  }

  /** the session ended: the stored events are final */
  finish(): void { this.done = true; }
}
