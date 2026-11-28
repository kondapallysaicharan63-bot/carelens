// Session check-in log (DOM-free). Wellbeing wording only: a suggestion to check in, never an emergency or diagnosis claim.
// Events live in memory for this page only; nothing is sent anywhere and no sound or notification is raised.
import type { PoseEvent } from '../core/events';

export const FALL_TEXT = 'Possible fall — check-in suggested';
export const INACTIVITY_TEXT = 'Long time without movement — check-in suggested';

export interface LogEntry {
  id: number;
  kind: PoseEvent['kind'];
  text: string;
  /** wall-clock time the suggestion was raised (ms since epoch) */
  at: number;
  acknowledged: boolean;
}

export const eventText = (kind: PoseEvent['kind']): string => (kind === 'fall' ? FALL_TEXT : INACTIVITY_TEXT);

export class EventLog {
  readonly entries: LogEntry[] = [];
  private nextId = 1;

  add(e: Pick<PoseEvent, 'kind'>, at: number): LogEntry {
    const entry: LogEntry = { id: this.nextId++, kind: e.kind, text: eventText(e.kind), at, acknowledged: false };
    this.entries.push(entry);
    return entry;
  }

  acknowledge(id: number): boolean {
    const e = this.entries.find((x) => x.id === id);
    if (!e || e.acknowledged) return false;
    e.acknowledged = true;
    return true;
  }

  /** the newest suggestion nobody has acknowledged yet (drives the banner) */
  get pending(): LogEntry | null {
    for (let i = this.entries.length - 1; i >= 0; i--) if (!this.entries[i]!.acknowledged) return this.entries[i]!;
    return null;
  }

  get pendingCount(): number {
    return this.entries.filter((e) => !e.acknowledged).length;
  }
}
