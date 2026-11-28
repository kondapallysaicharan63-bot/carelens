// Session check-in log (DOM-free). Wellbeing wording only: a suggestion to check in, never an emergency or diagnosis claim.
// Events live in memory for this page only; nothing is sent anywhere and no sound or notification is raised.
import type { PoseEvent } from '../core/events';

export const FALL_TEXT = 'Possible fall — check-in suggested';
export const INACTIVITY_TEXT = 'Long time without movement — check-in suggested';
export const DISTRESS_TEXT = 'Possible distress sound — check-in suggested';
export const FALL_CUE_TEXT = 'Possible fall — with a sound cue — check-in suggested';

export type LogKind = PoseEvent['kind'] | 'distress';

export interface LogEntry {
  id: number;
  kind: LogKind;
  text: string;
  /** wall-clock time the suggestion was raised (ms since epoch) */
  at: number;
  acknowledged: boolean;
  /** a possible fall that a thump / scream / glass sound corroborates (W10) */
  soundCue: boolean;
  /** distress sounds: the YAMNet label that raised it (not shown in the banner text) */
  label?: string;
  /** a distress sound that corroborates a possible fall is shown as that fall's check-in, not as its own: the id of that fall entry */
  mergedInto?: number;
}

export const eventText = (kind: LogKind, soundCue = false): string =>
  kind === 'fall' ? (soundCue ? FALL_CUE_TEXT : FALL_TEXT) : kind === 'distress' ? DISTRESS_TEXT : INACTIVITY_TEXT;

export class EventLog {
  readonly entries: LogEntry[] = [];
  private nextId = 1;

  add(e: { kind: LogKind; label?: string }, at: number): LogEntry {
    const entry: LogEntry = { id: this.nextId++, kind: e.kind, text: eventText(e.kind), at, acknowledged: false, soundCue: false, ...(e.label ? { label: e.label } : {}) };
    this.entries.push(entry);
    return entry;
  }

  /** Acknowledging a check-in also acknowledges the sounds merged into it. */
  acknowledge(id: number): boolean {
    const e = this.entries.find((x) => x.id === id);
    if (!e || e.acknowledged) return false;
    e.acknowledged = true;
    for (const m of this.entries) if (m.mergedInto === id) m.acknowledged = true;
    return true;
  }

  /** Entry `id` (a distress sound) becomes part of the possible fall `intoId`: it no longer shows as a check-in of its own. */
  mergeInto(id: number, intoId: number): boolean {
    const e = this.entries.find((x) => x.id === id), f = this.entries.find((x) => x.id === intoId);
    if (!e || !f || f.kind !== 'fall' || e.kind !== 'distress' || e.mergedInto !== undefined) return false;
    e.mergedInto = intoId;
    if (f.acknowledged) e.acknowledged = true;
    return true;
  }

  /** the check-ins that show (merged sounds do not) */
  get visible(): LogEntry[] { return this.entries.filter((e) => e.mergedInto === undefined); }

  /** a sound corroborates this possible fall: its text gains "— with a sound cue". False when unknown, not a fall or already marked. */
  markSoundCue(id: number): boolean {
    const e = this.entries.find((x) => x.id === id);
    if (!e || e.kind !== 'fall' || e.soundCue) return false;
    e.soundCue = true;
    e.text = eventText('fall', true);
    return true;
  }

  /** the newest suggestion nobody has acknowledged yet (drives the banner) */
  get pending(): LogEntry | null {
    for (let i = this.entries.length - 1; i >= 0; i--) if (!this.entries[i]!.acknowledged && this.entries[i]!.mergedInto === undefined) return this.entries[i]!;
    return null;
  }

  get pendingCount(): number {
    return this.entries.filter((e) => !e.acknowledged && e.mergedInto === undefined).length;
  }
}
