// Turns the W10 sound results into check-in suggestions, export rows and the fall "sound cue". DOM-free (the log is shared with the
// banner). Wellbeing wording only: "Possible distress sound — check-in suggested"; it never says an emergency or a condition was detected.
//
// One check-in per event: a thump / scream / glass sound near a possible fall (see soundCue) is not a second check-in; it upgrades the
// fall's ("— with a sound cue"), whichever came first. Audio restarts (a new microphone) start fresh trackers: reset() banks the totals
// so counts, minutes, per-hour buckets and regular-breathing time carry on within one capture.
import type { AudioEventsParams, LabelView } from '../core/audio';
import type { PoseEvent } from '../core/events';
import { soundCue, type PauseView, type SoundEvent } from '../core/sleepSounds';
import type { SoundsView } from '../capture/audioSession';
import { pausesSummary, snoringSummary, talkingSummary } from '../export/sounds';
import type { SessionRecorder } from '../export/recorder';
import type { EventRow } from '../export/schema';
import type { EventLog } from './eventLog';

interface FallRef { entryId: number; row: EventRow | null; tOnset: number; tConfirmed: number; cued: boolean }
interface Tracked extends SoundEvent { entryId?: number; row: EventRow | null; merged: boolean }
interface CurRef { tStart: number; sound: Tracked }

const KEEP_FALLS = 20;
const KEEP_SOUNDS = 50;
/** how often the session summary rows are refreshed while sounds are being heard (s of session time) */
const SUMMARY_EVERY_S = 5;
const MAX_PAUSES = 500;

const addArrays = (a: number[], b: number[]): number[] => Array.from({ length: Math.max(a.length, b.length) }, (_, i) => (a[i] ?? 0) + (b[i] ?? 0));

/** `base` (earlier audio sessions of this capture) + `cur` (the running one): counts, total time and per-hour buckets add up. */
export function mergeLabelViews(base: LabelView | null, cur: LabelView): LabelView {
  if (!base) return cur;
  return { ...cur, count: base.count + cur.count, total_s: base.total_s + cur.total_s, per_hour: addArrays(base.per_hour, cur.per_hour), per_hour_s: addArrays(base.per_hour_s, cur.per_hour_s) };
}

export function mergePauseViews(base: PauseView | null, cur: PauseView, minValidRegularS: number): PauseView {
  if (!base) return cur;
  const count = base.count + cur.count, regular = base.regular_s + cur.regular_s;
  return {
    count, regular_s: regular, no_sound_gaps: base.no_sound_gaps + cur.no_sound_gaps,
    pauses: [...base.pauses, ...cur.pauses].slice(-MAX_PAUSES),
    rate_per_h: regular >= minValidRegularS ? count / (regular / 3600) : null,
  };
}

export class SoundCheckins {
  private falls: FallRef[] = [];
  private sounds: Tracked[] = [];
  private readonly seen = new Map<string, number>();
  private readonly cur = new Map<string, CurRef>();
  private pauseSeen = 0;
  private lastSummaryS = -Infinity;
  private last: SoundsView | null = null; // merged view
  private base: SoundsView | null = null; // totals of the earlier audio sessions

  constructor(
    private readonly o: { log: EventLog; refresh: () => void; rec: SessionRecorder; ev: Pick<AudioEventsParams, 'cue_labels' | 'corroborate_s'>; minValidRegularS?: number; now?: () => number },
  ) {}

  private nowMs(): number { return (this.o.now ?? Date.now)(); }

  /** A new audio session starts (device switch, restart): its trackers begin at zero. Bank the totals, forget the running sounds. */
  reset(): void {
    if (this.last) this.base = this.last;
    this.cur.clear();
    this.seen.clear();
    this.pauseSeen = 0;
  }

  private merge(v: SoundsView): SoundsView {
    const b = this.base, min = this.o.minValidRegularS ?? 1800;
    if (!b) return v;
    return { ...v, snoring: mergeLabelViews(b.snoring, v.snoring), talking: mergeLabelViews(b.talking, v.talking), pauses: mergePauseViews(b.pauses, v.pauses, min) };
  }

  private fallInterval(f: FallRef): { t_onset: number; t_confirmed: number } { return { t_onset: f.tOnset, t_confirmed: f.tConfirmed }; }

  /** A possible fall was raised (`entryId` = its check-in log entry, `row` = its export row). Times are session seconds. */
  registerFall(e: Pick<PoseEvent, 'tOnset' | 'tConfirmed'>, entryId: number, row: EventRow | null): void {
    const tConfirmed = this.o.rec.now();
    this.falls.push({ entryId, row, tOnset: tConfirmed - Math.max(0, e.tConfirmed - e.tOnset), tConfirmed, cued: false });
    if (this.falls.length > KEEP_FALLS) this.falls.shift();
    this.recheck();
  }

  /** Every fall with a corroborating sound gets the cue, and those sounds stop being check-ins of their own. */
  private recheck(): void {
    let changed = false;
    for (const f of this.falls) {
      const match = this.sounds.filter((s) => soundCue(this.fallInterval(f), [s], this.o.ev));
      if (!match.length) continue;
      if (!f.cued) {
        f.cued = true;
        if (this.o.log.markSoundCue(f.entryId)) changed = true;
        if (f.row) this.o.rec.updateEvent(f.row, { sound_cue: true });
      }
      for (const s of match) {
        if (s.merged) continue;
        s.merged = true;
        if (s.entryId !== undefined && this.o.log.mergeInto(s.entryId, f.entryId)) changed = true;
        if (s.row) this.o.rec.updateEvent(s.row, { merged_into_fall: true });
      }
    }
    if (changed) this.o.refresh();
  }

  /** Call with every SoundsView of the running audio session; returns the view with earlier sessions' totals added (for the cards). */
  onSounds(raw: SoundsView): SoundsView {
    const rec = this.o.rec, nowS = rec.now();
    let added = false;
    for (const d of raw.distress) {
      const last = d.view.last;
      if (!last) continue;
      const s0 = nowS - (raw.t - last.t_start), dur = last.t_end - last.t_start;
      if (d.view.count > (this.seen.get(d.label) ?? 0)) {
        const sound: Tracked = { label: d.label, t_start: s0, t_end: s0 + dur, row: null, merged: false };
        const corroborates = this.falls.some((f) => soundCue(this.fallInterval(f), [sound], this.o.ev));
        sound.row = rec.addEvent({ kind: 'distress_sound', agoS: raw.t - last.t_start, durationS: dur, peak_score: last.peak_score, trigger: d.label, ...(corroborates ? { merged_into_fall: true } : {}) });
        if (corroborates) sound.merged = true;
        else { sound.entryId = this.o.log.add({ kind: 'distress', label: d.label }, this.nowMs()).id; added = true; }
        this.cur.set(d.label, { tStart: last.t_start, sound });
        this.sounds.push(sound);
        if (this.sounds.length > KEEP_SOUNDS) this.sounds.shift();
      } else {
        const c = this.cur.get(d.label);
        if (c && c.tStart === last.t_start) {   // the tracker extended or merged the same sound: keep the stored copy equal
          c.sound.t_end = c.sound.t_start + dur;
          if (c.sound.row) rec.updateEvent(c.sound.row, { t_end: (c.sound.row.t_onset ?? c.sound.row.t) + dur, peak_score: last.peak_score });
        }
      }
      this.seen.set(d.label, d.view.count);
    }
    if (added) this.o.refresh();
    this.recheck();   // a sound that just corroborated a fall adds the cue (and may merge a sound shown earlier)
    // possible pauses: one export event per new pause (the tracker only ever adds)
    if (raw.pauses.count > this.pauseSeen) {
      const p = raw.pauses.pauses[raw.pauses.pauses.length - 1];
      if (p) rec.addEvent({ kind: 'possible_pause', agoS: raw.t - p.t_start, durationS: p.duration_s });
      this.pauseSeen = raw.pauses.count;
    }
    const m = this.merge(raw);
    this.last = m;
    if (nowS - this.lastSummaryS >= SUMMARY_EVERY_S || added) this.writeSummaries(nowS);
    return m;
  }

  private writeSummaries(nowS: number): void {
    const v = this.last;
    if (!v) return;
    this.lastSummaryS = nowS;
    for (const row of [snoringSummary(v.snoring), pausesSummary(v.pauses), talkingSummary(v.talking)]) this.o.rec.upsertSummary(row);
  }

  /** The session ends: store the final summaries. */
  finish(): void { if (this.last && !this.o.rec.finished) this.writeSummaries(this.o.rec.now()); }

  /** One line for the end of a session ("Snoring sounds: 4 (3.2 min) · …"); null when no sound results arrived. */
  summaryText(): string | null {
    const v = this.last;
    if (!v) return null;
    const m = (s: number): string => (s / 60).toFixed(1);
    const rate = v.pauses.rate_per_h === null ? 'per-hour figure needs 30 min of regular breathing sounds' : `${v.pauses.rate_per_h.toFixed(1)} per hour`;
    return `Snoring sounds: ${v.snoring.count} (${m(v.snoring.total_s)} min) · Possible pauses in breathing sounds: ${v.pauses.count} (${rate}) · Talking time: ${m(v.talking.total_s)} min`;
  }
}
