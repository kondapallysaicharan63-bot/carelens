import { describe, expect, it, vi } from 'vitest';
import params from '../../shared/params.json';
import type { SoundsView } from '../src/capture/audioSession';
import { WAKE_TEXT, WakeLockKeeper, PLUG_TEXT } from '../src/capture/wakeLock';
import type { AudioEventsParams, LabelView } from '../src/core/audio';
import type { PauseView } from '../src/core/sleepSounds';
import { SessionRecorder } from '../src/export/recorder';
import { pausesSummary, snoringSummary, talkingSummary } from '../src/export/sounds';
import { validateSession } from '../src/export/schema';
import { DISTRESS_TEXT, EventLog, FALL_CUE_TEXT, FALL_TEXT, eventText } from '../src/ui/eventLog';
import { NightChoices } from '../src/ui/nightMode';
import { PAUSES_NO_MODEL, SOUNDS_NO_MODEL, PAUSES_NOTE, PAUSES_TITLE, PAUSES_WAIT, SNORING_NOTE, TALKING_NOTE, TALKING_TITLE, pausesDurations, pausesRate, pausesText, snoringHours, snoringText, talkingHours, talkingText } from '../src/ui/sleepCards';
import { SoundCheckins } from '../src/ui/soundCheckins';
import { NIGHT_HINT } from '../src/ui/nightMode';

const EV = params.audio_events as AudioEventsParams;
const lv = (o: Partial<LabelView> = {}): LabelView => ({ count: 0, last: null, ticks: [], total_s: 0, per_hour: [], per_hour_s: [], ...o });
const pv = (o: Partial<PauseView> = {}): PauseView => ({ count: 0, pauses: [], regular_s: 0, rate_per_h: null, no_sound_gaps: 0, ...o });

function setup() {
  const clock = { ms: 1_700_000_000_000 };
  const rec = new SessionRecorder(['microphone', 'camera'], () => clock.ms, 'Mozilla/5.0 Chrome/130');
  const log = new EventLog();
  const refresh = vi.fn();
  const sc = new SoundCheckins({ log, refresh, rec, ev: EV, now: () => clock.ms });
  /** a Thump / Screaming view at stream time t: one event whose onset was `ago` s earlier */
  const view = (t: number, label: string, ago = 1, peak = 0.9, count = 1): SoundsView => ({
    t, snoring: lv(), talking: lv(), pauses: pv(),
    distress: ['Screaming', 'Crying, sobbing', 'Glass', 'Thump, thud'].map((l) => ({ label: l, view: l === label ? lv({ count, last: { t_start: t - ago, t_end: t - ago + 1, peak_score: peak } }) : lv() })),
  });
  return { clock, rec, log, refresh, sc, view };
}

describe('wording', () => {
  it('uses wellbeing wording and the agreed texts', () => {
    expect(DISTRESS_TEXT).toBe('Possible distress sound — check-in suggested');
    expect(FALL_CUE_TEXT).toBe('Possible fall — with a sound cue — check-in suggested');
    expect(PAUSES_NOTE).toBe('Counts gaps in breathing sounds after regular breathing. Phone placement, bedding and movement all affect it. Not a sleep test — talk to a clinician if you’re worried about your sleep.');
    expect(PAUSES_TITLE).toBe('Possible pauses in breathing sounds');
    expect(TALKING_TITLE).toBe('Talking time (wellbeing trend)');
    expect(PLUG_TEXT).toBe('Keep the phone plugged in');
    const all = [DISTRESS_TEXT, PAUSES_NOTE, PAUSES_WAIT, SNORING_NOTE, TALKING_NOTE, NIGHT_HINT, PLUG_TEXT, ...Object.values(WAKE_TEXT),
      snoringText(lv({ count: 2, total_s: 90 })), pausesText(pv({ count: 1 })), pausesRate(pv({ regular_s: 600 })), talkingText(lv({ total_s: 60 })), eventText('distress'), eventText('fall', true)];
    for (const t of all) expect(t).not.toMatch(/apn?oea|apnea|diagnos|emergency (detected)?|distress detected|disease|sleep disorder/i);
  });
});

describe('card text', () => {
  it('counts, minutes and per-hour lists; the per-hour pause figure is withheld under 30 min', () => {
    expect(snoringText(lv({ count: 3, total_s: 150 }))).toBe('Snoring sounds this session: 3 · 2.5 min');
    expect(snoringHours(lv({ per_hour: [2, 1] }))).toBe('By session hour: 1: 2 · 2: 1');
    expect(talkingText(lv({ total_s: 90 }))).toBe('Talking time: 1.5 min');
    expect(talkingHours(lv({ per_hour_s: [60, 30] }))).toBe('By session hour: 1: 1.0 min · 2: 0.5 min');
    expect(pausesRate(pv())).toBe(PAUSES_WAIT);
    expect(pausesRate(pv({ regular_s: 600, count: 2 }))).toContain('withheld until 30 min');
    expect(pausesRate(pv({ regular_s: 2400, count: 4, rate_per_h: 6 }))).toContain('About 6.0 per hour');
    expect(pausesDurations(pv({ pauses: [{ t_start: 1, t_end: 16, duration_s: 15 }] }))).toBe('Lengths: 15 s');
  });
});

describe('SoundCheckins', () => {
  it('a distress sound raises a log entry (banner text) and an export event, once, and keeps the stored copy equal while it extends', () => {
    const { rec, log, refresh, sc, view } = setup();
    sc.onSounds(view(10, 'Screaming'));
    expect(log.entries.map((e) => e.text)).toEqual([DISTRESS_TEXT]);
    expect(log.pending?.kind).toBe('distress');
    expect(refresh).toHaveBeenCalled();
    sc.onSounds(view(10.48, 'Screaming', 1.48)); // same event grows: no new entry
    expect(log.entries).toHaveLength(1);
    expect(rec.events).toHaveLength(1);
    expect(rec.events[0]).toMatchObject({ kind: 'distress_sound', trigger: 'Screaming' });
    expect(log.acknowledge(1)).toBe(true);
    expect(log.pending).toBeNull();
    sc.onSounds(view(30, 'Glass', 1, 0.8, 1));
    expect(log.entries).toHaveLength(2);
    expect(rec.events.map((e) => e.trigger)).toEqual(['Screaming', 'Glass']);
  });
  it('a thump 4 s after the fall check-in only upgrades the fall (no second check-in); 20 s after is its own check-in', () => {
    for (const [gap, cue] of [[4, true], [40, false]] as const) {
      const { clock, rec, log, sc, view } = setup();
      clock.ms += 100_000;
      const fallEntry = log.add({ kind: 'fall' }, clock.ms);
      const row = rec.addEvent({ kind: 'possible_fall', agoS: 20 });
      sc.registerFall({ tOnset: 0, tConfirmed: 20 }, fallEntry.id, row);
      clock.ms += gap * 1000;
      sc.onSounds(view(500, 'Thump, thud', 0.5));
      expect(log.entries[0]!.text).toBe(cue ? FALL_CUE_TEXT : FALL_TEXT);
      expect(row!.sound_cue).toBe(cue ? true : undefined);
      expect(log.visible).toHaveLength(cue ? 1 : 2);
      expect(log.pendingCount).toBe(cue ? 1 : 2);
      const d = rec.events.find((e) => e.kind === 'distress_sound')!;
      expect(d.merged_into_fall).toBe(cue ? true : undefined); // exported, marked as merged
    }
  });
  it('a sound BEFORE the fall confirms: its banner is replaced in place by the one fall check-in with the cue; acknowledging it clears both', () => {
    const { clock, rec, log, sc, view } = setup();
    clock.ms += 50_000;
    sc.onSounds(view(40, 'Thump, thud', 0.5));
    expect(log.pending?.text).toBe(DISTRESS_TEXT);
    clock.ms += 15_000;
    const fe = log.add({ kind: 'fall' }, clock.ms);
    const row = rec.addEvent({ kind: 'possible_fall', agoS: 20 });
    sc.registerFall({ tOnset: 0, tConfirmed: 20 }, fe.id, row);
    expect(log.visible.map((e) => e.text)).toEqual([FALL_CUE_TEXT]);
    expect(log.pending?.text).toBe(FALL_CUE_TEXT);
    expect(log.pendingCount).toBe(1);
    expect(row!.sound_cue).toBe(true);
    expect(rec.events.find((e) => e.kind === 'distress_sound')!.merged_into_fall).toBe(true);
    expect(log.acknowledge(fe.id)).toBe(true);
    expect(log.pending).toBeNull();
    expect(log.entries.every((e) => e.acknowledged)).toBe(true);
    expect(validateSession(rec.toJSON())).toEqual([]);
  });
  it('crying never corroborates: separate check-ins', () => {
    const b = setup();
    b.clock.ms += 50_000;
    b.sc.onSounds(b.view(40, 'Crying, sobbing', 2));
    const fe2 = b.log.add({ kind: 'fall' }, b.clock.ms);
    const row2 = b.rec.addEvent({ kind: 'possible_fall', agoS: 5 });
    b.sc.registerFall({ tOnset: 0, tConfirmed: 5 }, fe2.id, row2);
    expect(row2!.sound_cue).toBeUndefined();
    expect(b.log.visible).toHaveLength(2);
  });
  it('an audio restart is explicit: the first sound of the new session is a new check-in and export row; totals carry over', () => {
    const { rec, log, sc, view } = setup();
    sc.onSounds(view(10, 'Thump, thud'));
    expect(log.entries).toHaveLength(1);
    sc.reset(); // a new AudioSession: fresh trackers (count back at 1 for the new thump)
    sc.onSounds(view(10, 'Thump, thud'));
    expect(log.entries).toHaveLength(2);
    expect(rec.events.filter((e) => e.kind === 'distress_sound')).toHaveLength(2);
  });
  it('night totals (counts, minutes, per-hour buckets, regular breathing, pauses) carry across an audio restart', () => {
    const { clock, sc } = setup();
    clock.ms += 10_000;
    const mk = (n: number, hourIdx: number, reg: number, pauses: number): SoundsView => ({
      t: 100, distress: [],
      snoring: lv({ count: n, total_s: 60 * n, per_hour: Object.assign([0, 0, 0].slice(0, hourIdx + 1), { [hourIdx]: n }), per_hour_s: Object.assign([0, 0, 0].slice(0, hourIdx + 1), { [hourIdx]: 60 * n }) }),
      talking: lv({ count: 1, total_s: 30, per_hour_s: [30] }),
      pauses: pv({ count: pauses, regular_s: reg, pauses: Array.from({ length: pauses }, (_, i) => ({ t_start: i, t_end: i + 15, duration_s: 15 })) }),
    });
    sc.onSounds(mk(2, 0, 1000, 1));
    sc.reset();
    const m = sc.onSounds(mk(3, 1, 900, 1)); // the second session's own totals
    expect(m.snoring.count).toBe(5);
    expect(m.snoring.total_s).toBe(300);
    expect(m.snoring.per_hour).toEqual([2, 3]); // keyed to session hours, added per bucket
    expect(m.snoring.per_hour_s).toEqual([120, 180]);
    expect(m.talking.total_s).toBe(60);
    expect(m.pauses.count).toBe(2);
    expect(m.pauses.pauses).toHaveLength(2);
    expect(m.pauses.regular_s).toBe(1900);
    expect(m.pauses.rate_per_h).toBeCloseTo(2 / (1900 / 3600), 9); // over 30 min together although neither session alone was
    expect(sc.summaryText()).toContain('Snoring sounds: 5');
  });
  it('writes snoring / pauses / talking summaries that validate, and one possible_pause event per new pause', () => {
    const { clock, rec, sc } = setup();
    clock.ms += 10_000;
    const v: SoundsView = {
      t: 100, snoring: lv({ count: 2, total_s: 120, per_hour: [2] }), talking: lv({ count: 1, total_s: 30, per_hour_s: [30] }), distress: [],
      pauses: pv({ count: 1, regular_s: 900, pauses: [{ t_start: 80, t_end: 95, duration_s: 15 }] }),
    };
    sc.onSounds(v);
    sc.onSounds({ ...v, t: 101 });
    sc.finish();
    expect(rec.events.filter((e) => e.kind === 'possible_pause')).toHaveLength(1);
    const kinds = rec.summaries.map((s) => s.kind).sort();
    expect(kinds).toEqual(['pauses', 'snoring', 'talking']);
    expect(rec.summaries.find((s) => s.kind === 'pauses')).toMatchObject({ count: 1, regular_min: 15, durations_s: [15] });
    expect(rec.summaries.find((s) => s.kind === 'pauses')!.rate_per_h).toBeUndefined(); // withheld: under 30 min
    expect(rec.summaries.find((s) => s.kind === 'talking')).toMatchObject({ minutes: 0.5, per_hour: [0.5] });
    expect(validateSession(rec.toJSON())).toEqual([]);
    expect(sc.summaryText()).toContain('Possible pauses in breathing sounds: 1');
  });
});

describe('export mapping', () => {
  it('withholds the rate and reports honestly when nothing regular was heard', () => {
    expect(pausesSummary(pv())).toMatchObject({ kind: 'pauses', reported: false, reason: 'no_regular_breathing', count: 0 });
    expect(pausesSummary(pv({ regular_s: 2400, count: 4, rate_per_h: 6 })).rate_per_h).toBe(6);
    expect(snoringSummary(lv({ count: 1, total_s: 60, per_hour: [1] }))).toMatchObject({ count: 1, minutes: 1 });
    expect(talkingSummary(lv({ total_s: 120, per_hour_s: [120] }))).toMatchObject({ minutes: 2, per_hour: [2] });
  });
});

describe('EventLog sound cue', () => {
  it('marks only possible falls, once', () => {
    const log = new EventLog();
    const f = log.add({ kind: 'fall' }, 0), d = log.add({ kind: 'distress', label: 'Glass' }, 0), i = log.add({ kind: 'inactivity' }, 0);
    expect(log.markSoundCue(d.id)).toBe(false);
    expect(log.markSoundCue(i.id)).toBe(false);
    expect(log.markSoundCue(f.id)).toBe(true);
    expect(log.markSoundCue(f.id)).toBe(false);
    expect(f.text).toBe('Possible fall — with a sound cue — check-in suggested');
  });
});

describe('WakeLockKeeper', () => {
  const doc = () => {
    const l = new Set<() => void>();
    return { visibilityState: 'visible', addEventListener: (_t: string, cb: () => void) => l.add(cb), removeEventListener: (_t: string, cb: () => void) => l.delete(cb), fire: () => l.forEach((c) => c()), size: () => l.size };
  };
  it('holds a screen lock, asks again when the page returns, and releases on stop', async () => {
    const release = vi.fn(async () => undefined);
    const request = vi.fn(async () => ({ release }));
    const d = doc();
    const seen: string[] = [];
    const k = new WakeLockKeeper({ wakeLock: { request } }, d as never, (s) => seen.push(s));
    expect(await k.start()).toBe('active');
    d.fire(); // already held: no second request
    expect(request).toHaveBeenCalledTimes(1);
    k.stop();
    expect(release).toHaveBeenCalled();
    expect(d.size()).toBe(0);
    expect(seen).toEqual(['active']);
  });
  it('a deferred request: a second start or a visibility change joins it, stop() before it resolves releases the late lock', async () => {
    const d = (() => { const l = new Set<() => void>(); return { visibilityState: 'visible', addEventListener: (_t: string, cb: () => void) => l.add(cb), removeEventListener: (_t: string, cb: () => void) => l.delete(cb), fire: () => l.forEach((c) => c()) }; })();
    const releases: ReturnType<typeof vi.fn>[] = [];
    let resolve: (s: { release: () => Promise<void> }) => void = () => undefined;
    const request = vi.fn(() => new Promise<{ release: () => Promise<void> }>((r) => { resolve = r; }));
    const k = new WakeLockKeeper({ wakeLock: { request } }, d as never);
    const p1 = k.start();
    const p2 = k.start();
    d.fire();
    expect(request).toHaveBeenCalledTimes(1);
    k.stop();
    const rel = vi.fn(async () => undefined);
    releases.push(rel);
    resolve({ release: rel });
    await Promise.all([p1, p2]);
    expect(rel).toHaveBeenCalledTimes(1);
    expect(k.status).toBe('idle');
  });
  it('an older sentinel is released when a newer one replaces it', async () => {
    const d = { visibilityState: 'visible', addEventListener: () => undefined, removeEventListener: () => undefined };
    const rels = [vi.fn(async () => undefined), vi.fn(async () => undefined)];
    let n = 0;
    const k = new WakeLockKeeper({ wakeLock: { request: async () => ({ release: rels[n++]! }) } }, d as never);
    await k.start();
    await (k as unknown as { request(): Promise<void> }).request(); // a second lock arrives while the first is held
    expect(rels[0]).toHaveBeenCalled();
    k.stop();
    expect(rels[1]).toHaveBeenCalled();
  });
  it('says so when the browser has no wake lock or refuses it', async () => {
    expect(await new WakeLockKeeper({}, doc() as never).start()).toBe('unsupported');
    expect(await new WakeLockKeeper({ wakeLock: { request: async () => { throw new Error('NotAllowedError'); } } }, doc() as never).start()).toBe('denied');
    expect(WAKE_TEXT.unsupported).toContain('auto-lock');
  });
});

describe('sound model unavailable', () => {
  it('pauses card says so instead of counting zeros', () => {
    expect(PAUSES_NO_MODEL).toBe("Pause counting needs the sound model, which didn't load.");
    expect(SOUNDS_NO_MODEL).toContain('sound model');
    expect(PAUSES_NO_MODEL).not.toMatch(/apn?oea|diagnos|emergenc/i);
  });
});

describe('NightChoices', () => {
  it('forces the microphone on and locked, camera off; unticking restores the earlier choices', () => {
    const n = new NightChoices();
    expect(n.apply(true, { camera: true, mic: false })).toEqual({ camera: false, mic: true, micLocked: true });
    expect(n.apply(true, { camera: true, mic: true })).toEqual({ camera: false, mic: true, micLocked: true }); // re-ticking the camera in night mode does not overwrite the saved choice
    expect(n.apply(false, { camera: false, mic: true })).toEqual({ camera: true, mic: false, micLocked: false });
    expect(n.apply(false, { camera: false, mic: false })).toEqual({ camera: false, mic: false, micLocked: false }); // never ticked: nothing to restore
  });
});
