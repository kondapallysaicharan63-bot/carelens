import { describe, expect, it } from 'vitest';
import params from '../../shared/params.json';
import { LabelEventTracker, detectLabelEvents, frameSeconds, labelSections, summarizeEvents, type AudioEventsParams, type AudioParams } from '../src/core/audio';
import { BumpStream, PauseTracker, assertSleepParams, audioEventsErrors, sleepSoundsErrors, soundCue, type SleepSoundsParams } from '../src/core/sleepSounds';
import { validateSession } from '../src/export/schema';

const AP = params.audio as AudioParams;
const EV = params.audio_events as AudioEventsParams;
const SL = params.sleep_sounds as SleepSoundsParams;
const ENV_FS = AP.fs_hz / AP.envelope.hop;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

describe('W10 params', () => {
  it('validate and reject broken values', () => {
    expect(() => assertSleepParams(EV, SL)).not.toThrow();
    const bad = (f: (e: AudioEventsParams, s: SleepSoundsParams) => void) => {
      const e = structuredClone(EV), s = structuredClone(SL);
      f(e, s);
      return [...audioEventsErrors(e), ...sleepSoundsErrors(s)];
    };
    for (const f of [
      (e: AudioEventsParams) => { e.snoring.off_thr = 0.9; }, (e: AudioEventsParams) => { e.cue_labels = ['Nope']; }, (e: AudioEventsParams) => { e.glass.label = 'Snoring'; },
      (e: AudioEventsParams) => { e.corroborate_s = 0; }, (_e: AudioEventsParams, s: SleepSoundsParams) => { s.pause_max_s = 5; },
      (_e: AudioEventsParams, s: SleepSoundsParams) => { s.pre_min_bumps = 2; }, (_e: AudioEventsParams, s: SleepSoundsParams) => { s.bump.floor_win_s = 5; }, (_e: AudioEventsParams, s: SleepSoundsParams) => { s.bump.merge_s = 2.5; },
    ]) expect(bad(f), String(f)).not.toEqual([]);
  });
  it('uses the verified YAMNet class names', () => {
    expect(Object.keys(labelSections({ audio: AP, audio_events: EV }))).toEqual(['Cough', 'Snoring', 'Screaming', 'Crying, sobbing', 'Glass', 'Thump, thud', 'Speech']);
    expect(EV.cue_labels).toEqual(['Thump, thud', 'Screaming', 'Glass']);
    expect(EV.corroborate_s).toBe(10);
  });
});

describe('LabelEventTracker totals and per-hour counts', () => {
  it('equal the batch detector on every prefix, across session hours', () => {
    const r = lcg(3);
    for (const label of ['Snoring', 'Speech', 'Screaming']) {
      const p = labelSections({ audio: AP, audio_events: EV })[label]!;
      const frames: [number, number][] = [];
      let t = 1;
      for (let i = 0; i < 900; i++) {
        frames.push([t, [0.05, 0.1, 0.25, 0.35, 0.6, 0.9][Math.floor(r() * 6)]!]);
        t += r() < 0.97 ? 0.48 : 600;
      }
      const tr = new LabelEventTracker(p, 500, frameSeconds(AP));
      frames.forEach(([ft, s], i) => {
        tr.push(ft, s);
        if (i % 7 === 0 || i === frames.length - 1) {
          const v = tr.view();
          const b = detectLabelEvents(frames.slice(0, i + 1), p);
          const sm = summarizeEvents(b.events, frameSeconds(AP));
          expect(v.count).toBe(b.count);
          expect(v.per_hour).toEqual(sm.per_hour);
          expect(v.per_hour_s).toHaveLength(sm.per_hour_s.length);
          v.per_hour_s.forEach((x, k) => expect(Math.abs(x - sm.per_hour_s[k]!)).toBeLessThan(1e-6));
          expect(Math.abs(v.total_s - sm.total_s)).toBeLessThan(1e-6);
        }
      });
    }
  });
  it('keeps constant memory over an 8 h stream', () => {
    const tr = new LabelEventTracker(EV.snoring, 500, frameSeconds(AP));
    const r = lcg(1);
    const n = Math.floor((8 * 3600) / 0.48);
    for (let i = 0; i < n; i++) tr.push(1 + i * 0.48, Math.floor(i / 40) % 3 === 0 && r() < 0.9 ? 0.7 : 0.05);
    const v = tr.view();
    expect(v.count).toBeGreaterThan(500);
    expect(v.ticks.length).toBeLessThanOrEqual(500);
    expect(v.per_hour).toHaveLength(8);
    expect(v.per_hour.reduce((a, b) => a + b, 0)).toBe(v.count);
    expect(v.total_s).toBeGreaterThan(3600);
  });
});

describe('PauseTracker', () => {
  it('counts one pause in regular breathing and gives no per-hour figure under 30 min', () => {
    const tr = new PauseTracker(SL, ENV_FS);
    const per = 60 / 14;
    let t = 0;
    for (let i = 0; i < 40; i++) { tr.pushBump(t); t += per; }
    t += 15 - per;
    for (let i = 0; i < 20; i++) { tr.pushBump(t); t += per; }
    const v = tr.view();
    expect(v.count).toBe(1);
    expect(v.pauses[0]!.duration_s).toBeCloseTo(15, 6);
    expect(v.rate_per_h).toBeNull();
  });
  it('a 90 s silence is no breathing sounds, speech or a spike in the gap voids it, irregular breathing before it voids it', () => {
    const make = (gap: number, setup?: (tr: PauseTracker) => void, irregular = false) => {
      const tr = new PauseTracker(SL, ENV_FS);
      const r = lcg(5);
      let t = 0;
      for (let i = 0; i < 20; i++) { tr.pushBump(t); t += 60 / 14 * (irregular ? 1 + 1.4 * (r() - 0.5) : 1); }
      setup?.(tr);
      t += gap;
      for (let i = 0; i < 5; i++) { tr.pushBump(t); t += 60 / 14; }
      return tr.view();
    };
    expect(make(15).count).toBe(1);
    const s = make(90);
    expect(s.count).toBe(0);
    expect(s.no_sound_gaps).toBe(1);
    expect(make(15, (tr) => tr.pushSpeech(88, 0.9)).count).toBe(0);
    expect(make(15, (tr) => tr.pushSpike(88, 89)).count).toBe(0);
    expect(make(15, undefined, true).count).toBe(0);
  });
  it('reports a rate once 30 min of regular breathing are covered', () => {
    const tr = new PauseTracker(SL, ENV_FS);
    let t = 0;
    for (let k = 0; k < 4; k++) { for (let i = 0; i < 7 * 14 * 3; i++) { tr.pushBump(t); t += 60 / 14; } t += 15; }
    const v = tr.view();
    expect(v.count).toBe(3);
    expect(v.regular_s).toBeGreaterThan(1800);
    expect(v.rate_per_h).toBeCloseTo(3 / (v.regular_s / 3600), 9);
  });
  it('a synthetic envelope with one gap gives one pause, without it none; no signal gives none', () => {
    const run = (gap: boolean) => {
      const tr = new PauseTracker(SL, ENV_FS);
      const r = lcg(9);
      const starts: number[] = [];
      for (let s = 1.3; s < 170; s += 60 / 14) if (!(gap && s >= 80 && s < 93)) starts.push(s);
      for (let i = 0; i < 170 * 50; i++) {
        const t = i / 50;
        let v = 1 + 0.04 * (r() - 0.5);
        for (const s of starts) { const x = (t - s) / (0.5 * 60 / 14); if (x >= 0 && x < 1) v += 0.6 * 0.5 * (1 - Math.cos(2 * Math.PI * x)); }
        tr.pushFrame(t, v);
      }
      return tr.view();
    };
    expect(run(true).count).toBe(1);
    expect(run(false).count).toBe(0);
    const tr = new PauseTracker(SL, ENV_FS);
    for (let i = 0; i < 3000; i++) tr.pushFrame(i / 50, i % 2 ? null : Number.NaN);
    expect(tr.view().count).toBe(0);
  });
  it('keeps bounded buffers over an 8 h bump stream and a 20 min frame stream', () => {
    const tr = new PauseTracker(SL, ENV_FS);
    const per = 60 / 14;
    let t = 0, n = 0;
    while (t < 8 * 3600) {
      tr.pushBump(t); n++;
      if (n % 400 === 0) t += 15;
      t += per;
      if (n % 50 === 0) tr.pushSpeech(t, 0.02);
      if (n % 997 === 0) tr.pushSpike(t + 1, t + 1.5);
    }
    const v = tr.view();
    expect(v.count).toBeGreaterThanOrEqual(12);
    expect(v.count).toBeLessThanOrEqual(17);
    expect(tr.sizes.recent).toBeLessThanOrEqual(600);
    expect(tr.sizes.speech).toBeLessThan(100);
    expect(tr.sizes.spikes).toBeLessThan(20);
    expect(v.pauses.length).toBeLessThanOrEqual(500);
    expect(v.rate_per_h!).toBeGreaterThan(1.4);
    expect(v.rate_per_h!).toBeLessThan(2.3);
    const st = new BumpStream(SL, ENV_FS);
    for (let i = 0; i < 50 * 1200; i++) st.push(i / 50, 1 + 0.5 * (i % 214 < 25 ? 1 : 0));
    expect(st.sizes.mono).toBeLessThanOrEqual(st.sizes.floorN);
    expect(st.sizes.rises).toBeLessThanOrEqual(16);
  });
});

describe('soundCue', () => {
  const thud = (t: number) => [{ label: 'Thump, thud', t_start: t, t_end: t + 1 }];
  const fall = { t_onset: 100, t_confirmed: 100 };
  it('a thump 4 s after gives a cue, 20 s after none; screaming and glass count, crying and cough do not', () => {
    expect(soundCue(fall, thud(104), EV)).toBe(true);
    expect(soundCue(fall, thud(120), EV)).toBe(false);
    expect(soundCue(fall, [{ label: 'Screaming', t_start: 90, t_end: 91 }], EV)).toBe(true);
    expect(soundCue(fall, [{ label: 'Glass', t_start: 110, t_end: 111 }], EV)).toBe(true);
    expect(soundCue(fall, [{ label: 'Crying, sobbing', t_start: 101, t_end: 105 }], EV)).toBe(false);
    expect(soundCue(fall, [{ label: 'Cough', t_start: 101, t_end: 102 }], EV)).toBe(false);
    expect(soundCue(fall, [], EV)).toBe(false);
  });
  it('uses the whole fall interval (onset to check-in)', () => {
    expect(soundCue({ t_onset: 100, t_confirmed: 120 }, thud(124), EV)).toBe(true);
    expect(soundCue({ t_onset: 100, t_confirmed: 120 }, thud(131), EV)).toBe(false);
  });
});

describe('export schema (W10)', () => {
  const ex = () => JSON.parse(JSON.stringify({ ...JSON.parse(require('node:fs').readFileSync('../shared/session.example.json', 'utf8')) }));
  const w = '2026-11-20T10:00:09.000-08:00';
  const good = () => {
    const s = ex();
    s.events.push({ t: 5, wall: w, kind: 'distress_sound', t_onset: 4, t_end: 6, peak_score: 0.8, trigger: 'Screaming', merged_into_fall: true }, { t: 9, wall: w, kind: 'possible_pause', t_onset: 1, t_end: 16 }, { t: 11, wall: w, kind: 'possible_fall', t_onset: 0.5, sound_cue: true });
    s.summaries = [...(s.summaries ?? []),
      { t: 9, wall: w, kind: 'snoring', reported: true, reason: 'ok', count: 3, minutes: 1.5, per_hour: [3] },
      { t: 9, wall: w, kind: 'pauses', reported: true, reason: 'ok', count: 1, regular_min: 12, durations_s: [15] },
      { t: 9, wall: w, kind: 'talking', reported: true, reason: 'ok', minutes: 2, per_hour: [2] }];
    return s;
  };
  it('accepts the new rows, stays backward compatible and rejects bad ones', () => {
    expect(validateSession(good())).toEqual([]);
    expect(validateSession(ex())).toEqual([]);
    const edits: ((s: any) => void)[] = [
      (s) => { s.events[s.events.length - 1].sound_cue = 'yes'; }, (s) => { s.events[s.events.length - 3].merged_into_fall = 1; }, (s) => { delete s.summaries[s.summaries.length - 1].minutes; },
      (s) => { s.summaries[s.summaries.length - 2].count = 1.5; }, (s) => { s.summaries[s.summaries.length - 3].per_hour = new Array(73).fill(1); },
      (s) => { s.summaries[s.summaries.length - 2].durations_s = [-1]; }, (s) => { s.events[s.events.length - 3].kind = 'apnea'; },
    ];
    for (const f of edits) { const s = good(); f(s); expect(validateSession(s), String(f)).not.toEqual([]); }
  });
});
