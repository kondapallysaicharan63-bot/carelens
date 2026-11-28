// W9 blink rate / eyes-closed time and activity level: TS port checks on the shared synthetic fixtures (bit-level parity is in
// test/parity), gating, params validators, shared face session, card flow, export mapping. Synthetic data only. Not a medical device.
import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import params from '../../shared/params.json';
import { BlinkTracker, assertBlinkParams, blinkErrors, blinkIndices, frameEar, headTurned, percentile, runBlinkFixture, type BlinkParams, type FacePt } from '../src/core/blink';
import { ActivitySummary, ActivityTimeline, naiveWallMs } from '../src/core/events';
import { runActivityFixture, type PoseChainParams } from '../src/core/pipeline';
import { compareOutputs, type Fixture } from '../src/core/parity';
import { DistinctFrames } from '../src/capture/distinct';
import { FaceHrSession } from '../src/ui/faceHr';
import { blinkGuidance, buildBlinkCard, BLINK_NOTE } from '../src/ui/blinkCard';
import { activityText, buildActivityCard, postureBarText } from '../src/ui/activityCard';
import { activitySummaryRow, blinkSummary } from '../src/export/wellbeing';
import { SessionRecorder } from '../src/export/recorder';
import { validateSession } from '../src/export/schema';

const BP = params.blink as unknown as BlinkParams;
const fx = (name: string): Fixture => JSON.parse(readFileSync(`../shared/fixtures/${name}.json`, 'utf8'));
const runB = (input: any): any => runBlinkFixture(input, params as any);

function sparse(input: any, i: number): FacePt[] | null {
  const fr = input.landmarks[i];
  if (!fr) return null;
  const lm: FacePt[] = [];
  (input.landmark_indices as number[]).forEach((k, j) => { lm[k] = { x: fr[j][0], y: fr[j][1] }; });
  return lm;
}

describe('blink params', () => {
  it('validate and reject inconsistent sets (mirror of carelens.params.blink_errors)', () => {
    expect(blinkErrors(BP)).toEqual([]);
    expect(assertBlinkParams(BP)).toBe(BP);
    for (const bad of [{ perclos_frac: 0.6 }, { closed_frac: 0.8 }, { reopen_frac: 1 }, { blink_min_ms: 600 }, { min_window_s: 90 }, { baseline_pct: 0 }, { min_rate_hz: 10 },
      { max_dt_s: 3 }, { baseline_s: 0 }, { eye_landmarks: { left: [33, 160, 158, 133, 153, 33], right: BP.eye_landmarks.right } }, { pose_landmarks: { nose: 1, forehead: 10, mouth: 13 } }]) {
      expect(blinkErrors({ ...BP, ...bad } as any), JSON.stringify(bad)).not.toEqual([]);
      expect(() => assertBlinkParams({ ...BP, ...bad } as any)).toThrow();
    }
    expect(BP.eye_landmarks).toEqual({ left: [33, 160, 158, 133, 153, 144], right: [362, 385, 387, 263, 373, 380] });
    expect(blinkIndices(BP)).toEqual([1, 10, 33, 133, 144, 152, 153, 158, 160, 263, 362, 373, 380, 385, 387]);
    expect(params.activity.min_report_s).toBeGreaterThan(0);
  });

  it('percentile matches the linear-interpolation definition', () => {
    expect(percentile([1, 2, 3, 4, 5], 50)).toBe(3);
    expect(percentile([0, 10], 90)).toBeCloseTo(9);
    expect(percentile([5], 90)).toBe(5);
  });
});

describe('blink fixtures', () => {
  it.each([['blink_8pm', 8], ['blink_15pm', 15], ['blink_25pm', 25]])('%s reports about the scripted rate and matches the reference', (name, bpm) => {
    const f = fx(name);
    const r = runB(f.input);
    expect(r.final.reason).toBe('ok');
    expect(r.final.reported).toBe(true);
    expect(Math.abs(r.final.blinks_per_min - bpm)).toBeLessThanOrEqual(3);
    expect(r.final.long_closures).toBe(0);
    expect(compareOutputs(f, r)).toEqual([]);
  });

  it('long closures are counted separately and raise eyes-closed time', () => {
    const r = runB(fx('blink_long_closures').input);
    expect(r.final.long_closures).toBe(4);
    expect(r.long_closures.length).toBe(4);
    expect(r.final.perclos).toBeGreaterThan(0.1);
    expect(r.blinks.every((b: any) => b.t_end - b.t_start <= 0.5)).toBe(true);
  });

  it('a turned head withholds with head_turned and no blink is counted while turned', () => {
    const r = runB(fx('blink_head_turned').input);
    const reasons = r.reason_runs.map((x: any) => x.reason);
    expect(reasons).toContain('head_turned');
    expect(reasons[reasons.length - 1]).toBe('ok');
    expect(r.blinks.some((b: any) => b.t_start > 40 && b.t_start < 50)).toBe(false);
  });

  it('15 distinct frames per second is low_fps: withheld, never a guess', () => {
    const r = runB(fx('blink_low_fps').input);
    expect(r.final).toMatchObject({ reason: 'low_fps', reported: false });
    expect(r.blinks).toEqual([]);
  });

  it('a lost face is no_face, does not count as time, and measuring resumes', () => {
    const input = JSON.parse(JSON.stringify(fx('blink_15pm').input));
    for (let i = 30 * 25; i < 40 * 25; i++) input.landmarks[i] = null;
    const r = runB(input);
    expect(r.reason_runs.some((x: any) => x.reason === 'no_face')).toBe(true);
    expect(r.final.reason).toBe('ok');
    expect(r.final.window_s).toBeLessThan(41);
  });

  it('EAR is the mean of both eyes; head pose estimate flags a turned or pitched face', () => {
    const input = fx('blink_15pm').input;
    const lm = sparse(input, 0)!;
    const ear = frameEar(lm, input.w / input.h, BP)!;
    expect(ear).toBeGreaterThan(0.2);
    expect(ear).toBeLessThan(0.4);
    expect(headTurned(lm, input.w / input.h, BP)).toBe(false);
    const turned = lm.slice();
    turned[1] = { x: lm[1]!.x + 0.08, y: lm[1]!.y };
    expect(headTurned(turned, input.w / input.h, BP)).toBe(true);
    const pitched = lm.slice();
    pitched[1] = { x: lm[1]!.x, y: lm[10]!.y + 0.95 * (lm[152]!.y - lm[10]!.y) };
    expect(headTurned(pitched, input.w / input.h, BP)).toBe(true);
  });

  it('a frame gap is unobserved time: it ends an open closure and is not counted', () => {
    const input = fx('blink_15pm').input;
    const tr = new BlinkTracker(BP);
    for (let i = 0; i < 25 * 30; i++) tr.update(i / 25, sparse(input, i), input.w / input.h);
    const before = tr.validTotal;
    tr.update(30 + 5, sparse(input, 0), input.w / input.h);
    expect(tr.validTotal).toBeCloseTo(before, 9);
  });
});

describe('activity port', () => {
  const P = params as unknown as PoseChainParams;
  it('fixtures match the reference and add up', () => {
    for (const n of ['activity_session', 'activity_gap_dropout']) {
      const f = fx(n);
      const r = runActivityFixture(f.input, P);
      expect(compareOutputs(f, r), n).toEqual([]);
      expect(r.summary.reported).toBe(true);
      const bucketSum = Object.values(r.timeline as Record<string, Record<string, number>>).flatMap((b) => Object.values(b)).reduce((a, b) => a + b, 0);
      expect(Math.abs(bucketSum - r.summary.total_s)).toBeLessThan(1.5) // the timeline caps an outage step at 1 s, the summary credits 0;
    }
    const s = runActivityFixture(fx('activity_session').input, P).summary;
    expect(s.absent_s).toBeGreaterThan(5);
    expect(s.longest_still_s).toBeGreaterThan(30);
  });

  const state = (t: number, present = true, motion = 0) => ({ t, present, motion, posture: present ? 'sitting' : 'absent' }) as any;
  it('splits active / still / absent, tracks the longest still stretch, buckets by hour', () => {
    const still = params.common.still_motion_bodysizes_s;
    const a = new ActivitySummary(still, 1, 30);
    const tl = new ActivityTimeline(Date.UTC(2026, 0, 5, 8, 59, 50), 1);
    for (let k = 0; k < 560; k++) {
      const t = k / 10;
      const kind = t < 10 ? 'still' : t < 15 ? 'active' : t < 40 ? 'still' : t < 46 ? 'absent' : 'still';
      const s = state(t, kind !== 'absent', kind === 'active' ? 0.5 : still / 2);
      a.update(s); tl.update(s);
    }
    const d = a.toDict();
    expect(d.active_s).toBeCloseTo(5, 0);
    expect(d.absent_s).toBeCloseTo(6, 0);
    expect(d.still_s).toBeCloseTo(45, 0);
    expect(d.longest_still_s).toBeCloseTo(25, 0);
    expect(Object.keys(tl.toDict())).toEqual(['2026-01-05 08:00', '2026-01-05 09:00']);
    expect(new ActivitySummary(still, 1, 30).toDict().reason).toBe('too_short');
    // a new stream restarts the clock: no negative time, a still stretch ends
    a.breakStream(); tl.breakStream();
    a.update(state(0)); a.update(state(0.5));
    expect(a.totalS).toBeGreaterThan(d.total_s);
  });

  it('naiveWallMs reads the local calendar fields as UTC', () => {
    const d = new Date(2026, 5, 1, 13, 45, 12);
    const n = new Date(naiveWallMs(d));
    expect([n.getUTCFullYear(), n.getUTCMonth(), n.getUTCDate(), n.getUTCHours(), n.getUTCMinutes(), n.getUTCSeconds()]).toEqual([2026, 5, 1, 13, 45, 12]);
  });
});

// ------------------------------------------------------------------ browser pieces (fake DOM)

class FakeEl {
  attrs: Record<string, string> = {}; kids: (FakeEl | string)[] = []; listeners: Record<string, (() => void)[]> = {};
  hidden = false; disabled = false; checked = false; className = ''; value = ''; private text = '';
  constructor(readonly tag: string, private readonly reg: Map<string, FakeEl>) {}
  setAttribute(k: string, v: string): void { this.attrs[k] = v; if (k === 'id') this.reg.set(v, this); }
  getAttribute(k: string): string | null { return this.attrs[k] ?? null; }
  removeAttribute(k: string): void { delete this.attrs[k]; }
  append(...c: (FakeEl | string)[]): void { this.kids.push(...c); }
  replaceChildren(...c: (FakeEl | string)[]): void { this.kids = c; this.text = ''; }
  addEventListener(ev: string, fn: () => void): void { (this.listeners[ev] ??= []).push(fn); }
  get classList() { return { toggle: () => undefined, remove: () => undefined, add: () => undefined }; }
  get textContent(): string { return this.text + this.kids.map((k) => (typeof k === 'string' ? k : k.textContent)).join(''); }
  set textContent(v: string) { this.text = v; this.kids = []; }
}
const reg = new Map<string, FakeEl>();
const $ = (id: string): FakeEl => reg.get(id)!;
const flip = (id: string, on: boolean): void => { $(id).checked = on; ($(id).listeners.change ?? []).forEach((f) => f()); };

describe('blink card', () => {
  beforeEach(() => { reg.clear(); (globalThis as any).document = { createElement: (t: string) => new FakeEl(t, reg) }; });
  afterEach(() => { delete (globalThis as any).document; });
  const meter = (q: number | null): any => `q:${q === null ? '-' : q.toFixed(2)}`;

  function rig(declared: number | null = 30) {
    const rows: any[] = [];
    const card = buildBlinkCard({ meter, record: (r) => rows.push(r), getDeclaredFps: () => declared });
    const actives: boolean[] = [];
    card.onActiveChange((a) => actives.push(a));
    return { card, rows, actives };
  }
  const feed = (card: ReturnType<typeof buildBlinkCard>, input: any, reliable = true, until = Infinity): void => {
    for (let i = 0; i < input.landmarks.length && i / input.fps < until; i++) card.onFrame({ t: i / input.fps, lm: sparse(input, i), aspect: input.w / input.h, reliable });
  };

  it('starts off, says so, shows the note, and ignores frames until switched on', () => {
    const { card, rows, actives } = rig();
    expect(actives).toEqual([false]);
    expect($('blink-guidance').textContent).toContain('Blink measurement is off');
    expect($('blink-note').textContent).toBe(BLINK_NOTE);
    expect(BLINK_NOTE).toBe('Experimental. Screens, reading, dry air and light all change blink rate. Compare with your own earlier results.');
    feed(card, fx('blink_15pm').input);
    expect(rows).toEqual([]);
    expect($('blink-value').getAttribute('data-reason')).toBe('idle');
  });

  it('reports blinks/min, eyes-closed share and long closures once calibrated, and records one refreshed summary', () => {
    const { card, rows, actives } = rig();
    card.setLoadState('ready');
    flip('blink-active-toggle', true);
    expect(actives).toEqual([false, true]);
    feed(card, fx('blink_15pm').input, true, 30);
    expect($('blink-value').getAttribute('data-reason')).toBe('calibrating');
    expect($('blink-guidance').textContent).toContain('Learning your open-eye level');
    expect($('blink-value').textContent).toBe('No reading yet');
    feed(card, fx('blink_long_closures').input);
    expect($('blink-value').getAttribute('data-reason')).toBe('ok');
    expect($('blink-value').textContent).toMatch(/^\d+ blinks per minute$/);
    expect($('blink-closed').textContent).toMatch(/^Eyes closed: \d+ % of the last minute$/);
    expect($('blink-long').textContent).toMatch(/Long closures \(over 0\.5 s\): \d+$/);
    const last = rows[rows.length - 1];
    expect(last).toMatchObject({ kind: 'blink', reported: true, reason: 'ok' });
    expect(last.blinks_per_min).toBeGreaterThan(0);
    expect(last.eyes_closed_pct).toBeGreaterThan(5);
    expect(last.long_closures).toBeGreaterThan(0);
    // every row the card records is a valid export row
    const rec = new SessionRecorder(['camera'], () => 1_700_000_000_000, 'Chrome/1');
    for (const r of rows) rec.upsertSummary(r);
    expect(rec.summaries.length).toBe(1);
    expect(validateSession(rec.toJSON())).toEqual([]);
  });

  it('a withheld state never overwrites a reported row; low fps and head turns show their guidance', () => {
    const { card, rows } = rig();
    card.setLoadState('ready');
    flip('blink-active-toggle', true);
    feed(card, fx('blink_head_turned').input);
    expect(rows.some((r) => r.reported)).toBe(true);
    expect(rows[rows.length - 1].reported).toBe(true);
    const { card: c2 } = rig();
    c2.setLoadState('ready');
    flip('blink-active-toggle', true);
    feed(c2, fx('blink_low_fps').input);
    expect($('blink-value').getAttribute('data-reason')).toBe('low_fps');
    expect($('blink-guidance').textContent).toContain('too few frames per second');
    expect($('blink-value').textContent).toBe('No reading yet');
  });

  it('frames whose timing could not be verified are withheld unless the camera declares a high enough rate', () => {
    for (const [declared, reason] of [[null, 'low_fps'], [15, 'low_fps'], [30, 'ok']] as [number | null, string][]) {
      const { card } = rig(declared);
      card.setLoadState('ready');
      flip('blink-active-toggle', true);
      feed(card, fx('blink_15pm').input, false);
      expect($('blink-value').getAttribute('data-reason'), String(declared)).toBe(reason);
    }
  });

  it('guidance covers every state, wording stays plain', () => {
    const texts = [blinkGuidance(false, 'ready', null), blinkGuidance(true, 'loading', null), blinkGuidance(true, 'failed', null, BP, 'boom'),
      ...(['no_face', 'low_fps', 'head_turned', 'calibrating', 'ok'] as const).map((r) => blinkGuidance(true, 'ready', r))];
    expect(texts[2]).toBe('The face model failed to load: boom');
    expect(blinkGuidance(true, 'ready', 'no_face')).toBe('Face the camera; glasses glare can affect this');
    for (const t of texts) expect(t).not.toMatch(/drows|fatigue|sleep|driv|diagnos|disease|tired/i);
    expect(new Set(texts).size).toBe(texts.length);
  });

  it('reset starts the calibration over', () => {
    const { card } = rig();
    card.setLoadState('ready');
    flip('blink-active-toggle', true);
    feed(card, fx('blink_15pm').input);
    expect($('blink-value').getAttribute('data-reason')).toBe('ok');
    card.reset();
    expect($('blink-value').getAttribute('data-reason')).toBe('idle');
  });
});

describe('face session shared by the heart-rate and blink cards', () => {
  const video = { videoWidth: 640, videoHeight: 480, readyState: 4 } as HTMLVideoElement;
  beforeEach(() => { (globalThis as any).document = { createElement: () => ({ getContext: () => null }) }; });
  afterEach(() => { delete (globalThis as any).document; });

  async function rig() {
    let cb: (n: number, m?: any) => void = () => {};
    const subscribe = (_v: HTMLVideoElement, c: any) => { cb = c; return () => {}; };
    const lms: any[] = [];
    const detect = vi.fn(() => lms[detect.mock.calls.length - 1] ?? null);
    const frames: any[] = [];
    const hooks = { onUpdate: () => {}, onOutline: () => {}, onLandmarks: (f: any) => frames.push(f) } as any;
    const s = new FaceHrSession(subscribe, hooks);
    await s.load(async () => ({ detect, close: () => {}, delegate: 'x' }) as any, video);
    return { s, detect, frames, lms, tick: (n: number, mediaTime?: number) => cb(n, mediaTime === undefined ? undefined : { mediaTime }) };
  }

  it('the model runs for the blink path with face heart rate off, and only distinct camera frames are emitted', async () => {
    const { s, detect, frames, tick } = await rig();
    s.setActive(false);
    tick(1000, 0); tick(1033, 0.033);
    expect(detect).not.toHaveBeenCalled(); // nothing wanted: the model does not run
    s.setBlinkActive(true);
    tick(1900, 0.0); tick(1933, 0.033); // the frame clock and the distinct-frame check start up on the first two camera frames
    frames.length = 0; detect.mockClear();
    for (let k = 2; k < 7; k++) { tick(2000 + k * 33, k / 30); tick(2000 + k * 33 + 16, k / 30); tick(2000 + k * 33 + 24, k / 30); } // 3 display callbacks per camera frame
    expect(frames.length).toBe(5);
    expect(detect.mock.calls.length).toBe(5); // repeated callbacks are not even run through the model
    expect(frames.every((f) => f.reliable === true && f.aspect === 640 / 480)).toBe(true);
    for (let i = 1; i < frames.length; i++) expect(frames[i].t).toBeGreaterThan(frames[i - 1].t);
    s.setBlinkActive(false);
    const n = detect.mock.calls.length;
    tick(5000, 1); tick(5033, 1.033);
    expect(detect.mock.calls.length).toBe(n);
  });

  it('heart rate alone does not feed the blink path; pausing (the hand test) holds both', async () => {
    const { s, frames, detect, tick } = await rig();
    for (let k = 0; k < 4; k++) tick(1000 + k * 33, k / 30);
    expect(frames).toEqual([]);
    s.setBlinkActive(true);
    s.setActive(false);
    s.setPaused(true);
    detect.mockClear();
    tick(3000, 5); tick(3033, 5.033);
    expect(detect).not.toHaveBeenCalled();
    expect(frames).toEqual([]);
    s.setPaused(false);
    for (let k = 0; k < 4; k++) tick(4000 + k * 33, 6 + k / 30);
    expect(frames.length).toBeGreaterThan(0);
  });

  it('without per-frame signals, identical landmarks are the same camera frame and the timing is flagged unreliable', async () => {
    const { s, frames, lms, tick } = await rig();
    s.setActive(false);
    s.setBlinkActive(true);
    const a = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
    const b = a.map((p) => ({ x: p.x + 0.001, y: p.y }));
    lms.push(a, a, a, b, b, b, a);
    for (let k = 0; k < 8; k++) tick(1000 + k * 33);
    expect(frames.every((f) => f.reliable === false)).toBe(true);
    expect(frames.length).toBeLessThanOrEqual(4);
  });
});

describe('distinct frames', () => {
  const v = {} as HTMLVideoElement;
  it('mediaTime decides once it has advanced; counter, then unknown otherwise', () => {
    const d = new DistinctFrames();
    expect(d.check(v, { mediaTime: 0 }, null)).toBe('skip'); // clock not running yet
    expect(d.check(v, { mediaTime: 0 }, 0)).toBe('unknown'); // media not advanced yet, no counter: undecided
    expect(d.check(v, { mediaTime: 0.033 }, 0.033)).toBe('new');
    expect(d.check(v, { mediaTime: 0.033 }, 0.033)).toBe('skip');
    expect(d.check({ getVideoPlaybackQuality: () => ({ totalVideoFrames: 7 }) } as any, undefined, 1)).toBe('new');
    const d2 = new DistinctFrames();
    const q = { totalVideoFrames: 3 };
    const vv = { getVideoPlaybackQuality: () => q } as any;
    expect(d2.check(vv, undefined, 1)).toBe('new');
    expect(d2.check(vv, undefined, 1.01)).toBe('skip');
    q.totalVideoFrames = 4;
    expect(d2.check(vv, undefined, 1.03)).toBe('new');
    d2.reset();
    expect(d2.check({} as any, undefined, 2)).toBe('unknown');
  });
});

describe('activity card', () => {
  beforeEach(() => { reg.clear(); (globalThis as any).document = { createElement: (t: string) => new FakeEl(t, reg) }; });
  afterEach(() => { delete (globalThis as any).document; });
  const st = (t: number, present = true, motion = 0, posture = 'sitting') => ({ t, state: { t, present, motion, posture: present ? posture : 'absent' } as any });

  it('shows active / still / away minutes after the minimum, a posture bar with text, and records one refreshed row', () => {
    const rows: any[] = [];
    const card = buildActivityCard({ record: (r) => rows.push(r), now: () => 1_700_000_000_000 });
    expect($('activity-value').textContent).toBe('Starts when the camera is on.');
    for (let k = 0; k <= 400; k++) {
      const t = k / 5;
      card.onView(st(t, t < 70, t >= 20 && t < 30 ? 0.6 : 0.01, t < 50 ? 'sitting' : 'standing'));
    }
    expect($('activity-value').textContent).toMatch(/^Active \d+\.\d min · Still \d+\.\d min · Away \d+\.\d min$/);
    expect(Number($('activity-value').getAttribute('data-active-s'))).toBeCloseTo(10, 0);
    expect(Number($('activity-value').getAttribute('data-away-s'))).toBeCloseTo(10, 0);
    expect($('activity-longest').textContent).toMatch(/^Longest still stretch: \d+\.\d min$/);
    expect($('activity-bar-text').textContent).toMatch(/^Posture time: sitting \d+ %, standing \d+ %, away \d+ %$/);
    expect($('activity-bar').getAttribute('aria-label')).toBe($('activity-bar-text').textContent);
    const last = rows[rows.length - 1];
    expect(last).toMatchObject({ kind: 'activity', reported: true, reason: 'ok' });
    expect(last.active_min).toBeCloseTo(10 / 60, 2);
    const rec = new SessionRecorder(['camera'], () => 1_700_000_000_000, 'Chrome/1');
    for (const r of rows) rec.upsertSummary(r);
    expect(rec.summaries.length).toBe(1);
    expect(validateSession(rec.toJSON())).toEqual([]);
    // the clock starting over (a new stream) never produces negative time
    const before = Number($('activity-value').getAttribute('data-still-s'));
    card.onView(st(0.2)); card.onView(st(0.4));
    for (let k = 0; k < 10; k++) card.onView(st(0.6 + k * 0.2));
    expect(Number($('activity-value').getAttribute('data-still-s'))).toBeGreaterThanOrEqual(before);
  });

  it('text helpers: wellbeing wording, shares add up', () => {
    expect(postureBarText({})).toBe('Posture time: no data yet');
    expect(postureBarText({ sitting: 30, standing: 10, absent: 10 })).toBe('Posture time: sitting 60 %, standing 20 %, away 20 %');
    expect(activityText({ reported: false } as any).headline).toContain('Collecting');
  });
});

describe('export rows', () => {
  it('blink and activity rows map from the core and validate; a withheld blink row carries only the reason', () => {
    const ok = blinkSummary({ reported: true, reason: 'ok', blinks_per_min: 14.96, perclos: 0.0449, long_closures: 2, window_s: 59.99, coverage: 0.98 });
    expect(ok).toEqual({ kind: 'blink', reported: true, reason: 'ok', duration_s: 59.99, blinks_per_min: 15, eyes_closed_pct: 4.5, long_closures: 2, quality: 0.98 });
    const no = blinkSummary({ reported: false, reason: 'head_turned', blinks_per_min: 9, perclos: 0.2, long_closures: 1, window_s: 30, coverage: 0.5 });
    expect(no).toEqual({ kind: 'blink', reported: false, reason: 'head_turned', duration_s: 30 });
    const act = activitySummaryRow({ reported: true, reason: 'ok', active_s: 90, still_s: 600, absent_s: 30, total_s: 720, longest_still_s: 240, active_min: 1.5 });
    expect(act).toMatchObject({ kind: 'activity', active_min: 1.5, still_min: 10, absent_min: 0.5, longest_still_min: 4, duration_s: 720 });
    const rec = new SessionRecorder(['camera'], () => 1_700_000_000_000, 'Chrome/1');
    rec.upsertSummary(ok); rec.upsertSummary(act); rec.upsertSummary({ ...ok, blinks_per_min: 12 });
    expect(rec.summaries.map((r) => [r.kind, r.blinks_per_min ?? r.active_min])).toEqual([['blink', 12], ['activity', 1.5]]);
    expect(validateSession(rec.toJSON())).toEqual([]);
  });

  it('validator rejects bad W9 fields and stays backward compatible', () => {
    const rec = new SessionRecorder(['camera'], () => 1_700_000_000_000, 'Chrome/1');
    expect(validateSession(rec.toJSON())).toEqual([]); // no summaries at all: still valid
    const base = rec.toJSON();
    const row = { t: 1, wall: '2026-11-20T10:00:00.000-08:00', kind: 'blink', reported: true, reason: 'ok', blinks_per_min: 14, eyes_closed_pct: 3 };
    expect(validateSession({ ...base, summaries: [row] })).toEqual([]);
    for (const [bad, frag] of [[{ ...row, eyes_closed_pct: 120 }, 'eyes_closed_pct'], [{ ...row, blinks_per_min: -1 }, 'blinks_per_min'], [{ ...row, long_closures: 1.5 }, 'long_closures'],
      [{ ...row, eyes_closed_pct: undefined }, 'reported rows need'], [{ ...row, kind: 'activity', active_min: 1 }, 'reported rows need'], [{ ...row, still_min: -2 }, 'still_min']] as [any, string][]) {
      expect(validateSession({ ...base, summaries: [JSON.parse(JSON.stringify(bad))] }).join('|'), frag).toContain(frag);
    }
  });
});
