// W8 rhythmic hand movement: TS port checks on the shared synthetic fixtures (bit-level parity is in test/parity), gating, params
// validators, export mapping and the UI flow. Synthetic data only. Not a medical device.
import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import params from '../../shared/params.json';
import { analyseTremor, assertTremorParams, forceLowFps, runTremorFixture, tremorErrors, type HandFrame, type TremorParams } from '../src/core/tremor';
import { compareOutputs, type Fixture } from '../src/core/parity';
import { tremorSummary } from '../src/export/tremor';
import { SessionRecorder } from '../src/export/recorder';
import { validateSession } from '../src/export/schema';
import { HandSession } from '../src/capture/handSession';
import { PoseSession } from '../src/capture/poseSession';
import { FaceHrSession } from '../src/ui/faceHr';
import {
  TREMOR_COMFORT, TREMOR_INSTRUCTIONS, TREMOR_NOTE, TREMOR_NO_RHYTHM, buildTremorTest, tremorResultText,
} from '../src/ui/tremorTest';

const TP: TremorParams = params.tremor as TremorParams;
const fx = (name: string): Fixture => JSON.parse(readFileSync(`../shared/fixtures/${name}.json`, 'utf8'));
const run = (input: any): any => runTremorFixture(input, params as any);

describe('tremor params', () => {
  it('validate and reject inconsistent sets (mirror of carelens.params.tremor_errors)', () => {
    expect(tremorErrors(TP)).toEqual([]);
    expect(assertTremorParams(TP)).toBe(TP);
    for (const bad of [{ ...TP, taper_stop_hi_hz: 15.5 }, { ...TP, taper_stop_lo_hz: 3.5 }, { ...TP, motion_band_hi_hz: 2.5 }, { ...TP, min_rate_hz: 24 },
      { ...TP, fs_hz: 20 }, { ...TP, min_window_s: 20 }, { ...TP, edge_trim_s: 3 }, { ...TP, min_prominence: 0 }, { ...TP, noise_floor_pct: 0 }, { ...TP, zero_pad_factor: 0 }]) {
      expect(tremorErrors(bad)).not.toEqual([]);
      expect(() => assertTremorParams(bad)).toThrow();
    }
    expect([TP.taper_stop_lo_hz, TP.band_lo_hz, TP.band_hi_hz, TP.taper_stop_hi_hz]).toEqual([2, 3, 12, 13.5]);
  });
});

describe('tremor fixtures', () => {
  it.each([['tremor_4hz', 4], ['tremor_9hz', 9]])('%s reports the scripted frequency', (name, hz) => {
    const r = run(fx(name).input);
    expect(r.reason).toBe('ok');
    expect(r.reported).toBe(true);
    expect(Math.abs(r.freq_hz - hz)).toBeLessThan(0.3);
    expect(r.amp_pct).toBeGreaterThan(1.5);
    expect(compareOutputs(fx(name), r)).toEqual([]);
  });

  it('a steady hand is no_rhythm with no numbers shown', () => {
    const r = run(fx('tremor_null').input);
    expect(r).toMatchObject({ reason: 'no_rhythm', reported: false, freq_hz: null, amp_pct: null });
  });

  it('20 fps with an 11 Hz movement is low_fps and never an aliased frequency', () => {
    expect(run(fx('tremor_low_fps').input)).toMatchObject({ reason: 'low_fps', reported: false, freq_hz: null, amp_pct: null });
  });

  it('big arm movement is moving', () => {
    expect(run(fx('tremor_moving').input)).toMatchObject({ reason: 'moving', reported: false, freq_hz: null });
  });

  it('hand lost mid-test is too_short; lost at the end only shortens; no frames is no_hand', () => {
    const inp = fx('tremor_4hz').input;
    const lose = (a: number, b: number) => ({ ...inp, landmarks: inp.landmarks.map((l: any, i: number) => (i / 30 >= a && i / 30 < b ? null : l)) });
    expect(run(lose(6, 9)).reason).toBe('too_short');
    const end = run(lose(11, 15));
    expect(end.reason).toBe('ok');
    expect(end.duration_s).toBeLessThan(11.1);
    expect(run(lose(0, 99)).reason).toBe('no_hand');
  });

  it('frames below the hand score do not count', () => {
    const inp = fx('tremor_4hz').input;
    expect(run({ ...inp, score: inp.score.map(() => 0.2) }).reason).toBe('no_hand');
  });
});

describe('result wording', () => {
  const base = { reported: false, freq_hz: null, amp_pct: null, p2p_pct: null, prominence: null, rms_pct: null, motion_pct: null, duration_s: 15, rate_hz: 30, n_frames: 450 } as const;
  it('ok shows frequency and amplitude as % of hand size; no_rhythm is neutral', () => {
    expect(tremorResultText({ ...base, reported: true, reason: 'ok', freq_hz: 6.04, amp_pct: 2.46 }, TP).headline).toBe('Rhythmic hand movement: 6.0 Hz, amplitude 2.5 % of hand size');
    const nr = tremorResultText({ ...base, reason: 'no_rhythm' }, TP);
    expect(nr).toEqual({ headline: 'No clear rhythmic hand movement detected', details: [], withheld: false });
    expect(TREMOR_NO_RHYTHM).toBe('No clear rhythmic hand movement detected');
  });
  it('never uses condition words', () => {
    const all = [TREMOR_INSTRUCTIONS, TREMOR_COMFORT, TREMOR_NOTE, TREMOR_NO_RHYTHM];
    for (const reason of ['no_hand', 'too_short', 'low_fps', 'moving', 'no_rhythm', 'ok'] as const) {
      const t = tremorResultText({ ...base, reason, freq_hz: 5, amp_pct: 2 }, TP);
      all.push(t.headline, ...t.details);
    }
    for (const s of all) expect(s).not.toMatch(/parkinson|diagnos|screening|disease|disorder|patient/i);
    expect(TREMOR_INSTRUCTIONS).toContain('40–60 cm');
    expect(TREMOR_NOTE).toBe('Experimental. Many things affect this — caffeine, tiredness, cold, holding position. Compare with your own earlier results.');
  });
});

describe('export summary', () => {
  it('maps a reported result and a neutral no_rhythm; both validate in a session', () => {
    const rec = new SessionRecorder(['camera'], () => 1_700_000_000_000, 'Chrome');
    rec.addMobilitySummary(tremorSummary(run(fx('tremor_4hz').input)));
    rec.addMobilitySummary(tremorSummary(run(fx('tremor_null').input)));
    const s = rec.toJSON() as any;
    expect(validateSession(s)).toEqual([]);
    expect(s.summaries[0]).toMatchObject({ kind: 'hand_tremor', reported: true, reason: 'ok' });
    expect(Math.abs(s.summaries[0].freq_hz - 4)).toBeLessThan(0.3);
    expect(s.summaries[0].amp_pct).toBeGreaterThan(0);
    expect(s.summaries[1]).toMatchObject({ kind: 'hand_tremor', reported: false, reason: 'no_rhythm' });
    expect(s.summaries[1].freq_hz).toBeUndefined();
    s.summaries[0].freq_hz = -1;
    expect(validateSession(s)).not.toEqual([]);
    delete s.summaries[0].freq_hz; // a reported row needs its headline numbers
    expect(validateSession(s)).not.toEqual([]);
  });
});

// ---- the UI flow (a minimal fake DOM: the project has no jsdom)
class FakeEl {
  attrs: Record<string, string> = {}; kids: (FakeEl | string)[] = []; listeners: Record<string, (() => void)[]> = {};
  hidden = false; disabled = false; className = ''; value = ''; private text = '';
  constructor(readonly tag: string, private readonly reg: Map<string, FakeEl>) {}
  setAttribute(k: string, v: string): void { this.attrs[k] = v; if (k === 'id') this.reg.set(v, this); }
  getAttribute(k: string): string | null { return this.attrs[k] ?? null; }
  removeAttribute(k: string): void { delete this.attrs[k]; }
  append(...c: (FakeEl | string)[]): void { this.kids.push(...c); }
  replaceChildren(...c: (FakeEl | string)[]): void { this.kids = c; this.text = ''; }
  addEventListener(ev: string, fn: () => void): void { (this.listeners[ev] ??= []).push(fn); }
  click(): void { if (!this.disabled) (this.listeners.click ?? []).forEach((f) => f()); }
  get classList() { return { toggle: () => undefined, remove: () => undefined, add: () => undefined }; }
  get textContent(): string { return this.text + this.kids.map((k) => (typeof k === 'string' ? k : k.textContent)).join(''); }
  set textContent(v: string) { this.text = v; this.kids = []; }
}
const reg = new Map<string, FakeEl>();
const $ = (id: string): FakeEl => reg.get(id)!;
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

describe('hand test flow', () => {
  beforeEach(() => { reg.clear(); (globalThis as any).document = { createElement: (t: string) => new FakeEl(t, reg) }; });
  afterEach(() => { delete (globalThis as any).document; });

  function setup(name: string, loadFails = false) {
    const inp = fx(name).input;
    const rows: any[] = [];
    const queue: (() => void)[] = [];
    let cb: ((now: number, meta?: any) => void) | null = null;
    let stopped = false, closed = 0, loads = 0, calls = 0;
    const video = { videoWidth: 640, videoHeight: 480, readyState: 4 } as unknown as HTMLVideoElement;
    const ui = buildTremorTest({
      tp: TP, record: (r) => rows.push(r), getVideo: () => video,
      subscribe: (_v, f) => { cb = f; return () => { stopped = true; }; },
      loadHand: async () => {
        loads++;
        if (loadFails) throw new Error('no model');
        return { detect: () => { const i = Math.min(calls++, inp.landmarks.length - 1); return inp.landmarks[i] ? { lm: inp.landmarks[i], score: inp.score[i] } : null; }, close: () => { closed++; } };
      },
      setTimeout: (fn) => queue.push(fn), clearTimeout: () => { queue.length = 0; },
    });
    let n = 0;
    const frame = (): void => { cb!(n * (1000 / 30), { mediaTime: n / 30 }); n++; };
    return { ui, rows, queue, frame, state: () => ({ stopped, closed, loads }) };
  }

  it('instructions, comfort line, model loads only for the test and closes after; result recorded once', async () => {
    const { rows, queue, frame, state } = setup('tremor_4hz');
    expect(state().loads).toBe(0); // nothing loaded before the test
    $('tm-pick').click();
    expect($('tm-instructions').textContent).toBe(TREMOR_INSTRUCTIONS);
    expect($('tm-comfort').textContent).toBe(TREMOR_COMFORT);
    expect(state().loads).toBe(0);
    $('tm-ready').click();
    await tick();
    expect(state().loads).toBe(1);
    expect($('tm-start').disabled).toBe(true); // no hand seen yet
    for (let i = 0; i < 6; i++) frame();
    expect($('tm-chk-hand').textContent).toContain('in view');
    expect($('tm-start').disabled).toBe(false);
    $('tm-start').click();
    while (queue.length) queue.shift()!();
    expect($('tm-live').hidden).toBe(false);
    for (let i = 0; i < 700 && $('tm-result').hidden; i++) frame();
    expect($('tm-result').hidden).toBe(false);
    expect($('tm-result-value').getAttribute('data-reason')).toBe('ok');
    expect(Math.abs(Number($('tm-result-value').getAttribute('data-freq')) - 4)).toBeLessThan(0.3);
    expect($('tm-result-value').textContent).toMatch(/^Rhythmic hand movement: 4\.0 Hz, amplitude \d\.\d % of hand size$/);
    expect($('tm-note').textContent).toBe(TREMOR_NOTE);
    expect(state().closed).toBe(1); // model closed as soon as the result is shown
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ kind: 'hand_tremor', reported: true, reason: 'ok' });
  });

  it('a steady hand gets the neutral no_rhythm text', async () => {
    const { rows, queue, frame } = setup('tremor_null');
    $('tm-pick').click(); $('tm-ready').click();
    await tick();
    for (let i = 0; i < 6; i++) frame();
    $('tm-start').click();
    while (queue.length) queue.shift()!();
    for (let i = 0; i < 700 && $('tm-result').hidden; i++) frame();
    expect($('tm-result-value').textContent).toBe('No clear rhythmic hand movement detected');
    expect(rows[0]).toMatchObject({ kind: 'hand_tremor', reported: false, reason: 'no_rhythm' });
  });

  it('Stop and disable() abandon the test: model closed, nothing recorded', async () => {
    const a = setup('tremor_4hz');
    $('tm-pick').click(); $('tm-ready').click();
    await tick();
    for (let i = 0; i < 6; i++) a.frame();
    $('tm-place-back').click();
    expect(a.state().closed).toBe(1);
    expect($('tm-safety').hidden).toBe(false);
    $('tm-ready').click();
    await tick();
    for (let i = 0; i < 6; i++) a.frame();
    a.ui.disable('The camera is off.');
    expect(a.state().closed).toBe(2);
    expect($('tm-intro').hidden).toBe(false);
    expect($('tm-disabled').hidden).toBe(false);
    expect($('tm-pick').disabled).toBe(true);
    expect(a.rows).toHaveLength(0);
  });

  it('a model that fails to load returns to the start with a message', async () => {
    const { frame: _f } = setup('tremor_4hz', true);
    $('tm-pick').click(); $('tm-ready').click();
    await tick();
    expect($('tm-intro').hidden).toBe(false);
    expect($('tm-disabled').textContent).toContain('failed to load');
  });
});

describe('analyseTremor input handling', () => {
  it('works on HandFrame objects with a different aspect ratio', () => {
    const inp = fx('tremor_4hz').input;
    const frames: HandFrame[] = inp.landmarks.map((lm: number[][], i: number) => ({ t: i / inp.fps, lm, score: inp.score[i] }));
    const r = analyseTremor(frames, 1.7778, TP); // a different aspect only rescales x: the result stays a rhythm near 4 Hz
    expect(r.reason).toBe('ok');
    expect(Math.abs(r.freq_hz! - 4)).toBeLessThan(0.3);
  });
});

// ---- distinct camera frames only (a repeated display callback must never count as a sample)
describe('hand session sample clock', () => {
  const tmpl: number[][] = (fx('tremor_4hz').input.landmarks[0] as number[][]);
  /** a 15 fps camera showing an 11 Hz movement (x displacement, normalised units) */
  const cameraFrame = (k: number): number[][] => tmpl.map((p) => [p[0]! + 0.012 * Math.sin(2 * Math.PI * 11 * (k / 15)), p[1]!]);

  function rig(opts: { counter: boolean; identical: boolean }) {
    const frames: HandFrame[] = [];
    let cb: (now: number, meta?: any) => void = () => {};
    let camFrame = 0;
    const video: any = { videoWidth: 640, videoHeight: 480, readyState: 4 };
    if (opts.counter) video.getVideoPlaybackQuality = () => ({ totalVideoFrames: camFrame + 1 });
    const session = new HandSession((_v, c) => { cb = c; return () => {}; }, (f) => frames.push(f));
    const handle = {
      detect: () => {
        const lm = cameraFrame(camFrame);
        // without an identical-landmarks signal the model output differs a little on every call (it re-tracks the same image)
        return { lm: opts.identical ? lm : lm.map((p) => [p[0]! + 1e-7 * Math.random(), p[1]!]), score: 1 };
      },
      close: vi.fn(),
    };
    const drive = async (seconds: number) => {
      await session.load(async () => handle, video);
      for (let i = 0; i < seconds * 60; i++) { camFrame = Math.floor(i / 4); cb(1000 + i * (1000 / 60)); } // rAF mode: 60 Hz callbacks, no mediaTime
    };
    return { session, frames, drive };
  }

  it.each([['playback-quality frame counter', { counter: true, identical: true }], ['identical-landmark check', { counter: false, identical: true }]])(
    '15 fps camera polled at 60 Hz with an 11 Hz movement: %s keeps only distinct frames -> low_fps, never an aliased ~4 Hz', async (_n, opts) => {
      const r = rig(opts);
      await r.drive(16);
      expect(r.frames.length).toBeLessThan(16 * 16); // ~15 per s, not 60
      const res = analyseTremor(r.frames, 4 / 3, TP);
      expect(res.reason).toBe('low_fps');
      expect(res.freq_hz).toBeNull();
      expect(res.rate_hz!).toBeLessThan(TP.min_rate_hz);
    });

  it('with no way to tell frames apart the session says its timing is unverified, and the result is forced to low_fps', async () => {
    const r = rig({ counter: false, identical: false });
    await r.drive(16);
    expect(r.session.timingReliable).toBe(false);
    const raw = analyseTremor(r.frames, 4 / 3, TP);
    expect(raw.rate_hz!).toBeGreaterThan(TP.min_rate_hz); // duplicates look like a fast camera ...
    expect(forceLowFps(raw)).toMatchObject({ reason: 'low_fps', reported: false, freq_hz: null }); // ... so the result is withheld
    expect(forceLowFps({ ...raw, reason: 'too_short' })).toMatchObject({ reason: 'too_short' });
  });

  it('mediaTime frames: a repeated mediaTime is skipped', async () => {
    const frames: HandFrame[] = [];
    let cb: (now: number, meta?: any) => void = () => {};
    const s = new HandSession((_v, c) => { cb = c; return () => {}; }, (f) => frames.push(f));
    await s.load(async () => ({ detect: () => ({ lm: tmpl, score: 1 }), close: () => {} }), { videoWidth: 640, videoHeight: 480, readyState: 4 } as any);
    for (let i = 0; i < 40; i++) cb(1000 + i * 16, { mediaTime: (i < 2 ? i : Math.floor(i / 4) + 1) / 15 });
    expect(frames.length).toBeLessThanOrEqual(10);
    expect(s.timingReliable).toBe(true);
  });
});

describe('hand test and the other models', () => {
  it('pose and face sessions hold their frame loops while paused and restart on resume', async () => {
    const video = { videoWidth: 640, videoHeight: 480, readyState: 4 } as HTMLVideoElement;
    let subs = 0, stops = 0, cb: (n: number) => void = () => {};
    const subscribe = (_v: HTMLVideoElement, c: (n: number) => void) => { subs++; cb = c; return () => { stops++; }; };
    const detect = vi.fn(() => null);
    const pose = new PoseSession(subscribe, { onView: () => {}, onEvent: () => {} }, () => ({ size: () => ({ w: 1, h: 1 }), shiftAndAdvance: () => 'first' as const, reset: () => {} }) as any);
    await pose.load(async () => ({ detect, close: () => {} }) as any, video);
    expect(subs).toBe(1);
    pose.setPaused(true);
    expect(stops).toBe(1);
    cb(2000); cb(2100);
    expect(detect).not.toHaveBeenCalled();
    pose.setPaused(true); // idempotent
    pose.setPaused(false);
    expect(subs).toBe(2);
    pose.end();

    (globalThis as any).document = { createElement: () => ({ getContext: () => null }) }; // FaceRoiSampler only needs canvases to exist
    const faceDetect = vi.fn(() => null);
    const outlines: unknown[] = [];
    const face = new FaceHrSession(subscribe, { onUpdate: () => {}, onOutline: (...a: unknown[]) => outlines.push(a) } as any);
    await face.load(async () => ({ detect: faceDetect, close: () => {}, delegate: 'x' }) as any, video);
    cb(3000); cb(3033); cb(3066);
    const before = faceDetect.mock.calls.length;
    face.setPaused(true);
    cb(3100); cb(3133); cb(3166);
    expect(faceDetect.mock.calls.length).toBe(before);
    face.setPaused(false);
    cb(4000); cb(4033); cb(4066);
    expect(faceDetect.mock.calls.length).toBeGreaterThan(before);
    face.end();
    delete (globalThis as any).document;
  });

  beforeEach(() => { reg.clear(); (globalThis as any).document = { createElement: (t: string) => new FakeEl(t, reg) }; });
  afterEach(() => { delete (globalThis as any).document; });

  function uiRig(loadFailsFirst = false, declared: number | null = 30, reliable = true) {
    const inp = fx('tremor_4hz').input;
    const active: boolean[] = [];
    const queue: (() => void)[] = [];
    let cb: ((now: number, meta?: any) => void) | null = null;
    let calls = 0, loads = 0;
    const video = { videoWidth: 640, videoHeight: 480, readyState: 4 } as unknown as HTMLVideoElement;
    const ui = buildTremorTest({
      tp: TP, record: () => {}, getVideo: () => video, onActive: (a) => active.push(a), getDeclaredFps: () => declared,
      subscribe: (_v, f) => { cb = f; return () => {}; },
      loadHand: async () => {
        if (loadFailsFirst && loads++ === 0) throw new Error('no model');
        return { detect: () => { const i = Math.min(calls++, inp.landmarks.length - 1); return { lm: inp.landmarks[i].map((p: number[]) => (reliable ? p : [p[0]! + 1e-9 * calls, p[1]!])), score: 1 }; }, close: () => {} };
      },
      setTimeout: (fn) => queue.push(fn), clearTimeout: () => { queue.length = 0; },
    });
    let n = 0;
    const frame = (): void => { cb!(n * (1000 / 30), reliable ? { mediaTime: n / 30 } : undefined); n++; };
    return { ui, active, queue, frame };
  }

  it('onActive is true from placement to the result (pose/face/movement held) and false before and after', async () => {
    const { active, queue, frame } = uiRig();
    $('tm-pick').click();
    expect(active).toEqual([]);
    $('tm-ready').click();
    await tick();
    expect(active).toEqual([true]);
    for (let i = 0; i < 6; i++) frame();
    $('tm-start').click();
    while (queue.length) queue.shift()!();
    for (let i = 0; i < 700 && $('tm-result').hidden; i++) frame();
    expect(active).toEqual([true, false]);
    expect($('tm-result-value').getAttribute('data-reason')).toBe('ok');
    $('tm-again').click();
    $('tm-ready').click();
    await tick();
    $('tm-place-back').click(); // leaving mid-way resumes the others too
    expect(active).toEqual([true, false, true, false]);
  });

  it('a load error is cleared by a later successful load', async () => {
    const { ui } = uiRig(true);
    $('tm-pick').click(); $('tm-ready').click();
    await tick();
    expect($('tm-disabled').hidden).toBe(false);
    expect($('tm-disabled').textContent).toContain('failed to load');
    $('tm-pick').click(); $('tm-ready').click();
    await tick();
    expect($('tm-disabled').hidden).toBe(true);
    expect($('tm-load').textContent).toBe('');
    ui.disable('The camera is off.'); // a permanent reason is not cleared by a later open
    expect($('tm-disabled').hidden).toBe(false);
  });

  it('unverifiable frame timing plus an unknown or slow declared frame rate withholds as low_fps; a fast declared rate analyses', async () => {
    for (const [declared, reason] of [[null, 'low_fps'], [15, 'low_fps'], [30, 'ok']] as const) {
      reg.clear();
      const { queue, frame } = uiRig(false, declared, false);
      $('tm-pick').click(); $('tm-ready').click();
      await tick();
      for (let i = 0; i < 6; i++) frame();
      $('tm-start').click();
      while (queue.length) queue.shift()!();
      for (let i = 0; i < 700 && $('tm-result').hidden; i++) frame();
      expect($('tm-result-value').getAttribute('data-reason'), String(declared)).toBe(reason);
    }
  });
});
