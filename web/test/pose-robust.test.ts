import { describe, expect, it, vi } from 'vitest';
import type { PoseHandle } from '../src/capture/pose';
import { MAX_CONSECUTIVE_ERRORS, PoseSession, type PoseView } from '../src/capture/poseSession';
import { onVideoFrame } from '../src/capture/frames';
import { PosePipeline } from '../src/core/pipeline';
import { faceGuidanceText } from '../src/ui/faceHr';
import { seedAndSubscribe } from '../src/ui/controls';
import params from '../../shared/params.json';

const FRAME_MS = 1000 / 30;

function standing(): number[][] {
  const lm = Array.from({ length: 33 }, () => [0, 0, 0.2]);
  const set = (k: number, x: number, y: number) => { lm[k] = [x / (4 / 3), y, 0.95]; };
  set(0, 0.5, 0.25); set(11, 0.57, 0.33); set(12, 0.43, 0.33); set(23, 0.55, 0.55); set(24, 0.45, 0.55);
  set(27, 0.55, 0.9); set(28, 0.45, 0.9);
  return lm;
}

function rig(chestMode: { v: 'sine' | 'flat' } = { v: 'sine' }) {
  const video = { videoWidth: 640, videoHeight: 480, readyState: 4 } as HTMLVideoElement;
  let cb: (n: number) => void = () => {};
  const subscribe = (_v: HTMLVideoElement, c: (n: number) => void) => { cb = c; return () => {}; };
  const handle = { detect: vi.fn(() => standing()), close: vi.fn() };
  const views: PoseView[] = [];
  const errors: boolean[] = [];
  let prev = 0, k = 0;
  const chest = {
    size: () => ({ w: 320, h: 240 }),
    shiftAndAdvance: () => {
      const y = chestMode.v === 'sine' ? 0.8 * Math.sin((2 * Math.PI * (14 / 60) * k++) / 7.5) : 0;
      const d = y - prev; prev = y;
      return k === 1 ? ('first' as const) : d;
    },
    reset: () => { prev = 0; k = 0; },
  };
  const loader = vi.fn(async () => ({ detect: handle.detect, close: handle.close }) as PoseHandle);
  const session = new PoseSession(subscribe, { onView: (v) => views.push(v), onEvent: () => {}, onError: (f) => errors.push(f) }, () => chest, undefined);
  let now = 1000;
  const run = (seconds: number) => { for (let i = 0; i < seconds * 30; i++) { now += FRAME_MS; cb(now); } };
  return { video, session, handle, views, errors, loader, run, jump: (ms: number) => { now += ms; }, get now() { return now; } };
}

describe('stale breathing reading', () => {
  it('is cleared after a 10-minute hidden tab (timestamp jump = gap)', async () => {
    const r = rig();
    await r.session.load(r.loader, r.video);
    r.run(70);
    expect(r.views.at(-1)!.breathing.estimate?.reported).toBe(true);
    r.jump(10 * 60 * 1000);
    r.run(0.5);
    expect(r.views.at(-1)!.breathing.estimate).toBeNull();
  });
  it('is dropped when a due estimate fails (flat trace) instead of being kept', async () => {
    const mode = { v: 'sine' as 'sine' | 'flat' };
    const r = rig(mode);
    await r.session.load(r.loader, r.video);
    r.run(70);
    expect(r.views.at(-1)!.breathing.estimate).not.toBeNull();
    mode.v = 'flat';
    r.run(45); // the whole 30 s window is flat now
    expect(r.views.at(-1)!.breathing.estimate).toBeNull();
  });
  it('is cleared when the frame aspect changes (rotation) and detectors do not see a motion spike', async () => {
    const r = rig();
    await r.session.load(r.loader, r.video);
    r.run(70);
    expect(r.views.at(-1)!.breathing.estimate).not.toBeNull();
    (r.video as { videoWidth: number }).videoWidth = 480;
    (r.video as { videoHeight: number }).videoHeight = 640;
    r.run(0.5);
    expect(r.views.at(-1)!.breathing.estimate).toBeNull();
    expect(r.views.at(-1)!.state.motion).toBeLessThan(params.common.still_motion_bodysizes_s);
  });
});

describe('PosePipeline aspect change', () => {
  it('rebuilds features with the new aspect: no spurious motion, presence kept, flag raised once', () => {
    const p = new PosePipeline(params as never);
    let last;
    for (let i = 0; i < 20; i++) last = p.step(i / 8, standing(), 4 / 3);
    expect(last!.aspectChanged).toBe(false);
    const a = p.step(20 / 8, standing(), 3 / 4);
    expect(a.aspectChanged).toBe(true);
    expect(a.state.present).toBe(true);
    expect(a.state.motion).toBeLessThan(params.common.still_motion_bodysizes_s);
    expect(p.step(21 / 8, standing(), 3 / 4).aspectChanged).toBe(false);
  });
});

describe('frame loop survives errors', () => {
  it('onVideoFrame reschedules even when the handler throws', () => {
    const cbs: ((n: number, m?: unknown) => void)[] = [];
    const video = { requestVideoFrameCallback: (c: (n: number, m?: unknown) => void) => { cbs.push(c); return 1; } } as unknown as HTMLVideoElement;
    const handler = vi.fn(() => { throw new Error('boom'); });
    onVideoFrame(video, handler);
    expect(cbs.length).toBe(1);
    expect(() => cbs[0]!(1)).toThrow('boom');
    expect(cbs.length).toBe(2); // rescheduled
  });
  it('a throwing landmarker is retried, surfaced, re-created after N failures and the period counts as a gap', async () => {
    const r = rig();
    await r.session.load(r.loader, r.video);
    r.run(2);
    const okViews = r.views.length;
    r.handle.detect.mockImplementation(() => { throw new Error('gpu lost'); });
    r.run(3); // 8 Hz * 3 s = well over N failures
    expect(r.errors.includes(true)).toBe(true);
    expect(r.views.length).toBe(okViews); // no views from failed frames
    expect(r.loader.mock.calls.length).toBeGreaterThanOrEqual(2); // re-created
    expect(r.handle.close).toHaveBeenCalled();
    expect(MAX_CONSECUTIVE_ERRORS).toBe(10);
    r.handle.detect.mockImplementation(() => standing()); // the new landmarker works
    await Promise.resolve();
    r.run(1);
    expect(r.views.length).toBeGreaterThan(okViews);
    expect(r.errors.at(-1)).toBe(false);
    expect(r.views.at(-1)!.breathing.estimate).toBeNull(); // reset by the gap
  });
});

describe('face card guidance is derived', () => {
  it('priority: failed > off > loading > reason', () => {
    expect(faceGuidanceText('loading', true, null)).toBe('Loading the face model…');
    expect(faceGuidanceText('ready', true, null)).toBe('Face the camera');
    expect(faceGuidanceText('ready', true, 'motion')).toBe('Hold still');
    expect(faceGuidanceText('ready', false, 'motion')).toBe('Face heart rate is off');
    expect(faceGuidanceText('loading', false, null)).toBe('Face heart rate is off');
    expect(faceGuidanceText('failed', false, 'ok', 'x')).toBe('Face model failed to load: x');
    // a reason arriving while loading/off must not erase the state; turning back on shows the current reason again
    expect(faceGuidanceText('ready', true, 'no_face')).toBe('Face the camera');
  });
});

describe('controls changed before the session exists', () => {
  it('seedAndSubscribe applies the current value immediately and on every change', () => {
    let value = 300;
    let handler: () => void = () => {};
    const applied: number[] = [];
    seedAndSubscribe(() => value, (h) => { handler = h; }, (v) => applied.push(v));
    expect(applied).toEqual([300]); // the user's choice made during startup is honoured
    value = 60; handler();
    expect(applied).toEqual([300, 60]);
  });
});
