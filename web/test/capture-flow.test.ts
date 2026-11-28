import { describe, expect, it, vi } from 'vitest';
import type { PoseHandle } from '../src/capture/pose';
import { PoseSession, type PoseView } from '../src/capture/poseSession';
import { buildConstraints, openStreamWithFallback } from '../src/capture/session';
import { Consent } from '../src/ui/consent';

const started = (camera: boolean, mic: boolean) => {
  const c = new Consent();
  c.setCamera(camera);
  c.setMic(mic);
  c.start();
  return c;
};
const err = (name: string) => Object.assign(new Error(name), { name });

describe('mic constraints', () => {
  it('always disables audio processing, with or without deviceId', () => {
    const off = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
    expect(buildConstraints({ camera: false, mic: true })!.audio).toEqual(off);
    expect(buildConstraints({ camera: false, mic: true }, { audioDeviceId: 'm1' })!.audio).toEqual({
      ...off,
      deviceId: { exact: 'm1' },
    });
  });
});

describe('stale saved deviceId', () => {
  it('retries once without the id and reports what was dropped', async () => {
    const gum = vi.fn().mockRejectedValueOnce(err('OverconstrainedError')).mockResolvedValueOnce({} as MediaStream);
    const dropped = vi.fn();
    const r = await openStreamWithFallback(started(true, false), { getUserMedia: gum }, { videoDeviceId: 'old' }, dropped);
    expect(gum).toHaveBeenCalledTimes(2);
    expect((gum.mock.calls[1]![0].video as MediaTrackConstraints).deviceId).toBeUndefined();
    expect(dropped).toHaveBeenCalledWith(['videoDeviceId']);
    expect(r.choice.videoDeviceId).toBeUndefined();
  });
  it('does not retry on permission denial or when there is no saved id', async () => {
    const denied = vi.fn().mockRejectedValue(err('NotAllowedError'));
    await expect(openStreamWithFallback(started(true, false), { getUserMedia: denied }, { videoDeviceId: 'x' })).rejects.toThrow();
    expect(denied).toHaveBeenCalledTimes(1);
    const nf = vi.fn().mockRejectedValue(err('NotFoundError'));
    await expect(openStreamWithFallback(started(true, false), { getUserMedia: nf }, {})).rejects.toThrow();
    expect(nf).toHaveBeenCalledTimes(1);
  });
  it('retries at most once', async () => {
    const gum = vi.fn().mockRejectedValue(err('NotFoundError'));
    await expect(openStreamWithFallback(started(true, false), { getUserMedia: gum }, { videoDeviceId: 'x' })).rejects.toThrow();
    expect(gum).toHaveBeenCalledTimes(2);
  });
});

function harness(over: { present?: boolean; nowMs?: () => number } = {}) {
  const video = { videoWidth: 640, videoHeight: 480, readyState: 4 } as HTMLVideoElement;
  const subs: { cb: (n: number) => void; stopped: boolean }[] = [];
  const subscribe = vi.fn((_v: HTMLVideoElement, cb: (n: number) => void) => {
    const s = { cb, stopped: false };
    subs.push(s);
    return () => {
      s.stopped = true;
    };
  });
  const handle = { detect: vi.fn(() => (over.present === false ? null : skeletonStanding())), close: vi.fn() } satisfies PoseHandle;
  const views: PoseView[] = [];
  const events: unknown[] = [];
  const chest = { size: () => ({ w: 320, h: 240 }), shiftAndAdvance: () => 0, reset: () => {} };
  const session = new PoseSession(subscribe, { onView: (v) => views.push(v), onEvent: (e) => events.push(e) }, () => chest, over.nowMs);
  return { video, subs, subscribe, handle, views, events, session };
}

function skeletonStanding(): number[][] {
  const lm = Array.from({ length: 33 }, () => [0, 0, 0.2]);
  const set = (k: number, x: number, y: number) => { lm[k] = [x / (4 / 3), y, 0.95]; };
  set(0, 0.5, 0.25); set(11, 0.57, 0.33); set(12, 0.43, 0.33); set(23, 0.55, 0.55); set(24, 0.45, 0.55);
  set(27, 0.55, 0.9); set(28, 0.45, 0.9);
  return lm;
}

describe('pose session', () => {
  it('keeps producing results after restart (stream swap)', async () => {
    const h = harness();
    await h.session.load(async () => h.handle, h.video);
    h.subs[0]!.cb(1000); h.subs[0]!.cb(1033); // the first frame only fixes the clock source
    const n = h.views.length;
    h.session.restart(); // e.g. camera switched
    expect(h.subs[0]!.stopped).toBe(true);
    h.subs[1]!.cb(5000); h.subs[1]!.cb(5033);
    expect(h.views.length).toBe(n + 1);
    expect(h.handle.close).not.toHaveBeenCalled();
  });
  it('Stop during model load closes the late model and never starts a loop', async () => {
    const h = harness();
    let resolve!: (p: PoseHandle) => void;
    const p = h.session.load(() => new Promise<PoseHandle>((r) => (resolve = r)), h.video);
    h.session.end();
    resolve(h.handle);
    await p;
    expect(h.handle.close).toHaveBeenCalledTimes(1);
    expect(h.subscribe).not.toHaveBeenCalled();
  });
  it('end() stops the loop and closes the model; restart after end is a no-op', async () => {
    const h = harness();
    await h.session.load(async () => h.handle, h.video);
    h.session.end();
    expect(h.subs[0]!.stopped).toBe(true);
    expect(h.handle.close).toHaveBeenCalledTimes(1);
    h.session.restart();
    expect(h.subscribe).toHaveBeenCalledTimes(1);
  });
  it('ignores frames with no video size', async () => {
    const h = harness();
    (h.video as { videoWidth: number }).videoWidth = 0;
    await h.session.load(async () => h.handle, h.video);
    h.subs[0]!.cb(1);
    h.subs[0]!.cb(40);
    expect(h.views).toEqual([]);
  });
  it('runs pose at the scheduled rate, not on every presented frame', async () => {
    const h = harness({ present: false });
    await h.session.load(async () => h.handle, h.video);
    for (let i = 0; i < 300; i++) h.subs[0]!.cb(1000 + i * (1000 / 30)); // 10 s at 30 fps
    expect(h.handle.detect.mock.calls.length).toBeGreaterThan(60);
    expect(h.handle.detect.mock.calls.length).toBeLessThan(85);
    expect(h.views.at(-1)!.state.present).toBe(false);
  });
  it('a slow inference skips the next pose frame', async () => {
    let now = 0;
    const h = harness({ nowMs: () => now });
    h.handle.detect.mockImplementation(() => { now += 200; return skeletonStanding(); }); // way over budget
    await h.session.load(async () => h.handle, h.video);
    for (let i = 0; i < 90; i++) h.subs[0]!.cb(1000 + i * (1000 / 30));
    // 3 s at 30 fps is ~24 due slots at 8 Hz; every other one is skipped
    expect(h.handle.detect.mock.calls.length).toBeLessThan(16);
    expect(h.handle.detect.mock.calls.length).toBeGreaterThan(8);
  });
});
