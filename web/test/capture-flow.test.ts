import { describe, expect, it, vi } from 'vitest';
import { PresenceSession, type PoseHandle } from '../src/capture/presence';
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

function harness() {
  const video = { videoWidth: 640 } as HTMLVideoElement;
  const subs: { cb: (n: number) => void; stopped: boolean }[] = [];
  const subscribe = vi.fn((_v: HTMLVideoElement, cb: (n: number) => void) => {
    const s = { cb, stopped: false };
    subs.push(s);
    return () => {
      s.stopped = true;
    };
  });
  const handle = { detect: vi.fn(() => true), close: vi.fn() } satisfies PoseHandle;
  const results: boolean[] = [];
  const session = new PresenceSession(subscribe, (p) => results.push(p), 0);
  return { video, subs, subscribe, handle, results, session };
}

describe('presence session', () => {
  it('keeps producing results after restart (stream swap)', async () => {
    const h = harness();
    await h.session.load(async () => h.handle, h.video);
    h.subs[0]!.cb(1);
    h.session.restart(); // e.g. camera switched
    expect(h.subs[0]!.stopped).toBe(true);
    h.subs[1]!.cb(2);
    expect(h.results).toEqual([true, true]);
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
    expect(h.results).toEqual([]);
  });
});
