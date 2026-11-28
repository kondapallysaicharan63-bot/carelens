import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import params from '../../shared/params.json';
import { AudioSession, type AudioDeps, type AudioHooks } from '../src/capture/audioSession';
import { LabelEventTracker, detectLabelEvents, type AudioParams, type BreathSoundUpdate } from '../src/core/audio';
import { AudioCardState } from '../src/ui/audioCards';

const P = params.audio as AudioParams;

// ---------------------------------------------------------------- fakes
class FakeNode {
  port = { onmessage: null as ((e: { data: Float32Array }) => void) | null, close: vi.fn() };
  connect(n: unknown) { return n; }
  disconnect() { /* no-op */ }
}
class FakeCtx {
  state = 'running';
  closed = false;
  nodes: FakeNode[] = [];
  modules = 0;
  destination = {};
  audioWorklet = { addModule: vi.fn(async () => { this.modules++; }) };
  resume: () => Promise<void> = async () => { this.state = 'running'; };
  constructor(public sampleRate: number, private rejectSource = false) {}
  createGain() { return { gain: { value: 1 }, connect: (n: unknown) => n }; }
  createMediaStreamSource() {
    if (this.rejectSource) throw Object.assign(new Error('rate mismatch'), { name: 'NotSupportedError' });
    return new FakeNode();
  }
  close = vi.fn(async () => { this.closed = true; });
}
const track = { readyState: 'live' as string };
const stream = { getAudioTracks: () => [track] } as unknown as MediaStream;

function makeHooks() {
  return {
    onLevel: vi.fn(), onBreath: vi.fn(), onCough: vi.fn(), onNoMic: vi.fn(), onYamnet: vi.fn(), onSounds: vi.fn(),
  } satisfies Record<keyof AudioHooks, unknown>;
}
function makeDeps(ctxs: FakeCtx[], clock = { ms: 0 }) {
  const nodes: FakeNode[] = [];
  const created: (number | undefined)[] = [];
  let i = 0;
  const deps: AudioDeps = {
    createContext: (rate) => { created.push(rate); return ctxs[i++] as unknown as AudioContext; },
    createWorkletNode: () => { const n = new FakeNode(); nodes.push(n); return n as unknown as AudioWorkletNode; },
    nowMs: () => clock.ms,
  };
  return { deps, nodes, created };
}

beforeEach(() => { vi.useFakeTimers(); vi.stubGlobal('MediaStream', class { constructor(public tracks: unknown[]) {} }); track.readyState = 'live'; });
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('AudioSession.start', () => {
  it('emits its initial onNoMic(false) and runs; the watchdog reports a dead and a recovered mic', async () => {
    const clock = { ms: 0 };
    const ctx = new FakeCtx(16000);
    const { deps, nodes } = makeDeps([ctx], clock);
    const h = makeHooks();
    const s = new AudioSession(h as unknown as AudioHooks, deps);
    expect(await s.start(stream)).toBe(true);
    expect(h.onNoMic).toHaveBeenCalledWith(false);
    clock.ms = 2000; vi.advanceTimersByTime(500);
    expect(h.onNoMic).toHaveBeenLastCalledWith(true);
    nodes[0]!.port.onmessage!({ data: new Float32Array(2048).fill(0.1) }); // blocks resume
    vi.advanceTimersByTime(500);
    expect(h.onNoMic).toHaveBeenLastCalledWith(false);
    expect(h.onLevel).toHaveBeenCalled();
    s.end();
    expect(ctx.closed).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('end() during a deferred resume(): no interval, no node, context closed, start resolves false, YAMNet skipped', async () => {
    const ctx = new FakeCtx(16000);
    ctx.state = 'suspended';
    let release!: () => void;
    ctx.resume = () => new Promise<void>((r) => { release = r; });
    const { deps, nodes } = makeDeps([ctx]);
    const s = new AudioSession(makeHooks() as unknown as AudioHooks, deps);
    const started = s.start(stream);
    await vi.advanceTimersByTimeAsync(0);
    s.end();
    release();
    expect(await started).toBe(false);
    expect(nodes.length).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
    expect(ctx.closed).toBe(true);
    const loader = vi.fn();
    await s.loadYamnet(loader);
    expect(loader).not.toHaveBeenCalled();
  });

  it('end() while addModule is pending also leaves nothing running', async () => {
    const ctx = new FakeCtx(16000);
    let release!: () => void;
    ctx.audioWorklet.addModule = vi.fn(() => new Promise<void>((r) => { release = r; }));
    const { deps, nodes } = makeDeps([ctx]);
    const s = new AudioSession(makeHooks() as unknown as AudioHooks, deps);
    const started = s.start(stream);
    await vi.advanceTimersByTimeAsync(0);
    s.end();
    release();
    expect(await started).toBe(false);
    expect(nodes.length).toBe(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('Firefox: a 16 kHz context that cannot take the stream is closed and retried at the default rate with the resampler', async () => {
    const c16 = new FakeCtx(16000, true), c48 = new FakeCtx(48000);
    const { deps, nodes, created } = makeDeps([c16, c48]);
    const h = makeHooks();
    const s = new AudioSession(h as unknown as AudioHooks, deps);
    expect(await s.start(stream)).toBe(true);
    expect(created).toEqual([16000, undefined]);
    expect(c16.closed).toBe(true);
    expect(c48.closed).toBe(false);
    expect(c48.modules).toBe(1); // the worklet module is added to the new context
    const tone = Float32Array.from({ length: 4800 }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 440 * i) / 48000));
    nodes[nodes.length - 1]!.port.onmessage!({ data: tone });
    expect(h.onLevel).toHaveBeenCalledTimes(1);
    const [rms] = h.onLevel.mock.calls[0]!;
    expect(rms).toBeGreaterThan(0.25); // resampled tone keeps its level (0.5/sqrt2 = 0.35)
    s.end();
  });

  it('other failures reject and close the context', async () => {
    const ctx = new FakeCtx(16000);
    ctx.audioWorklet.addModule = vi.fn(async () => { throw new Error('blocked'); });
    const { deps } = makeDeps([ctx]);
    const s = new AudioSession(makeHooks() as unknown as AudioHooks, deps);
    await expect(s.start(stream)).rejects.toThrow('blocked');
    expect(ctx.closed).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('AudioSession YAMNet wiring', () => {
  const run = async (classify: () => Record<string, number>) => {
    const ctx = new FakeCtx(16000);
    const { deps, nodes } = makeDeps([ctx]);
    const h = makeHooks();
    const s = new AudioSession(h as unknown as AudioHooks, deps);
    await s.start(stream);
    const handle = { classify: vi.fn(classify), close: vi.fn() };
    await s.loadYamnet(async () => handle);
    for (let i = 0; i < 20; i++) nodes[0]!.port.onmessage!({ data: new Float32Array(2048) }); // 40960 samples = 2.56 s
    return { s, h, handle };
  };
  it('classifies every 0.48 s once the first window is full and reports the cough count', async () => {
    let n = 0;
    const { h, handle, s } = await run(() => ({ Cough: n++ === 0 ? 0.9 : 0.0, Speech: 0 }));
    expect(handle.classify.mock.calls.length).toBe(3); // at 1.455 s, 1.935 s, 2.415 s
    expect(h.onCough.mock.calls.at(-1)![0].count).toBe(1);
    s.end();
  });
  it('W10: asks for the snoring / speech / distress labels with the window end time and reports the sound view', async () => {
    let n = 0;
    const { h, handle, s } = await run(() => ({ Cough: 0, Speech: 0.8, Snoring: 0, Screaming: n++ < 2 ? 0.9 : 0, 'Crying, sobbing': 0, Glass: 0, 'Thump, thud': 0 }));
    const [, labels, tEnd] = handle.classify.mock.calls[0] as unknown as [unknown, string[], number];
    expect(labels).toEqual(['Cough', 'Speech', 'Snoring', 'Screaming', 'Crying, sobbing', 'Glass', 'Thump, thud']);
    expect(tEnd).toBeCloseTo(1.455, 3);
    const v = h.onSounds.mock.calls.at(-1)![0];
    expect(v.distress.map((d: { label: string }) => d.label)).toEqual(['Screaming', 'Crying, sobbing', 'Glass', 'Thump, thud']);
    expect(v.distress[0].view.count).toBe(1);
    expect(v.talking.count).toBe(1);
    expect(v.talking.total_s).toBeGreaterThan(0);
    expect(v.pauses.count).toBe(0);
    s.end();
  });
  it('closes the handle on a classify failure and stops classifying', async () => {
    const { h, handle, s } = await run(() => { throw new Error('boom'); });
    expect(handle.close).toHaveBeenCalledTimes(1);
    expect(handle.classify).toHaveBeenCalledTimes(1);
    expect(h.onYamnet).toHaveBeenLastCalledWith('failed', 'boom');
    s.end();
    expect(handle.close).toHaveBeenCalledTimes(1); // not closed twice
  });
});

describe('audio card state', () => {
  const ok = (bpm: number): BreathSoundUpdate => ({ t: 30, bpm, quality: 0.8, reported: true, reason: 'ok', snr: 10, bumps: 8, alt: null, range_log10: 1, clip_frac: 0, speech_frac: null, span_s: 30 });
  it('reset clears a stuck "no microphone" state and a stale rate', () => {
    const st = new AudioCardState();
    st.setBreath(ok(15));
    expect(st.breath.text).toBe('15 breaths/min');
    st.setNoMic(true);
    expect(st.breath.reason).toBe('no_mic');
    expect(st.breath.text).toBeNull();
    st.reset();
    expect(st.breath).toMatchObject({ text: null, quality: null, reason: 'idle' });
  });
  it('reset clears the cough count, last-detected time and timeline', () => {
    const st = new AudioCardState();
    st.setCough({ count: 2, last: { t_start: 10, t_end: 11, peak_score: 0.9 }, ticks: [3, 10], t: 12 }, 100_000);
    expect(st.count).toBe(2);
    expect(st.lastWallMs).toBe(100_000 - 2000);
    st.reset();
    expect([st.count, st.lastWallMs, st.ticks, st.t]).toEqual([0, null, [], 0]);
  });
});

describe('LabelEventTracker == detectLabelEvents', () => {
  const dir = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'shared', 'fixtures');
  const check = (frames: [number, number][], every: number) => {
    const tr = new LabelEventTracker(P.cough, 1_000_000);
    frames.forEach((f, i) => {
      tr.push(f[0], f[1]);
      if (i % every !== 0 && i !== frames.length - 1) return;
      const b = detectLabelEvents(frames.slice(0, i + 1), P.cough);
      const v = tr.view();
      expect(v.count).toBe(b.count);
      expect(v.last).toEqual(b.events.at(-1) ?? null);
      expect(v.ticks).toEqual(b.events.map((e) => e.t_start));
    });
  };
  for (const f of readdirSync(dir).filter((n) => n.startsWith('audio_cough_'))) {
    it(`fixture ${f}`, () => check(JSON.parse(readFileSync(join(dir, f), 'utf8')).input.frames, 1));
  }
  it('a long random sequence (bursts, dips, borderline values)', () => {
    let s = 12345;
    const r = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
    const frames: [number, number][] = [];
    let burst = 0;
    for (let i = 0; i < 6000; i++) {
      if (burst === 0 && r() < 0.04) burst = 1 + Math.floor(r() * 4);
      const v = burst > 0 ? (burst--, 0.25 + 0.7 * r()) : r() < 0.1 ? 0.3 + 0.2 * r() : 0.05 * r();
      frames.push([0.975 + i * 0.48, v]);
    }
    check(frames, 37);
  });
  it('keeps bounded memory (ticks capped) while the count keeps growing', () => {
    const tr = new LabelEventTracker(P.cough, 50);
    for (let i = 0; i < 20000; i++) tr.push(i * 0.48, i % 10 === 0 ? 0.9 : 0);
    const v = tr.view();
    expect(v.count).toBeGreaterThan(1900);
    expect(v.ticks.length).toBeLessThanOrEqual(51);
  });
});
