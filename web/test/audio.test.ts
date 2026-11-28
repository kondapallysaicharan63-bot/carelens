import { describe, expect, it } from 'vitest';
import params from '../../shared/params.json';
import {
  BreathSoundRate, EnvelopeStream, Resampler, assertAudioParams, detectLabelEvents, envelope, type AudioParams,
} from '../src/core/audio';

const P = assertAudioParams(params.audio as AudioParams);

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

/** breath envelope: one bump per breath (or two, exhale scaled), 50 Hz */
function breathEnv(dur: number, bpm: number, opt: { amp?: number; jitter?: number; double?: boolean; exhale?: number; seed?: number } = {}): number[] {
  const { amp = 0.6, jitter = 0.05, double = false, exhale = 0.5, seed = 7 } = opt;
  const width = double ? 0.6 : 0.9;
  const r = lcg(seed);
  const half = double ? 30 / bpm : 60 / bpm;
  const out: number[] = [];
  for (let i = 0; i < dur * 50; i++) {
    const t = i / 50;
    const k = Math.floor(t / half);
    const x = (t - k * half) / (width * half);
    const a = !double || k % 2 === 0 ? 1 : exhale;
    out.push(1 + (x < 1 ? amp * a * 0.5 * (1 - Math.cos(2 * Math.PI * x)) : 0) + jitter * gauss(r));
  }
  return out;
}
function run(env: Array<number | null>, speech: [number, number][] = []) {
  const e = new BreathSoundRate(P);
  speech.forEach(([t, s]) => e.pushSpeech(t, s));
  return env.map((v, i) => e.push(i / 50, v)).filter((u) => u !== null);
}

describe('audio params', () => {
  it('validate and reject broken values', () => {
    expect(() => assertAudioParams({ ...P, envelope: { ...P.envelope, frame_len: 500 } })).toThrow();
    expect(() => assertAudioParams({ ...P, cough: { ...P.cough, off_thr: 0.9 } })).toThrow();
  });
});

describe('envelope', () => {
  it('stream equals batch, frame count and timing', () => {
    const r = lcg(3);
    const x = Float64Array.from({ length: 20000 }, () => 0.1 * gauss(r));
    const b = envelope(x, P);
    expect(b.envelope.length).toBe(1 + Math.floor((20000 - 512) / 320));
    const st = new EnvelopeStream(P);
    const out = [];
    for (let i = 0; i < x.length; i += 777) out.push(...st.push(x.subarray(i, i + 777)));
    expect(out.length).toBe(b.envelope.length);
    out.forEach((f, i) => expect(f.env).toBeCloseTo(b.envelope[i]!, 9));
    expect(out[0]!.t).toBeCloseTo(512 / 16000, 12);
    expect(out[1]!.t - out[0]!.t).toBeCloseTo(0.02, 12);
  });
  it('silence is flat and clipping is counted', () => {
    const e = envelope(new Float64Array(16000), P).envelope;
    expect(Math.max(...e) - Math.min(...e)).toBe(0);
    expect(envelope(new Float64Array(2000).fill(1), P).clip_frac[0]).toBe(1);
  });
});

describe('Resampler', () => {
  it('48 kHz -> 16 kHz keeps a 440 Hz tone and removes a 12 kHz one', () => {
    const n = 48000;
    const rs = new Resampler(48000, 16000);
    const tone = (f: number) => {
      const y = rs.push(Float64Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * f * i) / 48000)));
      let s = 0;
      for (let i = 2000; i < y.length - 100; i++) s += y[i]! * y[i]!;
      return { len: y.length, rms: Math.sqrt(s / (y.length - 2100)) };
    };
    const a = tone(440);
    expect(Math.abs(a.len - 16000)).toBeLessThan(40);
    expect(a.rms).toBeGreaterThan(0.65);
    expect(tone(12000).rms).toBeLessThan(0.1);
  });
});

describe('breathing from sound', () => {
  it('reports the rate; early windows are settling', () => {
    const ups = run(breathEnv(50, 15));
    expect(ups[0]!.reason).toBe('settling');
    const rep = ups.filter((u) => u!.reported);
    expect(rep.length).toBeGreaterThan(8);
    rep.forEach((u) => expect(Math.abs(u!.bpm! - 15)).toBeLessThan(0.6));
  });
  it('inhale + exhale audible: true rate (alternating bumps) or withheld, never 2x', () => {
    for (const exhale of [0.5, 2]) {
      const rep = run(breathEnv(60, 15, { double: true, exhale })).filter((u) => u!.reported);
      expect(rep.length).toBeGreaterThan(5);
      rep.forEach((u) => expect(Math.abs(u!.bpm! - 15)).toBeLessThan(0.6));
    }
    const eq = run(breathEnv(50, 15, { double: true, exhale: 1 })).filter((u) => u!.t >= 30);
    eq.forEach((u) => { expect(u!.reported).toBe(false); expect(u!.bpm).toBeNull(); expect(u!.reason).toBe('ambiguous'); });
  });
  it('regression sweep 6-40/min: no report off by more than 10 % (no 2x, no 1/2x)', () => {
    const conds = [{}, { double: true, exhale: 1 }, { double: true, exhale: 0.5 }, { double: true, exhale: 2 }];
    let windows = 0, wrong = 0;
    for (const c of conds) for (const bpm of [6, 8, 12, 15, 20, 28, 40]) for (const seed of [1, 2]) for (const jitter of [0.03, 0.1]) {
      for (const u of run(breathEnv(50, bpm, { ...c, seed: seed * 13 + bpm, jitter })).filter((x) => x!.t >= 22)) {
        windows++;
        if (u!.reported && Math.abs(u!.bpm! - bpm) > 0.1 * bpm) wrong++;
      }
    }
    expect(windows).toBeGreaterThan(300);
    expect(wrong).toBe(0);
  });
  it('gates: quiet, talking, no_mic', () => {
    expect(new Set(run(breathEnv(40, 15, { amp: 0.03, jitter: 0.03 })).filter((u) => u!.t > 20).map((u) => u!.reason))).toEqual(new Set(['too_quiet']));
    const sp: [number, number][] = Array.from({ length: 80 }, (_, i) => [1 + i * 0.48, 0.9]);
    expect(new Set(run(breathEnv(40, 15), sp).filter((u) => u!.t > 20).map((u) => u!.reason))).toEqual(new Set(['talking']));
    const nm = run(Array(401).fill(null));
    expect(nm.map((u) => u!.reason)).toEqual(['no_mic', 'no_mic', 'no_mic', 'no_mic']);
  });
});

describe('label events', () => {
  const ev = (s: number[], over: Partial<AudioParams['cough']> = {}) => detectLabelEvents(s.map((v, i) => [i * 0.48, v] as const), { ...P.cough, ...over });
  it('hysteresis, merge, refractory, min_frames', () => {
    expect(ev([0, 0.6, 0.4, 0.35, 0.2, 0]).events).toEqual([{ t_start: 0.48, t_end: 1.44, peak_score: 0.6 }]);
    expect(ev([0, 0.45, 0.499, 0.45, 0]).count).toBe(0);
    expect(ev([0, 0.6, 0.3, 0.299, 0.6]).count).toBe(1);
    expect(ev([0, 0.6, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0.6]).count).toBe(3);
    expect(ev([0, 0.6, 0, 0, 0, 0.6, 0]).count).toBe(1);
    expect(ev([0, 0.6, 0, 0, 0, 0.6, 0], { refractory_s: 0 }).count).toBe(2);
    expect(ev([0, 0.6, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.6, 0], { min_frames: 2 }).count).toBe(1);
  });
});
