// Contact PPG TS port: unit tests (FFT vs naive DFT, parabolic interpolation, taper, streaming gates).
// Fixture parity lives in test/parity/params.test.ts via src/core/parity.ts. Not a medical device.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  ContactPPG, assertContactPpgParams, contactPpgErrors, bandBins, bandpass, detrendRelative, fft, fftSize, findBeats, ifft, interp, parabolicLin,
  parabolicLog, percentile, qualityScore, taper, type ContactPpgParams,
} from '../src/core/ppg';

const SHARED = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'shared');
const P: ContactPpgParams = JSON.parse(readFileSync(join(SHARED, 'params.json'), 'utf8')).contact_ppg;

function naiveDft(x: number[]): { re: number[]; im: number[] } {
  const n = x.length, re = new Array<number>(n).fill(0), im = new Array<number>(n).fill(0);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) {
    const a = (-2 * Math.PI * k * j) / n;
    re[k]! += x[j]! * Math.cos(a); im[k]! += x[j]! * Math.sin(a);
  }
  return { re, im };
}

describe('fft', () => {
  it('matches a naive DFT for several sizes and round-trips through ifft', () => {
    for (const n of [2, 8, 64, 256]) {
      const x = Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) + 0.3 * Math.cos(i * 1.9) + ((i * 7919) % 13) / 13);
      const ref = naiveDft(x);
      const re = Float64Array.from(x), im = new Float64Array(n);
      fft(re, im);
      for (let k = 0; k < n; k++) {
        expect(re[k]).toBeCloseTo(ref.re[k]!, 8);
        expect(im[k]).toBeCloseTo(ref.im[k]!, 8);
      }
      ifft(re, im);
      for (let i = 0; i < n; i++) { expect(re[i]).toBeCloseTo(x[i]!, 10); expect(im[i]).toBeCloseTo(0, 10); }
    }
  });
  it('rejects non power-of-two lengths', () => {
    expect(() => fft(new Float64Array(12), new Float64Array(12))).toThrow();
  });
  it('fftSize is the next power of two >= factor * n', () => {
    expect(fftSize(300, 4)).toBe(2048);
    expect(fftSize(256, 4)).toBe(1024);
  });
});

describe('parabolic interpolation', () => {
  it('recovers the vertex of an exact parabola', () => {
    const f = (x: number) => 5 - (x - 0.3) ** 2;
    expect(parabolicLin(f(-1), f(0), f(1))).toBeCloseTo(0.3, 12);
    expect(parabolicLin(1, 1, 1)).toBe(0);
    expect(parabolicLin(0, 1, 100)).toBe(0.5 * (0 - 100) / (0 - 2 + 100) < -0.5 ? -0.5 : 0.5 * (0 - 100) / (0 - 2 + 100));
  });
  it('log version finds a Gaussian peak between bins', () => {
    const pw = Array.from({ length: 9 }, (_, k) => Math.exp(-((k - 4.2) ** 2) / 2));
    expect(parabolicLog(pw, 4)).toBeCloseTo(0.2, 6);
  });
});

describe('band-pass taper', () => {
  it('has the documented shape', () => {
    for (const f of [0, 0.3, 0.5, 3.5, 4, 10]) expect(taper(f, P)).toBe(0);
    for (const f of [0.7, 1, 2.2, 3.0]) expect(taper(f, P)).toBe(1);
    expect(taper(0.6, P)).toBeCloseTo(0.5, 12);
    expect(taper(3.25, P)).toBeCloseTo(0.5, 12);
    let prev = 0;
    for (let f = 0.5; f <= 0.7; f += 0.01) { const g = taper(f, P); expect(g).toBeGreaterThanOrEqual(prev); prev = g; }
  });
  it('band-pass keeps an in-band tone and removes a slow drift', () => {
    const n = 300, fs = 30, nFft = fftSize(n, P.zero_pad_factor);
    const x = Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * 1.5 * i) / fs) + 3 * Math.sin((2 * Math.PI * 0.1 * i) / fs));
    const y = bandpass(x, nFft, fs, P);
    let e = 0;
    for (let i = 60; i < n - 60; i++) e = Math.max(e, Math.abs(y[i]! - Math.sin((2 * Math.PI * 1.5 * i) / fs)));
    expect(e).toBeLessThan(0.2);
  });
  it('bandBins covers 0.7..3.0 Hz', () => {
    expect(bandBins(2048, 30, 0.7, 3.0)).toEqual([48, 204]);
  });
});

describe('helpers', () => {
  it('interp holds the ends and is linear inside', () => {
    expect(interp([-1, 0.5, 1.5, 9], [0, 1, 2], [0, 10, 30])).toEqual([0, 5, 20, 30]);
  });
  it('percentile matches numpy linear', () => {
    expect(percentile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95)).toBeCloseTo(9.5, 12);
    expect(percentile([3, 1, 2], 50)).toBe(2);
  });
  it('detrendRelative removes a line and divides by the mean', () => {
    const y = Array.from({ length: 50 }, (_, i) => 100 + 0.5 * i);
    const { rel, mean } = detrendRelative(y);
    expect(mean).toBeCloseTo(112.25, 9);
    for (const v of rel) expect(Math.abs(v)).toBeLessThan(1e-12);
  });
  it('findBeats finds sinusoid maxima with sub-sample timing and enforces spacing', () => {
    const fs = 30, x = Array.from({ length: 300 }, (_, i) => Math.sin((2 * Math.PI * 1.2 * i) / fs));
    const b = findBeats(x, fs, 0.4, 0.5);
    expect(b.length).toBeGreaterThanOrEqual(11);
    for (let i = 1; i < b.length; i++) expect(((b[i]! - b[i - 1]!) / fs) * 1000).toBeCloseTo(1000 / 1.2, -1);
    expect(findBeats(x, fs, 2.0, 0.5).length).toBeLessThan(b.length);
  });
  it('quality is monotone and bounded', () => {
    expect(qualityScore(10, 1, 1, P)).toBeLessThan(qualityScore(100, 1, 1, P));
    expect(qualityScore(30, 4, 1, P)).toBeLessThan(qualityScore(30, 0, 1, P));
    expect(qualityScore(1e9, 0, 1, P)).toBeLessThanOrEqual(1);
    expect(qualityScore(0, null, 1e6, P)).toBeGreaterThanOrEqual(0);
  });
});

describe('ContactPPG streaming', () => {
  const pulse = (t: number, bpm: number) => {
    const ph = ((t * bpm) / 60) % 1;
    return Math.sin(2 * Math.PI * ph) + 0.3 * Math.sin(4 * Math.PI * ph + 1);
  };
  const feed = (cp: ContactPPG, t0: number, t1: number, f: (t: number) => [number, number, number]) => {
    const out = [];
    for (let t = t0; t < t1; t += 1 / 30) { const [r, g, c] = f(t); const u = cp.update(t, r, g, 0, c); if (u) out.push(u); }
    return out;
  };
  it('settles, reports the right rate, then flags a lifted finger and saturation', () => {
    const cp = new ContactPPG(P);
    const good = (t: number): [number, number, number] => [200 + 1.5 * pulse(t, 78), 60 + 0.2 * pulse(t, 78), 0];
    const a = feed(cp, 0, 25, good);
    expect(a[0]!.reason).toBe('settling');
    const ok = a.filter((u) => u.reported);
    expect(ok.length).toBeGreaterThan(10);
    for (const u of ok) expect(Math.abs(u.bpm! - 78)).toBeLessThan(1);
    const lifted = feed(cp, 25, 30, () => [40, 38, 0]);
    expect(lifted.every((u) => u.reason === 'no_finger' && !u.reported && u.bpm === null)).toBe(true);
    const sat = feed(cp, 30, 34, () => [255, 60, 0.9]);
    expect(sat.slice(1).every((u) => u.reason === 'saturated' && u.bpm === null)).toBe(true); // first frame is a |dR| jolt
    expect(cp.contactSeconds(34)).toBe(0);
  });
  it('ignores non-monotonic timestamps and a >0.5 s gap is "gaps"', () => {
    const cp = new ContactPPG(P);
    expect(cp.update(1, 200, 60)).toBeNull();
    expect(cp.update(0.5, 200, 60)).toBeNull();
    const cp2 = new ContactPPG(P);
    const good = (t: number): [number, number, number] => [200 + 1.5 * pulse(t, 78), 60, 0];
    feed(cp2, 0, 20, good);
    const after = feed(cp2, 20.9, 24, good);
    expect(after.some((u) => u.reason === 'gaps')).toBe(true);
  });
});

describe('contact_ppg params validation', () => {
  it('accepts the shipped params and rejects inconsistent ones', () => {
    expect(contactPpgErrors(P)).toEqual([]);
    expect(() => assertContactPpgParams(P)).not.toThrow();
    for (const bad of [{ band_lo_hz: 0.4 }, { taper_stop_hi_hz: 20 }, { q_w_pi: 0.5 }, { min_snr: 0 }, { pi_min: 5 }, { q_snr_full_db: 1 }]) {
      expect(() => new ContactPPG({ ...P, ...bad }), JSON.stringify(bad)).toThrow(/invalid contact_ppg/);
    }
  });
});

// Small halving/doubling regression sweep (the Python one is larger). Own PRNG: only the rule "never report
// a value more than 10% off" is checked, not parity.
function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function trace(bpm: number, seed: number, noise: number, wander: number, rrJitter: number) {
  const rnd = mulberry(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const beats: number[] = [];
  for (let b = -1 + rnd(); b < 45; b += (60 / bpm) * (1 + rrJitter * gauss())) beats.push(b);
  const pulse = (tau: number) => Math.exp(-0.5 * ((tau - 0.18) / 0.09) ** 2) + 0.4 * Math.exp(-0.5 * ((tau - 0.45) / 0.12) ** 2);
  const out: { t: number; r: number; g: number }[] = [];
  for (let i = 0; i < 40 * 30; i++) {
    const t = i / 30;
    let w = 0;
    for (const b0 of beats) if (t >= b0 && t < b0 + 1.2) w += pulse(t - b0);
    out.push({ t, r: 200 + (2 * w) / 1.4 + wander * Math.sin(2 * Math.PI * 0.1 * t + 0.7) + noise * gauss(), g: 60 + 0.1 * w + 0.5 * gauss() });
  }
  return out;
}

describe('half/double-rate regression sweep', () => {
  it('never reports a value >10% off; clean traces mostly report', () => {
    let windows = 0, cleanWin = 0, cleanRep = 0;
    for (const bpm of [45, 72, 110, 150, 170, 180]) for (const seed of [1, 2]) for (const noise of [0, 0.5]) for (const wander of [0, 4]) for (const rrj of [0, 0.05]) {
      const cp = new ContactPPG(P);
      for (const f of trace(bpm, seed, noise, wander, rrj)) {
        const u = cp.update(f.t, f.r, f.g, 0, 0);
        if (!u || u.t <= 12) continue;
        windows++;
        if (u.reported) expect(Math.abs(u.bpm! - bpm), `${bpm} ${seed} ${noise} ${wander} ${rrj}`).toBeLessThanOrEqual(0.1 * bpm);
        if (noise === 0) { cleanWin++; if (u.reported) cleanRep++; }
      }
    }
    expect(windows).toBeGreaterThanOrEqual(300);
    expect(cleanRep / cleanWin).toBeGreaterThanOrEqual(0.8);
  });
});
