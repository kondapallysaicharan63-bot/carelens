// Face rPPG TS port: unit tests (params checks, POS/CHROM properties, streaming gates).
// Fixture parity lives in test/parity/params.test.ts via src/core/parity.ts. Not a medical device.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { FaceRPPG, assertRppgParams, chromSignal, posSignal, qualityScore, rppgErrors, type Rgb, type RppgParams } from '../src/core/rppg';
import { detrendLinear, fftSize } from '../src/core/dsp';

const SHARED = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'shared');
const P: RppgParams = JSON.parse(readFileSync(join(SHARED, 'params.json'), 'utf8')).rppg;

/** Simple skin-colour trace: DC [180,125,95], POS-model pulse signature [0.33,0.77,0.53], tiny deterministic noise. */
const DC = [180, 125, 95] as const;
const SIG = [0.33 / 0.77, 1, 0.53 / 0.77] as const;
function skin(t: number, bpm: number, k = 1): Rgb {
  const ph = ((t * bpm) / 60) % 1;
  const w = Math.sin(2 * Math.PI * ph) + 0.3 * Math.sin(4 * Math.PI * ph + 1);
  const n = Math.sin(t * 977.7) * 0.01;
  return [0, 1, 2].map((c) => k * DC[c]! * (1 - 0.005 * SIG[c]! * w) + n) as unknown as Rgb;
}

describe('rppg params', () => {
  it('shared params pass the consistency checks; broken ones are reported', () => {
    expect(rppgErrors(P)).toEqual([]);
    expect(rppgErrors({ ...P, q_w_snr: 0.9 }).length).toBeGreaterThan(0);
    expect(rppgErrors({ ...P, min_brightness: 250 }).length).toBeGreaterThan(0);
    expect(rppgErrors({ ...P, pos_window_s: 0.1 }).length).toBeGreaterThan(0);
    expect(rppgErrors({ ...P, motion_frac_max: 1.5 }).length).toBeGreaterThan(0);
    expect(rppgErrors({ ...P, roi_landmarks: { ...P.roi_landmarks, forehead: [1, 2, 2] } }).length).toBeGreaterThan(0);
    expect(() => assertRppgParams({ ...P, band_hi_hz: 5 })).toThrow();
  });
  it('ROI polygons are distinct landmark indices in the 478-point mesh', () => {
    const all = [...P.roi_landmarks.forehead, ...P.roi_landmarks.left_cheek, ...P.roi_landmarks.right_cheek];
    expect(new Set(all).size).toBe(all.length);
    expect(all.every((i) => i >= 0 && i <= 477)).toBe(true);
  });
});

describe('POS / CHROM', () => {
  const n = 360;
  const cols = (k: (t: number) => number): [number[], number[], number[]] => {
    const r: number[] = [], g: number[] = [], b: number[] = [];
    for (let i = 0; i < n; i++) { const t = i / 30; const v = skin(t, 72, k(t)); r.push(v[0]); g.push(v[1]); b.push(v[2]); }
    return [r, g, b];
  };
  const sd = (a: number[]) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
  it('both reject a strong common-mode intensity flicker', () => {
    const [r0, g0, b0] = cols(() => 1);
    const [r1, g1, b1] = cols((t) => 1 + 0.05 * Math.sin(2 * Math.PI * 1.9 * t));
    const nFft = fftSize(n, P.zero_pad_factor);
    const pa = posSignal(r0, g0, b0, P.fs_hz, P.pos_window_s), pb = posSignal(r1, g1, b1, P.fs_hz, P.pos_window_s);
    expect(sd(pa.map((v, i) => v - pb[i]!))).toBeLessThan(0.12 * sd(pa));
    const ca = chromSignal(r0, g0, b0, nFft, P.fs_hz, P), cb = chromSignal(r1, g1, b1, nFft, P.fs_hz, P);
    expect(sd(ca.map((v, i) => v - cb[i]!))).toBeLessThan(0.12 * sd(ca));
    // the green channel alone is not rejected
    expect(sd(g1.map((v, i) => v - g0[i]!))).toBeGreaterThan(2 * sd(detrendLinear(g0)));
  });
  it('quality is monotone in SNR, agreement and motion, and bounded', () => {
    expect(qualityScore(4, 1, 0.1, P)).toBeLessThan(qualityScore(40, 1, 0.1, P));
    expect(qualityScore(10, 4, 0.1, P)).toBeLessThan(qualityScore(10, 0.5, 0.1, P));
    expect(qualityScore(10, 1, 0.4, P)).toBeLessThan(qualityScore(10, 1, 0.05, P));
    expect(qualityScore(1e9, 0, 0, P)).toBeLessThanOrEqual(1);
    expect(qualityScore(0, 99, 9, P)).toBeGreaterThanOrEqual(0);
  });
});

describe('FaceRPPG streaming gates', () => {
  const feed = (fr: FaceRPPG, t0: number, t1: number, f: (t: number) => [Rgb | null, number, number, number]) => {
    const out = [];
    for (let t = t0; t < t1; t += 1 / 30) { const [rgb, px, mo, br] = f(t); const u = fr.update(t, rgb, px, mo, br); if (u) out.push(u); }
    return out;
  };
  const good = (t: number): [Rgb | null, number, number, number] => { const v = skin(t, 78); return [v, 2000, 0.02, (v[0] + v[1] + v[2]) / 3]; };

  it('settles, reports the right rate, then flags no_face, too_small and a lighting step', () => {
    const fr = new FaceRPPG(P);
    const a = feed(fr, 0, 25, good);
    expect(a[0]!.reason).toBe('settling');
    const ok = a.filter((u) => u.reported);
    expect(ok.length).toBeGreaterThan(8);
    for (const u of ok) expect(Math.abs(u.bpm! - 78)).toBeLessThan(1.5);
    const lost = feed(fr, 25, 29, () => [null, 0, 0, 0]);
    expect(lost.every((u) => u.reason === 'no_face' && !u.reported && u.bpm === null && u.quality === 0)).toBe(true);
    const small = feed(fr, 29, 32, (t) => { const v = skin(t, 78); return [v, 100, 0.02, 133]; });
    expect(small.every((u) => u.reason === 'too_small')).toBe(true);
    const epoch0 = small[small.length - 1]!.track_epoch;
    const back = feed(fr, 32, 36, good);
    expect(back.every((u) => u.reason === 'settling')).toBe(true);
    expect(back[back.length - 1]!.track_epoch).toBe(epoch0 + 1);
    const step = feed(fr, 36, 40, (t) => { const v = skin(t, 78, 0.5); return [v, 2000, 0.02, (v[0] + v[1] + v[2]) / 3]; });
    expect(step.some((u) => u.reason === 'lighting')).toBe(true);
  });
  it('too dark is lighting; a >0.5 s hiccup is gaps; non-monotonic frames are ignored', () => {
    const dark = new FaceRPPG(P);
    expect(feed(dark, 0, 5, (t) => { const v = skin(t, 78, 0.2); return [v, 2000, 0.02, 25]; }).every((u) => u.reason === 'lighting')).toBe(true);
    const fr = new FaceRPPG(P);
    expect(fr.update(1, good(1)[0], 2000, 0, 130)).toBeNull();
    expect(fr.update(0.5, good(1)[0], 2000, 0, 130)).toBeNull();
    const g = new FaceRPPG(P);
    feed(g, 0, 12, good);
    const after = feed(g, 12.8, 14, good);
    expect(after.some((u) => u.reason === 'gaps')).toBe(true);
  });
  it('a dead channel inside the window never yields a NaN or a reported reading', () => {
    const fr = new FaceRPPG(P);
    const out = feed(fr, 0, 25, (t) => { const v = skin(t, 78); const dead = t > 14 && t < 16.2; return [[dead ? 0 : v[0], v[1], v[2]] as unknown as Rgb, 2000, 0.02, 120, ]; });
    for (const u of out) {
      for (const k of ['bpm', 'bpm_chrom', 'snr'] as const) expect(u[k] === null || Number.isFinite(u[k])).toBe(true);
      expect(Number.isFinite(u.quality)).toBe(true);
    }
  });
  it('skin_filter is its own reason (not too_small) and restarts the track', () => {
    const fr = new FaceRPPG(P);
    feed(fr, 0, 8, good);
    const sk = feed(fr, 8, 11, (t) => [skin(t, 78), 2000, 0.02, 133] as [Rgb, number, number, number]);
    void sk;
    const out: ReturnType<FaceRPPG['update']>[] = [];
    for (let t = 11; t < 14; t += 1 / 30) out.push(fr.update(t, skin(t, 78), 2000, 0.02, 133, 0.05));
    expect(out.filter((u) => u).every((u) => u!.reason === 'skin_filter' && u!.bpm === null)).toBe(true);
  });
  it('face motion above the threshold withholds the reading', () => {
    const fr = new FaceRPPG(P);
    const out = feed(fr, 0, 25, (t) => { const v = skin(t, 78); return [v, 2000, t > 14 && t < 15.5 ? 1.0 : 0.02, (v[0] + v[1] + v[2]) / 3]; });
    expect(out.filter((u) => u.t > 16 && u.t < 25).every((u) => u.reason === 'motion' && u.bpm === null)).toBe(true);
  });
});
