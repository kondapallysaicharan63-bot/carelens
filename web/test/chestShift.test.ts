import { describe, expect, it } from 'vitest';
import { BreathingChannel, chestRoi } from '../src/core/breathing';
import { GrayShiftTracker, estimateShift, rowProfile } from '../src/core/chestShift';
import params from '../../shared/params.json';

// deterministic noise
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 2 ** 32; };
}
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

/** smooth "shirt" texture sampled at y - shift (analytic, so sub-pixel shifts are exact) */
const texture = (y: number): number =>
  120 + 30 * Math.sin(y * 0.55) + 22 * Math.sin(y * 0.23 + 1) + 15 * Math.sin(y * 1.1 + 2) + 18 * Math.exp(-((y - 40) ** 2) / 30);
const profile = (n: number, shift: number, noise = 0, r = rng(1)): Float64Array =>
  Float64Array.from({ length: n }, (_, i) => texture(i - shift) + (noise ? noise * gauss(r) : 0));

describe('estimateShift (1-D cross-correlation + parabolic peak)', () => {
  const cs = params.chest_shift;
  it('recovers known sub-pixel shifts within 0.1 px (clean)', () => {
    const base = profile(80, 0);
    for (const d of [-3, -2.4, -1.5, -0.75, -0.3, -0.1, 0, 0.1, 0.25, 0.5, 0.8, 1.3, 2.7, 3.5]) {
      const r = estimateShift(base, profile(80, d), cs.max_shift_px, cs.min_corr);
      expect(r, `shift ${d}`).not.toBeNull();
      expect(Math.abs(r!.dy - d), `shift ${d}`).toBeLessThan(0.1);
    }
  });
  it('has the right sign: content moving down is positive', () => {
    expect(estimateShift(profile(80, 0), profile(80, 1.2), 4, 0.2)!.dy).toBeGreaterThan(1);
    expect(estimateShift(profile(80, 0), profile(80, -1.2), 4, 0.2)!.dy).toBeLessThan(-1);
  });
  it('stays accurate with noise (sd 3 levels): mean error < 0.1 px over 200 trials', () => {
    const r = rng(7);
    let sum = 0, worst = 0;
    for (let k = 0; k < 200; k++) {
      const d = (r() - 0.5) * 4;
      const e = estimateShift(profile(80, 0, 3, r), profile(80, d, 3, r), 4, 0.2)!.dy - d;
      sum += Math.abs(e); worst = Math.max(worst, Math.abs(e));
    }
    expect(sum / 200).toBeLessThan(0.1);
    expect(worst).toBeLessThan(0.4);
  });
  it('returns null for flat or uncorrelated profiles and too-short patches', () => {
    expect(estimateShift(new Float64Array(80).fill(100), new Float64Array(80).fill(100), 4, 0.2)).toBeNull();
    const r = rng(3);
    const noise = (): Float64Array => Float64Array.from({ length: 80 }, () => gauss(r));
    expect(estimateShift(noise(), noise(), 4, 0.6)).toBeNull();
    expect(estimateShift(new Float64Array(8), new Float64Array(8), 4, 0.2)).toBeNull();
  });
  it('rowProfile averages columns and caps the patch width', () => {
    const w = 200, h = 10;
    const img = new Uint8ClampedArray(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[y * w + x] = y * 10;
    const p = rowProfile(img, w, { x0: 0, y0: 2, x1: 200, y1: 8 }, 96);
    expect(Array.from(p)).toEqual([20, 30, 40, 50, 60, 70]);
  });
});

describe('GrayShiftTracker on synthetic translated frames', () => {
  it('tracks a vertical translation of a textured frame incl. sub-pixel steps', () => {
    const w = 160, h = 120;
    const frame = (shift: number) => {
      const g = new Uint8ClampedArray(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) g[y * w + x] = texture(y - shift) + 6 * Math.sin(x * 0.3);
      return g;
    };
    const t = new GrayShiftTracker(params.chest_shift);
    const box = { x0: 30, y0: 20, x1: 130, y1: 100 };
    expect(t.shift(frame(0), w, box)).toBe('first'); // no previous frame
    let pos = 0;
    for (const s of [0.4, 0.9, 1.2, 0.7]) {
      const dy = t.shift(frame(s), w, box) as number;
      pos += dy;
      expect(Math.abs(pos - s)).toBeLessThan(0.15);
    }
    t.reset();
    expect(t.shift(frame(0), w, box)).toBe('first');
  });
  it('an unusable shift is reported, not returned as 0: small box -> too_small, uncorrelated frames -> weak', () => {
    const w = 160, h = 120, t = new GrayShiftTracker({ ...params.chest_shift, min_corr: 0.6 });
    const r = rng(5);
    const noisy = () => Uint8ClampedArray.from({ length: w * h }, () => 128 + 60 * gauss(r));
    t.shift(noisy(), w, { x0: 30, y0: 20, x1: 130, y1: 100 });
    expect(t.shift(noisy(), w, { x0: 30, y0: 20, x1: 130, y1: 100 })).toBe('weak');
    expect(t.shift(noisy(), w, { x0: 30, y0: 20, x1: 130, y1: 26 })).toBe('too_small');
  });
});

describe('BreathingChannel (pose gating + displacement trace)', () => {
  const vit = params.vitals as any;
  const common = params.common;
  // landmarks in frame-height units, upright torso in a 320x240 working frame
  const lm = (() => {
    const a = Array.from({ length: 33 }, () => [0, 0, 0.2]);
    a[11] = [0.55, 0.35, 0.95]; a[12] = [0.4, 0.35, 0.95]; a[23] = [0.53, 0.6, 0.95]; a[24] = [0.42, 0.6, 0.95];
    return a;
  })();
  const fake = (dyOf: (i: number) => number | null | 'weak' | 'too_small') => {
    let i = 0, resets = 0, calls = 0;
    return {
      size: () => ({ w: 320, h: 240 }),
      shiftAndAdvance: () => { calls++; const v = dyOf(i++); return v === null ? ('first' as const) : v; },
      reset: () => { resets++; },
      get resets() { return resets; }, get calls() { return calls; },
    };
  };

  it('chest ROI sits between shoulders and mid-torso, inside the frame', () => {
    const b = chestRoi(lm, 240, 320, vit.roi)!;
    expect(b.y0).toBeGreaterThan(0.35 * 240 - 1);
    expect(b.y1).toBeLessThan(0.6 * 240);
    expect(b.x0).toBeGreaterThan(0.38 * 240);
    expect(b.x1).toBeLessThan(0.57 * 240);
    const hidden = lm.map((r) => [r[0]!, r[1]!, 0.1]);
    expect(chestRoi(hidden, 240, 320, vit.roi)).toBeNull();
  });

  it('recovers a 14 breaths/min sinusoid delivered as per-frame shifts at 8 Hz', () => {
    const ch = new BreathingChannel(vit, common);
    const fps = 8, hz = 14 / 60, amp = 0.8;
    let prev = 0;
    const src = fake((i) => { const y = amp * Math.sin(2 * Math.PI * hz * (i / fps)); const d = y - prev; prev = y; return i === 0 ? null : d; });
    const st = { present: true, motion: 0.01, landmarks: lm };
    let last = null as null | { value: number; reported: boolean };
    for (let i = 0; i < 8 * 60; i++) { const e = ch.update(i / fps, st, false, src).estimate; if (e) last = e; }
    expect(ch.status).toBe('ok');
    expect(last!.reported).toBe(true);
    expect(Math.abs(last!.value - 14)).toBeLessThan(1);
  });

  it('resets on absent, moving and held frames and reports why', () => {
    const ch = new BreathingChannel(vit, common);
    const src = fake(() => 0.1);
    ch.update(0, { present: true, motion: 0.01, landmarks: lm }, false, src);
    expect(ch.status).toBe('ok');
    ch.update(0.1, { present: true, motion: 0.5, landmarks: lm }, false, src);
    expect(ch.status).toBe('moving');
    ch.update(0.2, { present: false, motion: 0, landmarks: null }, false, src);
    expect(ch.status).toBe('absent');
    ch.update(0.3, { present: true, motion: 0.01, landmarks: lm }, true, src);
    expect(ch.status).toBe('held');
    expect(src.resets).toBe(3);
  });

  it('never injects zeros for unusable shifts: weak frames are skipped, too_small resets with guidance status', () => {
    const ch = new BreathingChannel(vit, common);
    const st = { present: true, motion: 0.01, landmarks: lm };
    const weak = fake(() => 'weak');
    for (let i = 0; i < 10; i++) ch.update(i / 8, st, false, weak);
    expect(ch.estimator.updateDisplacement(10, 0).due).toBe(false);
    // nothing was pushed by the weak frames: the buffer only has the probe sample
    const small = fake(() => 'too_small');
    expect(ch.update(0, st, false, small).estimate).toBeNull();
    expect(ch.status).toBe('too_small');
    expect(small.resets).toBe(1);
    // persistently weak -> the window starts over
    const ch2 = new BreathingChannel(vit, common);
    const w2 = fake(() => 'weak');
    for (let i = 0; i < 20; i++) ch2.update(i / 8, st, false, w2);
    expect(w2.resets).toBeGreaterThan(0);
  });
});
