import { describe, expect, it } from 'vitest';
import { FeatureExtractor, type BodyState, type Landmarks } from '../src/core/features';
import { PresenceTracker } from '../src/core/tracking';
import { FallDetector, InactivityDetector } from '../src/core/events';
import { PosePipeline } from '../src/core/pipeline';
import { BreathingRateEstimator, estimateWindow } from '../src/core/breathing';
import { butterBandpassSos, sosfiltfilt, welch } from '../src/core/welch';
import params from '../../shared/params.json';

const ASPECT = 4 / 3;

/** standing / sitting / lying skeletons in frame-height units, then x / aspect as MediaPipe returns them */
function skeleton(kind: 'stand' | 'sit' | 'lie', x = 0.5, vis = 0.95): Landmarks {
  const lm: Landmarks = Array.from({ length: 33 }, () => [0, 0, 0.2]);
  const set = (k: number, px: number, py: number) => { lm[k] = [px / ASPECT, py, vis]; };
  if (kind === 'stand') {
    set(0, x, 0.25); set(2, x + 0.02, 0.235); set(5, x - 0.02, 0.235);
    set(11, x + 0.07, 0.33); set(12, x - 0.07, 0.33); set(23, x + 0.05, 0.55); set(24, x - 0.05, 0.55);
    set(25, x + 0.05, 0.72); set(26, x - 0.05, 0.72); set(27, x + 0.05, 0.9); set(28, x - 0.05, 0.9);
  } else if (kind === 'sit') {
    set(0, x, 0.32); set(2, x + 0.02, 0.305); set(5, x - 0.02, 0.305);
    set(11, x + 0.07, 0.4); set(12, x - 0.07, 0.4); set(23, x + 0.05, 0.62); set(24, x - 0.05, 0.62);
    set(25, x + 0.07, 0.64); set(26, x - 0.07, 0.64); set(27, x + 0.07, 0.86); set(28, x - 0.07, 0.86);
  } else {
    const y = 0.84;
    set(0, x + 0.3, y - 0.02); set(11, x + 0.22, y - 0.03); set(12, x + 0.22, y + 0.03);
    set(23, x, y - 0.03); set(24, x, y + 0.03); set(25, x - 0.17, y - 0.03); set(26, x - 0.17, y + 0.03);
    set(27, x - 0.34, y - 0.03); set(28, x - 0.34, y + 0.03);
  }
  return lm;
}

describe('FeatureExtractor', () => {
  const fe = () => new FeatureExtractor(params.features as any, ASPECT);
  it('classifies standing, sitting and lying with torso angle', () => {
    expect(fe().update(0, skeleton('stand')).posture).toBe('standing');
    expect(fe().update(0, skeleton('sit')).posture).toBe('sitting');
    const lying = fe().update(0, skeleton('lie'));
    expect(lying.posture).toBe('lying');
    expect(lying.torsoAngle).toBeGreaterThan(60);
    expect(fe().update(0, skeleton('stand')).torsoAngle).toBeLessThan(5);
  });
  it('is absent without landmarks or with low core visibility, and without ankles falls back to sitting/standing ratio', () => {
    expect(fe().update(0, null).present).toBe(false);
    expect(fe().update(0, skeleton('stand', 0.5, 0.2)).posture).toBe('absent');
    const noAnkles = skeleton('stand');
    noAnkles[27]![2] = 0.1; noAnkles[28]![2] = 0.1;
    expect(fe().update(0, noAnkles).posture).toBe('standing'); // default_leg_ratio 1.6 > 1.2
  });
  it('motion is zero for a still pose and positive for a moving one; reset() forgets the previous frame', () => {
    const f = fe();
    f.update(0, skeleton('stand'));
    expect(f.update(0.1, skeleton('stand')).motion).toBeCloseTo(0, 6);
    expect(f.update(0.2, skeleton('stand', 0.6)).motion).toBeGreaterThan(0.08);
    f.reset();
    expect(f.update(5, skeleton('stand', 0.2)).motion).toBeCloseTo(0, 6);
  });
  it('a sideways step above walk_speed is walking', () => {
    const f = fe();
    let s: BodyState = f.update(0, skeleton('stand', 0.2));
    for (let i = 1; i <= 15; i++) s = f.update(i / 15, skeleton('stand', 0.2 + i * 0.02));
    expect(s.posture).toBe('walking');
  });
});

describe('PresenceTracker', () => {
  const tr = () => new PresenceTracker(params.tracking);
  it('needs the enter span; a lone detection never creates presence', () => {
    const t = tr();
    const out = [0, 1, 2, 3, 4].map((i) => t.update(i / 15, skeleton('stand'), ASPECT));
    expect(out.map((o) => o[0] !== null)).toEqual([false, false, true, true, true]);
    const u = tr();
    expect([skeleton('stand'), null, null, skeleton('stand'), null].map((lm, i) => u.update(i / 15, lm, ASPECT)[0])).toEqual([null, null, null, null, null]);
  });
  it('bridges a short dropout (held) and drops presence after hold_s', () => {
    const t = tr();
    for (let i = 0; i < 6; i++) t.update(i / 15, skeleton('stand'), ASPECT);
    const held = t.update(6 / 15, null, ASPECT);
    expect(held[1]).toBe(true);
    expect(held[0]).not.toBeNull();
    const gone = t.update(6 / 15 + 1, null, ASPECT);
    expect(gone).toEqual([null, false]);
    expect(t.present).toBe(false);
  });
  it('skipGap keeps presence (no re-entry debounce) and a one-euro filter smooths jitter but passes a jump through raw', () => {
    const t = tr();
    for (let i = 0; i < 6; i++) t.update(i / 15, skeleton('stand'), ASPECT);
    t.skipGap(30);
    const o = t.update(6 / 15 + 30, skeleton('stand'), ASPECT);
    expect(o[0]).not.toBeNull();
    const a = tr();
    for (let i = 0; i < 6; i++) a.update(i / 15, skeleton('stand'), ASPECT);
    const jumped = skeleton('lie');
    const out = a.update(6 / 15, jumped, ASPECT)[0]!;
    expect(out[11]![0]).toBeCloseTo(jumped[11]![0]!, 9); // raw passthrough after a large jump
  });
});

const still = (t: number, over: Partial<BodyState> = {}): BodyState => ({
  t, present: true, height: 0.6, extent: 0.6, torsoAngle: 5, hipY: 0.55, center: [0.5, 0.55], motion: 0, speedX: 0,
  posture: 'standing', landmarks: null, ...over,
});

describe('FallDetector', () => {
  const mk = () => new FallDetector(params.events.fall, params.common.still_motion_bodysizes_s);
  /** stand until 1 s, drop to lying within 0.4 s, lie still */
  function drive(d: FallDetector, end: number, hook?: (t: number) => void): { kind: string; tConfirmed: number } | null {
    let ev = null;
    for (let t = 0; t <= end; t += 0.1) {
      hook?.(t);
      const lying = t >= 1.4;
      const ang = t < 1 ? 5 : lying ? 85 : 5 + (t - 1) * 200;
      const e = d.update(still(Number(t.toFixed(2)), { torsoAngle: ang, posture: ang > 60 ? 'lying' : 'standing', motion: t >= 1.4 ? 0 : 0.5 }));
      if (e) ev = e;
    }
    return ev;
  }
  it('confirms a sudden drop after still_seconds of stillness, with wellness wording', () => {
    const e = drive(mk(), 40)!;
    expect(e.kind).toBe('fall');
    expect(e.tConfirmed).toBeGreaterThan(20);
    expect(/check-in suggested/i.test((e as any).message)).toBe(true);
    expect(/diagnos/i.test((e as any).message)).toBe(false);
  });
  it('a slow lie-down is not a fall; standing back up cancels a pending fall', () => {
    const d = mk();
    for (let t = 0; t < 30; t += 0.1) {
      const ang = Math.min(90, 5 + t * 10); // 10 deg/s
      expect(d.update(still(t, { torsoAngle: ang, posture: ang > 60 ? 'lying' : 'standing' }))).toBeNull();
    }
    const d2 = mk();
    d2.update(still(0)); d2.update(still(0.5)); d2.update(still(0.8, { torsoAngle: 85, posture: 'lying' }));
    for (let t = 1; t < 30; t += 0.5) {
      const e = d2.update(still(t, t < 3 ? { torsoAngle: 85, posture: 'lying' } : {})); // up again at 3 s
      expect(e).toBeNull();
    }
  });
  it('pause(): a camera gap is not stillness and does not erase the pending fall', () => {
    const naive = drive(mk(), 40)!;
    const d = mk();
    let gapped: { tConfirmed: number } | null = null;
    // same story, but 100 s of unobserved camera after 3 s: timestamps jump, pause() is told
    for (let t = 0; t <= 140; t += 0.1) {
      const tt = Number(t.toFixed(2));
      const jump = tt >= 3.05 ? 100 : 0;
      if (tt === 3.1) d.pause(100);
      const lying = tt >= 1.4;
      const ang = tt < 1 ? 5 : lying ? 85 : 5 + (tt - 1) * 200;
      const e = d.update(still(tt + jump, { torsoAngle: ang, posture: ang > 60 ? 'lying' : 'standing', motion: tt >= 1.4 ? 0 : 0.5 }));
      if (e) gapped = e as any;
    }
    expect(gapped).not.toBeNull();
    expect(gapped!.tConfirmed - 100).toBeGreaterThan(naive.tConfirmed - 0.2); // gap excluded: same observed stillness needed
    expect(gapped!.tConfirmed - 100).toBeLessThan(naive.tConfirmed + 1);
  });
  it('held frames never trigger a drop and their carried motion keeps resetting stillness', () => {
    const d = mk();
    d.update(still(0)); d.update(still(0.5));
    expect(d.update(still(0.8, { torsoAngle: 85, posture: 'lying', motion: 0 }), true)).toBeNull(); // held: not a measurement
    for (let t = 1; t < 60; t += 0.5) expect(d.update(still(t, { torsoAngle: 85, posture: 'lying' }), true)).toBeNull();
  });
});

describe('InactivityDetector', () => {
  const still_ = params.common.still_motion_bodysizes_s;
  it('fires once after the threshold of stillness; movement or absence resets; gaps are excluded', () => {
    const d = new InactivityDetector(60, still_);
    const fired: number[] = [];
    for (let t = 0; t <= 130; t += 1) { const e = d.update(still(t)); if (e) fired.push(e.tConfirmed); }
    expect(fired).toEqual([60]);
    const d2 = new InactivityDetector(60, still_);
    for (let t = 0; t < 50; t++) d2.update(still(t));
    d2.update(still(50, { motion: 0.3 })); // moved
    expect(d2.update(still(110))).toBeNull(); // timer restarted
    const d3 = new InactivityDetector(60, still_);
    d3.update(still(0)); d3.pause(500);
    expect(d3.update(still(530))).toBeNull(); // 30 s observed
    expect(d3.update(still(561))!.kind).toBe('inactivity');
  });
  it('the demo threshold is the same detector with another number (default 120 s)', () => {
    const p = new PosePipeline(params as any, { inactivitySeconds: params.scheduling.inactivity_demo_default_s });
    expect(p.inactivitySeconds).toBe(120);
  });
});

describe('PosePipeline (landmarks -> tracker -> features -> detectors)', () => {
  it('reports a held frame and keeps the carried posture', () => {
    const p = new PosePipeline(params as any);
    let last;
    for (let i = 0; i < 8; i++) last = p.step(i / 15, skeleton('sit'), ASPECT);
    expect(last!.state.posture).toBe('sitting');
    const held = p.step(8 / 15, null, ASPECT);
    expect(held.held).toBe(true);
    expect(held.state.posture).toBe('sitting');
    expect(p.step(8 / 15 + 2, null, ASPECT).state.present).toBe(false);
  });
});

describe('scipy-compatible filter/spectrum building blocks (reference numbers from scipy 1.x)', () => {
  const x = Array.from({ length: 60 }, (_, i) => Math.sin((2 * Math.PI * 0.25 * i) / 15) + 0.3 * Math.cos((2 * Math.PI * 1.9 * i) / 15));
  it('butterworth band-pass + sosfiltfilt equals scipy', () => {
    const y = sosfiltfilt(butterBandpassSos(3, 0.1 / 7.5, 0.7 / 7.5), x);
    [[0, 0.18204933], [10, 0.79620054], [30, 0.23158298], [59, -0.10670708]].forEach(([i, v]) => expect(y[i!]).toBeCloseTo(v!, 6));
  });
  it('welch equals scipy (periodic hann, linear detrend, one-sided density)', () => {
    const { f, pxx } = welch(x, 15, 40, 20, 256);
    [[0, 0.14073454], [5, 0.27589859], [10, 0.13615686], [20, 0.00056], [30, 0.06499227]].forEach(([k, v]) => expect(pxx[k!]).toBeCloseTo(v!, 6));
    expect(f[5]).toBeCloseTo(0.29296875, 9);
  });
});

describe('breathing estimator and band', () => {
  const vit = params.vitals as any;
  const trace = (bpm: number, fps: number, secs: number) => Array.from({ length: secs * fps }, (_, i) => 1.5 * Math.sin((2 * Math.PI * bpm * i) / (60 * fps)));
  it('the band is 0.1-0.7 Hz (6-42 breaths/min) on both sides', () => {
    expect([vit.breathing.band_lo_hz, vit.breathing.band_hi_hz]).toEqual([0.1, 0.7]);
    expect([params.breathing_signal.band_lo_hz, params.breathing_signal.band_hi_hz]).toEqual([0.1, 0.7]);
  });
  it('recovers 36 breaths/min (above the old 0.5 Hz limit) and withholds noise', () => {
    const fps = 15, y = trace(36, fps, 40);
    const r = estimateWindow(y.map((_, i) => i / fps), y, 0.1, 0.7, 0.35, vit.signal)!;
    expect(r.reported).toBe(true);
    expect(Math.abs(r.value - 36)).toBeLessThan(1);
    let s = 1;
    const noise = y.map(() => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 2 ** 32 - 0.5; });
    const q = estimateWindow(noise.map((_, i) => i / fps), noise, 0.1, 0.7, 0.35, vit.signal)!;
    expect(q.reported).toBe(false);
  });
  it('emits one estimate per step once the window is 90 % full', () => {
    const e = new BreathingRateEstimator(vit, 0.35);
    const fps = 15, y = trace(15, fps, 60);
    const out = y.map((v, i) => e.updateDisplacement(i / fps, v).estimate).filter(Boolean);
    expect(out.length).toBeGreaterThan(10);
    expect(out[0]!.t).toBeGreaterThanOrEqual(27);
    e.reset();
    expect(e.updateDisplacement(100, 0)).toEqual({ due: false, estimate: null });
  });

  it('distinguishes "not due" from "due but failed" (flat trace)', () => {
    const e = new BreathingRateEstimator(vit, 0.35);
    const fps = 15;
    let due = 0, failed = 0;
    for (let i = 0; i < 60 * fps; i++) { const r = e.updateDisplacement(i / fps, 5); if (r.due) { due++; if (!r.estimate) failed++; } }
    expect(due).toBeGreaterThan(5);
    expect(failed).toBe(due);
  });
});
