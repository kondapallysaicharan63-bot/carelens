// Fingertip mode plumbing: ROI statistics, frame clock, 30 s measurement, sparkline, guidance. Not a medical device.
import { describe, expect, it } from 'vitest';
import { FrameClock, roiStats } from '../src/capture/ppgSampler';
import { MEASURE_S, MeasurementTracker, guidance, sparklinePoints } from '../src/ui/fingertip';
import type { PpgReason } from '../src/core/ppg';

describe('roiStats', () => {
  it('averages channels and counts R >= 250 as clipped', () => {
    const d = [250, 10, 0, 255, 255, 20, 0, 255, 100, 30, 0, 255, 90, 40, 0, 255];
    const s = roiStats(d, 4);
    expect(s.r).toBeCloseTo((250 + 255 + 100 + 90) / 4);
    expect(s.g).toBeCloseTo(25);
    expect(s.clipFrac).toBe(0.5);
  });
});

describe('FrameClock', () => {
  it('uses mediaTime once it advances, never mixing timelines (first frame yields null)', () => {
    const c = new FrameClock();
    expect(c.time(1000, { mediaTime: 5.0 })).toBeNull();
    expect(c.time(1033, { mediaTime: 5.033 })).toBeCloseTo(5.033);
    expect(c.time(1066, { mediaTime: 5.066 })).toBeCloseTo(5.066);
  });
  it('falls back to expectedDisplayTime, then performance.now timeline', () => {
    const d = new FrameClock();
    d.time(0, { mediaTime: 0, expectedDisplayTime: 100 });
    expect(d.time(33, { mediaTime: 0, expectedDisplayTime: 133 })).toBeCloseTo(0.133);
    const n = new FrameClock();
    expect(n.time(500)).toBeNull();
    expect(n.time(533)).toBeCloseTo(0.533);
  });
});

const upd = (t: number, reason: PpgReason, bpm: number | null = null, contact_epoch = 1) => ({ t, reason, reported: reason === 'ok', bpm, contact_epoch });

describe('MeasurementTracker', () => {
  it('reports the median of reported windows after 30 s', () => {
    const m = new MeasurementTracker();
    let st = m.push(upd(1, 'settling'));
    for (let t = 2; t <= 8; t++) st = m.push(upd(t, 'settling'));
    for (let t = 9; t <= 31; t++) st = m.push(upd(t, 'ok', 70 + (t % 3)));
    expect(st.done).toBe(true);
    expect(st.medianBpm).toBe(71);
    expect(st.reported).toBe(23);
    expect(m.push(upd(40, 'ok', 200))).toEqual(st); // result is frozen
  });
  it('restarts when the core contact epoch changes, even without a no_finger update', () => {
    const m = new MeasurementTracker();
    for (let t = 1; t <= 20; t++) m.push(upd(t, 'ok', 60, 1));
    const st = m.push(upd(21, 'settling', null, 2));
    expect(st.elapsedS).toBe(0);
    expect(st.windows).toBe(1);
    expect(st.reported).toBe(0);
  });
  it('restarts when the finger is lost; too few readings give no result', () => {
    const m = new MeasurementTracker();
    for (let t = 1; t <= 20; t++) m.push(upd(t, 'ok', 60));
    expect(m.push(upd(21, 'no_finger')).elapsedS).toBe(0);
    let st = m.push(upd(22, 'settling'));
    expect(st.elapsedS).toBe(0);
    for (let t = 23; t <= 22 + MEASURE_S; t++) st = m.push(upd(t, 'low_signal', 60));
    expect(st.done).toBe(true);
    expect(st.medianBpm).toBeNull();
  });
});

describe('sparklinePoints and guidance', () => {
  it('maps samples into the viewBox, centred, and ignores tiny noise', () => {
    const pts = sparklinePoints([0, 1, 0, -1, 0], 100, 50).split(' ').map((p) => p.split(',').map(Number));
    expect(pts[0]![0]).toBe(0);
    expect(pts[4]![0]).toBe(100);
    expect(pts[1]![1]!).toBeLessThan(pts[3]![1]!);
    for (const [, y] of pts) expect(y!).toBeGreaterThanOrEqual(0);
    const flat = sparklinePoints([100, 100.01, 99.99, 100], 100, 50).split(' ').map((p) => Number(p.split(',')[1]));
    for (const y of flat) expect(Math.abs(y - 25)).toBeLessThan(1);
    expect(sparklinePoints([1], 10, 10)).toBe('');
  });
  it('has wellbeing guidance for every reason', () => {
    const reasons: PpgReason[] = ['ok', 'no_finger', 'saturated', 'settling', 'low_signal', 'disagree', 'gaps'];
    for (const r of reasons) expect(guidance(r, 3).length).toBeGreaterThan(5);
    expect(guidance('no_finger')).toContain('Cover the camera and flash fully with your fingertip');
    expect(guidance('saturated')).toContain('Press more lightly');
    expect(guidance('settling', 2.2)).toContain('Hold still');
    expect(guidance('settling', 2.2)).toContain('3 s');
    for (const r of [...reasons, 'idle' as const]) expect(guidance(r)).not.toMatch(/diagnos|medical|heart (attack|disease)/i);
  });
});
