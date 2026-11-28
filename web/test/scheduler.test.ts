import { describe, expect, it } from 'vitest';
import { PoseScheduler } from '../src/core/scheduler';
import params from '../../shared/params.json';

const P = params.scheduling;

describe('PoseScheduler', () => {
  it('runs pose at about pose_hz on a 30 fps frame stream', () => {
    const s = new PoseScheduler(P);
    let runs = 0;
    for (let i = 0; i < 300; i++) if (s.shouldRunPose(i * (1000 / 30))) runs++; // 10 s
    expect(runs).toBeGreaterThanOrEqual(65);
    expect(runs).toBeLessThanOrEqual(81);
  });
  it('an over-budget frame skips the next due pose frame instead of queueing', () => {
    const s = new PoseScheduler(P);
    const ran: number[] = [];
    for (let i = 0; i < 90; i++) {
      const now = i * (1000 / 30);
      if (s.shouldRunPose(now)) {
        ran.push(now);
        if (ran.length === 3) s.reportDuration(P.pose_budget_ms + 1); // slow frame
      }
    }
    expect(s.skipped).toBe(1);
    const gaps = ran.slice(1).map((t, k) => t - ran[k]!);
    expect(Math.max(...gaps)).toBeGreaterThan(200); // one period skipped
    expect(Math.max(...gaps)).toBeLessThan(300); // but never a burst or a longer stall
  });
  it('a within-budget frame does not skip', () => {
    const s = new PoseScheduler(P);
    expect(s.shouldRunPose(0)).toBe(true);
    s.reportDuration(P.pose_budget_ms - 1);
    expect(s.shouldRunPose(140)).toBe(true);
    expect(s.skipped).toBe(0);
  });
  it('does not burst to catch up after a long stall', () => {
    const s = new PoseScheduler(P);
    expect(s.shouldRunPose(0)).toBe(true);
    expect(s.shouldRunPose(5000)).toBe(true); // late
    expect(s.shouldRunPose(5033)).toBe(false);
    expect(s.shouldRunPose(5066)).toBe(false);
  });
  it('gapBefore: normal cadence is 0, a long timestamp jump is an outage without the usual period', () => {
    const s = new PoseScheduler(P);
    expect(s.gapBefore(10)).toBe(0);
    expect(s.gapBefore(10.13)).toBe(0);
    expect(s.gapBefore(10.9)).toBe(0); // below gap_min_s
    expect(s.gapBefore(40.9)).toBeCloseTo(30 - 1 / P.pose_hz, 6);
    s.reset();
    expect(s.gapBefore(100)).toBe(0);
  });
  it('face landmarker runs every frame only while its card is active', () => {
    expect(PoseScheduler.shouldRunFace(true)).toBe(true);
    expect(PoseScheduler.shouldRunFace(false)).toBe(false);
  });
  it('is configured for ~8 Hz pose and demo inactivity options include the default', () => {
    expect(P.pose_hz).toBe(8);
    expect(P.inactivity_demo_options_s).toContain(P.inactivity_demo_default_s);
  });
});
