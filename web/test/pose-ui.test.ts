import { describe, expect, it } from 'vitest';
import { EventLog, FALL_TEXT, INACTIVITY_TEXT, eventText } from '../src/ui/eventLog';
import { BREATHING_HINT, BREATHING_NOTE, CHECKIN_NOTE, breathingGuidance, formatClock, postureLabel } from '../src/ui/poseCards';

describe('check-in wording', () => {
  it('uses the agreed wellbeing wording and never emergency/diagnosis claims', () => {
    expect(FALL_TEXT).toBe('Possible fall — check-in suggested');
    expect(INACTIVITY_TEXT).toBe('Long time without movement — check-in suggested');
    expect(eventText('fall')).toBe(FALL_TEXT);
    expect(eventText('inactivity')).toBe(INACTIVITY_TEXT);
    for (const t of [FALL_TEXT, INACTIVITY_TEXT, BREATHING_HINT, BREATHING_NOTE, CHECKIN_NOTE]) {
      expect(t).not.toMatch(/emergenc|diagnos|apnea|detected a fall|call(ing)? (911|112|999)|alert(ing)? (someone|family)/i);
    }
    expect(BREATHING_HINT).toBe('Sit still facing the camera with your chest visible');
    expect(CHECKIN_NOTE).toMatch(/does not call anyone/);
  });
});

describe('EventLog', () => {
  it('keeps events in order, drives the banner from the newest unacknowledged one, and acknowledges once', () => {
    const log = new EventLog();
    expect(log.pending).toBeNull();
    const a = log.add({ kind: 'fall' }, 1000);
    const b = log.add({ kind: 'inactivity' }, 2000);
    expect(log.pending!.id).toBe(b.id);
    expect(log.pendingCount).toBe(2);
    expect(log.acknowledge(b.id)).toBe(true);
    expect(log.acknowledge(b.id)).toBe(false);
    expect(log.pending!.id).toBe(a.id);
    log.acknowledge(a.id);
    expect(log.pending).toBeNull();
    expect(log.acknowledge(999)).toBe(false);
    expect(log.entries.map((e) => e.text)).toEqual([FALL_TEXT, INACTIVITY_TEXT]);
  });
});

describe('card text helpers', () => {
  it('posture labels (standing/sitting/lying/—)', () => {
    expect(['standing', 'sitting', 'lying', 'absent'].map((p) => postureLabel(p as never))).toEqual(['standing', 'sitting', 'lying', '—']);
    expect(postureLabel('walking')).toBe('standing (walking)');
  });
  it('breathing guidance tracks why there is no reading', () => {
    expect(breathingGuidance('absent', false, false)).toBe(BREATHING_HINT);
    expect(breathingGuidance('moving', false, false)).toBe('Hold still');
    expect(breathingGuidance('ok', false, false)).toMatch(/Collecting/);
    expect(breathingGuidance('ok', false, true)).toMatch(/weak/);
    expect(breathingGuidance('ok', true, true)).toMatch(/Good signal/);
  });
  it('formats wall-clock time', () => {
    expect(formatClock(new Date(2026, 0, 2, 3, 4, 5).getTime())).toBe('03:04:05');
  });
});
