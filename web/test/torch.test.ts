import { describe, expect, it, vi } from 'vitest';
import { describeMediaError, nextDeviceId, setTorch, supportsTorch } from '../src/capture/devices';

const track = (caps: unknown) => ({
  getCapabilities: () => caps as MediaTrackCapabilities,
  applyConstraints: vi.fn().mockResolvedValue(undefined),
});

describe('torch gating', () => {
  it('shown only when capability torch is true', () => {
    expect(supportsTorch(track({ torch: true }))).toBe(true);
    expect(supportsTorch(track({ torch: false }))).toBe(false);
    expect(supportsTorch(track({}))).toBe(false);
    expect(supportsTorch(null)).toBe(false);
    expect(supportsTorch({ getCapabilities: undefined } as never)).toBe(false);
    expect(supportsTorch({ getCapabilities: () => { throw new Error('x'); } })).toBe(false);
  });
  it('applies advanced torch constraint', async () => {
    const t = track({ torch: true });
    expect(await setTorch(t, true)).toBe(true);
    expect(t.applyConstraints).toHaveBeenCalledWith({ advanced: [{ torch: true }] });
  });
  it('does nothing without capability', async () => {
    const t = track({});
    expect(await setTorch(t, true)).toBe(false);
    expect(t.applyConstraints).not.toHaveBeenCalled();
  });
});

describe('device helpers', () => {
  it('cycles to the next camera', () => {
    const d = [{ deviceId: 'a' }, { deviceId: 'b' }, { deviceId: 'c' }];
    expect(nextDeviceId(d, 'a')).toBe('b');
    expect(nextDeviceId(d, 'c')).toBe('a');
    expect(nextDeviceId([{ deviceId: 'a' }], 'a')).toBeUndefined();
  });
  it('explains errors', () => {
    expect(describeMediaError({ name: 'NotAllowedError' })).toMatch(/denied/i);
    expect(describeMediaError({ name: 'NotReadableError' })).toMatch(/another app/i);
    expect(describeMediaError({ name: 'NotFoundError' })).toMatch(/no matching/i);
  });
});
