import { describe, expect, it } from 'vitest';
import { isBlackSequence, isBlackStats, lumaStats, type ImageDataLike } from '../src/capture/blackframe';

function frame(fn: (i: number) => [number, number, number], w = 16, h = 12): ImageDataLike {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const [r, g, b] = fn(i);
    data.set([r, g, b, 255], i * 4);
  }
  return { data, width: w, height: h };
}
const black = () => frame(() => [0, 0, 0]);
const nearBlack = () => frame((i) => [i % 2, 1, 2]);
const gradient = () => frame((i) => [i % 256, (i * 2) % 256, 100]);
const darkRoom = () => frame((i) => [i % 2 ? 2 : 30, 5, 5]); // dark but textured

describe('black-frame detector', () => {
  it('flags an all-black frame', () => expect(isBlackStats(lumaStats(black()))).toBe(true));
  it('flags sensor-noise near-black', () => expect(isBlackStats(lumaStats(nearBlack()))).toBe(true));
  it('does not flag a normal image', () => expect(isBlackStats(lumaStats(gradient()))).toBe(false));
  it('does not flag a dark but varied scene', () => expect(isBlackStats(lumaStats(darkRoom()))).toBe(false));
  it('does not flag a uniformly bright frame', () => expect(isBlackStats(lumaStats(frame(() => [200, 200, 200])))).toBe(false));
  it('requires every sample to be black', () => {
    expect(isBlackSequence([black(), black(), black()])).toBe(true);
    expect(isBlackSequence([black(), gradient(), black()])).toBe(false);
  });
  it('empty sequence is not black', () => expect(isBlackSequence([])).toBe(false));
});
