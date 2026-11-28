import { describe, expect, it } from 'vitest';
import { qualityLevel, qualityText } from '../src/ui/quality';

describe('quality mapping', () => {
  it.each([
    [0, 'weak'], [0.33, 'weak'], [0.34, 'ok'], [0.5, 'ok'], [0.66, 'ok'], [0.67, 'good'], [1, 'good'],
    [-1, 'weak'], [5, 'good'],
  ])('%s -> %s', (q, lvl) => expect(qualityLevel(q)).toBe(lvl));
  it('handles missing values', () => {
    expect(qualityLevel(null)).toBe('none');
    expect(qualityLevel(undefined)).toBe('none');
    expect(qualityLevel(NaN)).toBe('none');
  });
  it('always has a text label', () => {
    expect(qualityText(0.1)).toContain('weak');
    expect(qualityText(0.5)).toContain('ok');
    expect(qualityText(0.9)).toContain('good');
  });
});
