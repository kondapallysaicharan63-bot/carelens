// Face-HR capture helpers and UI text (DOM-free parts). The signal core is tested in rppg.test.ts.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { contentRect, eyeSpan, faceMotion, isSkin, maskedMean, polysCentroid, roiPolygons, type Pt } from '../src/capture/faceRoi';
import type { RppgParams, RppgReason } from '../src/core/rppg';
import { FINGERTIP_HINT, LIMITS_NOTE, guidance } from '../src/ui/faceHr';

const SHARED = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'shared');
const P: RppgParams = JSON.parse(readFileSync(join(SHARED, 'params.json'), 'utf8')).rppg;

const lm: Pt[] = Array.from({ length: 478 }, (_, i) => ({ x: (i % 40) / 40, y: Math.floor(i / 40) / 12 }));

describe('face ROI helpers', () => {
  it('builds the three polygons from the landmark indices in params', () => {
    const polys = roiPolygons(lm, P.roi_landmarks);
    expect(polys.forehead).toHaveLength(P.roi_landmarks.forehead.length);
    expect(polys.left_cheek[0]).toEqual(lm[P.roi_landmarks.left_cheek[0]!]);
    expect(polysCentroid(polys).x).toBeGreaterThan(0);
  });
  it('face motion is in eye spans per second, frame-rate independent, 0 without history', () => {
    const span = eyeSpan(lm, 1);
    expect(span).toBeGreaterThan(0);
    expect(faceMotion(null, { x: 0.5, y: 0.5 }, 0.033, span, 1)).toBe(0);
    const a = faceMotion({ x: 0.5, y: 0.5 }, { x: 0.5 + 0.1 * span, y: 0.5 }, 0.1, span, 1);
    expect(a).toBeCloseTo(1, 9); // 0.1 span in 0.1 s
    const b = faceMotion({ x: 0.5, y: 0.5 }, { x: 0.5 + 0.05 * span, y: 0.5 }, 0.05, span, 1);
    expect(b).toBeCloseTo(a, 9);
  });
  it('YCrCb skin filter keeps skin tones and drops grey/black/blue (not every dark brown: the polygons avoid hair)', () => {
    expect(isSkin(224, 172, 150, P)).toBe(true);
    expect(isSkin(180, 125, 95, P)).toBe(true);
    expect(isSkin(20, 20, 20, P)).toBe(false);   // black hair/shadow
    expect(isSkin(120, 130, 125, P)).toBe(false); // grey wall
    expect(isSkin(30, 90, 200, P)).toBe(false);   // blue
  });
  it('maskedMean averages only masked (and skin) pixels', () => {
    // 4 pixels: skin A (masked), skin B (masked), hair (masked), skin C (not masked)
    const img = [224, 172, 150, 255, 180, 125, 95, 255, 20, 20, 20, 255, 10, 10, 10, 255];
    const mask = [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0];
    const s = maskedMean(img, mask, 4, P, true)!;
    expect(s.px).toBe(3);
    expect(s.skinFrac).toBeCloseTo(2 / 3, 9);
    expect(s.rgb[0]).toBeCloseTo(202, 9);
    const all = maskedMean(img, mask, 4, P, false)!;
    expect(all.px).toBe(3);
    expect(all.skinFrac).toBe(1);
    const none = maskedMean([20, 20, 20, 255], [0, 0, 0, 255], 1, P, true)!;
    expect(none).toMatchObject({ px: 1, skinFrac: 0, brightness: 0 }); // all rejected by the filter: skin_filter, not too_small
    expect(maskedMean(img, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 4, P, true)).toBeNull();
  });
});

describe('contentRect (object-fit: contain letterboxing)', () => {
  it('maps the video into the displayed content rectangle', () => {
    expect(contentRect(400, 300, 640, 480)).toEqual({ x: 0, y: 0, w: 400, h: 300 });
    const wide = contentRect(400, 200, 640, 480); // pillarboxed: height-limited
    expect(wide.h).toBeCloseTo(200, 9); expect(wide.w).toBeCloseTo(266.6667, 3); expect(wide.x).toBeCloseTo(66.6667, 3); expect(wide.y).toBe(0);
    const tall = contentRect(400, 400, 1280, 720); // letterboxed: width-limited
    expect(tall.w).toBe(400); expect(tall.h).toBeCloseTo(225, 9); expect(tall.y).toBeCloseTo(87.5, 9);
    expect(contentRect(300, 200, 0, 0)).toEqual({ x: 0, y: 0, w: 300, h: 200 });
  });
});

describe('face card text', () => {
  it('maps every reason to guidance, with the specified wording', () => {
    expect(guidance('no_face')).toBe('Face the camera');
    expect(guidance('too_small')).toBe('Move closer');
    expect(guidance('motion')).toBe('Hold still');
    expect(guidance('lighting')).toBe('Use steady, even light — avoid flicker and backlight');
    expect(guidance('settling')).toBe('Settling…');
    expect(guidance('skin_filter')).toBe('Use neutral, even light — very warm or coloured light hides the skin signal');
    const all: RppgReason[] = ['ok', 'no_face', 'too_small', 'skin_filter', 'motion', 'lighting', 'settling', 'low_signal', 'disagree', 'gaps'];
    for (const r of all) expect(guidance(r).length).toBeGreaterThan(3);
  });
  it('states the limits and never uses diagnostic wording', () => {
    expect(LIMITS_NOTE).toBe('Camera heart-rate estimates are less reliable in low or uneven light, with movement, and can vary with skin tone.');
    expect(FINGERTIP_HINT.toLowerCase()).toContain('fingertip');
    for (const s of [LIMITS_NOTE, FINGERTIP_HINT]) expect(s.toLowerCase()).not.toMatch(/diagnos/);
  });
});
