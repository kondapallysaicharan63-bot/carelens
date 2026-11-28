// Face-ROI capture helpers for "Heart rate (face)": landmark polygons, face motion, skin filter, and the offscreen
// canvas that rasterises the polygons into a mask and averages the video pixels inside it. The signal processing
// lives in the DOM-free core (src/core/rppg.ts); everything here only produces its per-frame input.
import type { RppgParams, Rgb } from '../core/rppg';

export interface Pt { x: number; y: number }
export interface FaceRoiSample {
  rgb: Rgb;
  /** pixels inside the ROI polygons (before the skin filter) */
  px: number;
  /** fraction of those pixels kept by the skin filter and averaged (1 when the filter is off) */
  skinFrac: number;
  /** mean (R+G+B)/3 over the ROI, 0-255 */
  brightness: number;
}

export const ROI_CANVAS_W = 160; // keep the per-frame cost small on phones
export const EYE_L = 33, EYE_R = 263; // outer eye corners (normaliser for face motion)

export type RoiPolys = { forehead: Pt[]; left_cheek: Pt[]; right_cheek: Pt[] };

/** The three ROI polygons (normalised image coordinates) from a 478-point landmark list. */
export function roiPolygons(lm: ArrayLike<Pt>, idx: RppgParams['roi_landmarks']): RoiPolys {
  const pick = (ids: number[]) => ids.map((i) => ({ x: lm[i]!.x, y: lm[i]!.y }));
  return { forehead: pick(idx.forehead), left_cheek: pick(idx.left_cheek), right_cheek: pick(idx.right_cheek) };
}

export function polysCentroid(polys: RoiPolys): Pt {
  let x = 0, y = 0, n = 0;
  for (const poly of [polys.forehead, polys.left_cheek, polys.right_cheek]) for (const p of poly) { x += p.x; y += p.y; n++; }
  return { x: x / n, y: y / n };
}

/** Eye span (outer eye corners) in normalised units scaled by the frame aspect so x and y are comparable. */
export function eyeSpan(lm: ArrayLike<Pt>, aspect: number): number {
  return Math.hypot((lm[EYE_R]!.x - lm[EYE_L]!.x) * aspect, lm[EYE_R]!.y - lm[EYE_L]!.y);
}

/** Face motion: speed of the ROI centroid in eye spans per second (0 for the first frame or a non-positive dt). */
export function faceMotion(prev: Pt | null, cur: Pt, dt: number, span: number, aspect: number): number {
  if (!prev || !(dt > 0) || !(span > 0)) return 0;
  return Math.hypot((cur.x - prev.x) * aspect, cur.y - prev.y) / span / dt;
}

/** Full-range BT.601 YCrCb skin test with the ranges from params. */
export function isSkin(r: number, g: number, b: number, p: Pick<RppgParams, 'skin_cr_min' | 'skin_cr_max' | 'skin_cb_min' | 'skin_cb_max'>): boolean {
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  return cr >= p.skin_cr_min && cr <= p.skin_cr_max && cb >= p.skin_cb_min && cb <= p.skin_cb_max;
}

/** Mean RGB over the mask pixels that pass the skin filter (if asked). `px` counts all mask pixels; `skinFrac` the share
 *  kept. Null if the mask is empty; if no pixel passes the filter the sample has rgb [0,0,0] and skinFrac 0. */
export function maskedMean(
  img: ArrayLike<number>, mask: ArrayLike<number>, pixels: number, p: RppgParams, useSkin: boolean,
): FaceRoiSample | null {
  let r = 0, g = 0, b = 0, n = 0, inMask = 0;
  for (let i = 0; i < pixels; i++) {
    if (mask[4 * i + 3]! < 128) continue;
    inMask++;
    const R = img[4 * i]!, G = img[4 * i + 1]!, B = img[4 * i + 2]!;
    if (useSkin && !isSkin(R, G, B, p)) continue;
    r += R; g += G; b += B; n++;
  }
  if (inMask === 0) return null;
  if (n === 0) return { rgb: [0, 0, 0], px: inMask, skinFrac: 0, brightness: 0 };
  return { rgb: [r / n, g / n, b / n], px: inMask, skinFrac: n / inMask, brightness: (r + g + b) / (3 * n) };
}

/** The rectangle (in box coordinates) that a video of videoW x videoH occupies inside a boxW x boxH element with
 *  object-fit: contain (centred, letterboxed). Falls back to the whole box for unknown video size. */
export function contentRect(boxW: number, boxH: number, videoW: number, videoH: number): { x: number; y: number; w: number; h: number } {
  if (!(videoW > 0 && videoH > 0 && boxW > 0 && boxH > 0)) return { x: 0, y: 0, w: boxW, h: boxH };
  const s = Math.min(boxW / videoW, boxH / videoH);
  const w = videoW * s, h = videoH * s;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

/** Offscreen canvases: video -> <=160 px wide frame, polygons -> mask (Path2D), masked mean. */
export class FaceRoiSampler {
  private readonly vc = document.createElement('canvas');
  private readonly mc = document.createElement('canvas');
  private readonly vctx = this.vc.getContext('2d', { willReadFrequently: true });
  private readonly mctx = this.mc.getContext('2d', { willReadFrequently: true });

  constructor(private readonly p: RppgParams) {}

  sample(video: HTMLVideoElement, polys: RoiPolys): FaceRoiSample | null {
    if (!this.vctx || !this.mctx || video.videoWidth === 0) return null;
    const w = ROI_CANVAS_W, h = Math.max(1, Math.round((w * video.videoHeight) / video.videoWidth));
    if (this.vc.width !== w || this.vc.height !== h) { this.vc.width = this.mc.width = w; this.vc.height = this.mc.height = h; }
    this.vctx.drawImage(video, 0, 0, w, h);
    this.mctx.clearRect(0, 0, w, h);
    const path = new Path2D();
    for (const poly of [polys.forehead, polys.left_cheek, polys.right_cheek]) {
      poly.forEach((q, i) => (i === 0 ? path.moveTo(q.x * w, q.y * h) : path.lineTo(q.x * w, q.y * h)));
      path.closePath();
    }
    this.mctx.fillStyle = '#fff';
    this.mctx.fill(path);
    return maskedMean(this.vctx.getImageData(0, 0, w, h).data, this.mctx.getImageData(0, 0, w, h).data, w * h, this.p, this.p.skin_filter);
  }
}
