// Fingertip PPG capture helpers: ROI statistics from a small canvas and frame timestamps.
// The signal processing itself lives in the DOM-free core (src/core/ppg.ts).
import type { FrameMeta } from './frames';

export const ROI_W = 64;
export const ROI_H = 48;

export interface RoiSample {
  r: number;
  g: number;
  b: number;
  /** fraction of ROI pixels with R >= 250 */
  clipFrac: number;
}

/** Mean R, G, B and R-clip fraction over an RGBA buffer. */
export function roiStats(data: ArrayLike<number>, pixels: number): RoiSample {
  let r = 0, g = 0, b = 0, clip = 0;
  for (let i = 0; i < pixels; i++) {
    const R = data[4 * i]!;
    r += R; g += data[4 * i + 1]!; b += data[4 * i + 2]!;
    if (R >= 250) clip++;
  }
  const n = Math.max(1, pixels);
  return { r: r / n, g: g / n, b: b / n, clipFrac: clip / n };
}

/** Draws the video to a 64x48 canvas and averages its central 50% x 50%. Returns null until the video has frames. */
export class RoiSampler {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D | null;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = ROI_W;
    this.canvas.height = ROI_H;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
  }

  sample(video: HTMLVideoElement): RoiSample | null {
    if (!this.ctx || video.readyState < 2 || video.videoWidth === 0) return null;
    this.ctx.drawImage(video, 0, 0, ROI_W, ROI_H);
    const x0 = ROI_W / 4, y0 = ROI_H / 4, w = ROI_W / 2, h = ROI_H / 2;
    return roiStats(this.ctx.getImageData(x0, y0, w, h).data, w * h);
  }
}

/**
 * Frame timestamps in seconds. Prefers requestVideoFrameCallback's mediaTime, then expectedDisplayTime,
 * then the callback time (performance.now timeline). The source is fixed on the second frame, by which
 * one of them actually advances; the first frame yields null so timelines are never mixed.
 */
export class FrameClock {
  private mode: 'media' | 'display' | 'now' | null = null;
  private first: { media?: number; display?: number } | null = null;

  time(nowMs: number, meta?: FrameMeta): number | null {
    const media = typeof meta?.mediaTime === 'number' && Number.isFinite(meta.mediaTime) ? meta.mediaTime : undefined;
    const display = typeof meta?.expectedDisplayTime === 'number' && Number.isFinite(meta.expectedDisplayTime) ? meta.expectedDisplayTime / 1000 : undefined;
    if (this.mode === null) {
      if (this.first === null) {
        this.first = { media, display };
        return null;
      }
      const adv = (a?: number, b?: number) => a !== undefined && b !== undefined && b > a && b - a < 1;
      this.mode = adv(this.first.media, media) ? 'media' : adv(this.first.display, display) ? 'display' : 'now';
    }
    if (this.mode === 'media') return media ?? null;
    if (this.mode === 'display') return display ?? null;
    return nowMs / 1000;
  }

  reset(): void {
    this.mode = null;
    this.first = null;
  }
}
