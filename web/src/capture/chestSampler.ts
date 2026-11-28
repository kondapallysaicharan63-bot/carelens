// Platform side of the breathing signal: draws the current video frame to a small grayscale buffer and hands it to the
// DOM-free shift tracker (src/core/chestShift.ts). The chest box comes from pose landmarks (src/core/breathing.ts).
import type { Box, ChestImageSource, ShiftResult } from '../core/breathing';
import { GrayShiftTracker, type ChestShiftParams } from '../core/chestShift';

export class VideoChestSource implements ChestImageSource {
  private readonly canvas = document.createElement('canvas');
  private readonly ctx = this.canvas.getContext('2d', { willReadFrequently: true });
  private readonly tracker: GrayShiftTracker;

  constructor(private readonly video: HTMLVideoElement, private readonly p: ChestShiftParams) {
    this.tracker = new GrayShiftTracker(p);
  }

  size(): { w: number; h: number } | null {
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!(vw > 0 && vh > 0)) return null;
    return { w: this.p.frame_w_px, h: Math.max(1, Math.round((this.p.frame_w_px * vh) / vw)) };
  }

  shiftAndAdvance(box: Box): ShiftResult {
    const sz = this.size();
    if (!sz || !this.ctx) return 'weak';
    if (this.canvas.width !== sz.w || this.canvas.height !== sz.h) { this.canvas.width = sz.w; this.canvas.height = sz.h; this.tracker.reset(); }
    this.ctx.drawImage(this.video, 0, 0, sz.w, sz.h);
    const rgba = this.ctx.getImageData(0, 0, sz.w, sz.h).data;
    const gray = new Uint8ClampedArray(sz.w * sz.h);
    for (let i = 0; i < gray.length; i++) gray[i] = 0.299 * rgba[4 * i]! + 0.587 * rgba[4 * i + 1]! + 0.114 * rgba[4 * i + 2]!;
    return this.tracker.shift(gray, sz.w, box);
  }

  reset(): void { this.tracker.reset(); }
}
