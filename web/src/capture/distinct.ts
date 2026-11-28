// Tells DISTINCT camera frames from repeated display callbacks (requestAnimationFrame fires at the display rate, e.g. 60 Hz for a
// 15 fps camera). A repeated frame must never become a sample: it would fake a high frame rate. Used by the hand test (W8) and the
// blink measurement (W9). Order of evidence: rVFC mediaTime once it has advanced, else the playback quality frame counter, else
// "unknown" (the caller compares the model output between frames and flags the timing as unreliable).
import type { FrameMeta } from './frames';

export type FrameVerdict = 'skip' | 'new' | 'unknown';

export class DistinctFrames {
  private lastMedia: number | null = null;
  private mediaAdvanced = false;
  private lastCounted: number | null = null;

  reset(): void {
    this.lastMedia = null;
    this.mediaAdvanced = false;
    this.lastCounted = null;
  }

  /** Call once per presented frame. `t` is the frame clock's time (null while it starts up: skip). 'skip' = a repeat of the
   *  previous camera frame (or no time yet); 'new' = a camera frame, known from a signal; 'unknown' = no signal. */
  check(video: HTMLVideoElement, meta: FrameMeta | undefined, t: number | null): FrameVerdict {
    const media = typeof meta?.mediaTime === 'number' && Number.isFinite(meta.mediaTime) ? meta.mediaTime : null;
    const prevMedia = this.lastMedia;
    if (media !== null) {
      this.lastMedia = media;
      if (prevMedia !== null && media !== prevMedia) this.mediaAdvanced = true;
    }
    if (t === null) return 'skip';
    if (this.mediaAdvanced && media === prevMedia) return 'skip';
    let counted: number | null = null;
    if (!this.mediaAdvanced) {
      const q = (video as { getVideoPlaybackQuality?: () => { totalVideoFrames: number } }).getVideoPlaybackQuality?.();
      counted = q && Number.isFinite(q.totalVideoFrames) && q.totalVideoFrames > 0 ? q.totalVideoFrames : null;
      if (counted !== null) {
        if (counted === this.lastCounted) return 'skip';
        this.lastCounted = counted;
      }
    }
    return this.mediaAdvanced || counted !== null ? 'new' : 'unknown';
  }
}
