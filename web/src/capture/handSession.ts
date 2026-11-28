// Frame loop for the hand test: loads the hand model on demand, runs it on every presented video frame and hands
// {t, landmarks, score} to the DOM-free tremor core. end() is safe at any time (also during model load) and closes the model.
// Nothing leaves the device. Not a medical device.
import type { HandFrame } from '../core/tremor';
import type { FrameMeta } from './frames';
import type { HandHandle } from './hand';
import { FrameClock } from './ppgSampler';

export type FrameSubscriber = (video: HTMLVideoElement, cb: (nowMs: number, meta?: FrameMeta) => void) => () => void;

export class HandSession {
  private handle: HandHandle | null = null;
  private stopLoop: (() => void) | null = null;
  private ended = false;
  private clock = new FrameClock();
  private lastMedia: number | null = null;
  private mediaAdvanced = false;
  private lastCounted: number | null = null;
  private lastLm: number[][] | null = null;
  /** false once a frame arrived with no way to tell distinct camera frames from repeated display callbacks */
  timingReliable = true;

  constructor(
    private readonly subscribe: FrameSubscriber,
    private readonly onFrame: (f: HandFrame, aspect: number) => void,
  ) {}

  get isEnded(): boolean { return this.ended; }

  async load(loader: () => Promise<HandHandle>, video: HTMLVideoElement): Promise<void> {
    const h = await loader();
    if (this.ended) { h.close(); return; }
    this.handle = h;
    this.clock.reset();
    this.stopLoop = this.subscribe(video, (now, meta) => this.frame(video, now, meta));
  }

  private frame(video: HTMLVideoElement, now: number, meta?: FrameMeta): void {
    if (this.ended || !this.handle || video.videoWidth === 0 || (video.readyState ?? 4) < 2) return;
    const t = this.clock.time(now, meta);
    // Only DISTINCT camera frames may become samples: a repeated frame (requestAnimationFrame fires at the display rate, e.g. 60 Hz
    // for a 15 fps camera) would fake a high sample rate and could alias a fast movement to a low frequency.
    const media = typeof meta?.mediaTime === 'number' && Number.isFinite(meta.mediaTime) ? meta.mediaTime : null;
    const prevMedia = this.lastMedia;
    if (media !== null) {
      this.lastMedia = media;
      if (prevMedia !== null && media !== prevMedia) this.mediaAdvanced = true;
    }
    if (t === null) return;
    if (this.mediaAdvanced && media === prevMedia) return;
    let counted: number | null = null;
    if (!this.mediaAdvanced) {
      const q = (video as { getVideoPlaybackQuality?: () => { totalVideoFrames: number } }).getVideoPlaybackQuality?.();
      counted = q && Number.isFinite(q.totalVideoFrames) && q.totalVideoFrames > 0 ? q.totalVideoFrames : null;
      if (counted !== null) {
        if (counted === this.lastCounted) return;
        this.lastCounted = counted;
      }
    }
    const signal = this.mediaAdvanced || counted !== null;
    let det = null;
    try { det = this.handle.detect(video, now); } catch { /* a failed frame counts as "no hand" */ }
    if (!signal) {
      this.timingReliable = false; // last resort: identical landmarks mean the same camera frame again
      if (det && this.lastLm && sameLandmarks(det.lm, this.lastLm)) return;
      this.lastLm = det ? det.lm : null;
    }
    this.onFrame({ t, lm: det ? det.lm : null, score: det ? det.score : 0 }, video.videoWidth / video.videoHeight);
  }

  end(): void {
    this.ended = true;
    this.stopLoop?.();
    this.stopLoop = null;
    this.handle?.close();
    this.handle = null;
  }
}

function sameLandmarks(a: number[][], b: number[][]): boolean {
  for (let i = 0; i < a.length; i++) if (a[i]![0] !== b[i]![0] || a[i]![1] !== b[i]![1]) return false;
  return true;
}
