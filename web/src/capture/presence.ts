export interface PoseHandle {
  detect(video: HTMLVideoElement, ts: number): boolean;
  close(): void;
}
export type FrameSubscriber = (video: HTMLVideoElement, cb: (nowMs: number) => void) => () => void;

/**
 * Owns the pose model + frame loop for one capture session. The loop can be re-bound after the
 * stream is swapped (restart), and end() is safe at any time, including during model load.
 */
export class PresenceSession {
  private handle: PoseHandle | null = null;
  private stopLoop: (() => void) | null = null;
  private ended = false;
  private video: HTMLVideoElement | null = null;
  private lastRun = 0;

  constructor(
    private readonly subscribe: FrameSubscriber,
    private readonly onResult: (present: boolean) => void,
    private readonly minIntervalMs = 100,
  ) {}

  get isEnded(): boolean {
    return this.ended;
  }

  /** Loads the model (async). If end() happened meanwhile, the model is closed immediately. */
  async load(loader: () => Promise<PoseHandle>, video: HTMLVideoElement): Promise<void> {
    this.video = video;
    const h = await loader();
    if (this.ended) {
      h.close();
      return;
    }
    this.handle = h;
    this.restart();
  }

  /** (Re)start the frame loop on the current video element; call after every stream swap. */
  restart(): void {
    this.stopLoop?.();
    this.stopLoop = null;
    if (this.ended || !this.handle || !this.video) return;
    const video = this.video;
    this.stopLoop = this.subscribe(video, (now) => {
      if (this.ended || !this.handle || video.videoWidth === 0 || now - this.lastRun < this.minIntervalMs) return;
      this.lastRun = now;
      this.onResult(this.handle.detect(video, now));
    });
  }

  end(): void {
    this.ended = true;
    this.stopLoop?.();
    this.stopLoop = null;
    this.handle?.close();
    this.handle = null;
  }
}
