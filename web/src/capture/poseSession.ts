// Owns the pose model + frame loop for one capture session and feeds the DOM-free core:
// landmarks -> PosePipeline (tracker, features, fall + inactivity) and BreathingChannel (chest shift -> breathing).
// Pose runs at ~scheduling.pose_hz (PoseScheduler); an over-budget frame skips the next pose frame. The loop can be
// re-bound after the stream is swapped (restart) and end() is safe at any time, including during model load.
// Nothing leaves the device and no alert sound or notification is raised. Not a medical device.
import params from '../../../shared/params.json';
import { BreathingChannel, type BreathingEstimate } from '../core/breathing';
import type { ChestImageSource } from '../core/chestShift';
import type { PoseEvent } from '../core/events';
import type { BodyState } from '../core/features';
import { PosePipeline, type PoseChainParams } from '../core/pipeline';
import { PoseScheduler } from '../core/scheduler';
import { FrameClock } from './ppgSampler';
import type { FrameMeta } from './frames';
import type { PoseHandle } from './pose';

export type FrameSubscriber = (video: HTMLVideoElement, cb: (nowMs: number, meta?: FrameMeta) => void) => () => void;

export interface PoseView {
  t: number;
  state: BodyState;
  held: boolean;
  breathing: { status: BreathingChannel['status']; estimate: BreathingEstimate | null };
}

export interface PoseHooks {
  onView(v: PoseView): void;
  onEvent(e: PoseEvent): void;
  /** processing errors are being retried (true) / have cleared (false) */
  onError?(failing: boolean): void;
}

/** consecutive frame errors before the landmarker is re-created */
export const MAX_CONSECUTIVE_ERRORS = 10;

export class PoseSession {
  private handle: PoseHandle | null = null;
  private stopLoop: (() => void) | null = null;
  private ended = false;
  private paused = false;
  private video: HTMLVideoElement | null = null;
  private clock = new FrameClock();
  private scheduler = new PoseScheduler(params.scheduling);
  private pipeline: PosePipeline;
  private breathing: BreathingChannel;
  private lastEstimate: BreathingEstimate | null = null;
  private loader: (() => Promise<PoseHandle>) | null = null;
  private errors = 0;
  private recreating = false;
  private forceGap = false;
  private lastGoodT: number | null = null;
  private lastEstimateT = 0;
  private inactivitySeconds = params.scheduling.inactivity_demo_default_s;

  constructor(
    private readonly subscribe: FrameSubscriber,
    private readonly hooks: PoseHooks,
    private readonly makeChestSource: (video: HTMLVideoElement) => ChestImageSource,
    private readonly nowMs: () => number = () => performance.now(),
  ) {
    this.pipeline = this.newPipeline();
    this.breathing = this.newBreathing();
  }

  private newPipeline(): PosePipeline {
    return new PosePipeline(params as unknown as PoseChainParams, { inactivitySeconds: this.inactivitySeconds });
  }
  private newBreathing(): BreathingChannel {
    return new BreathingChannel(params.vitals as never, params.common);
  }

  get isEnded(): boolean {
    return this.ended;
  }

  /** Demo inactivity threshold: the same detector with another number; the stillness timer restarts. */
  setInactivitySeconds(s: number): void {
    this.inactivitySeconds = s;
    this.pipeline.setInactivitySeconds(s);
  }

  /** Hold the frame loop (the hand test needs the whole CPU for its frame rate); resume() starts timing and detectors over. */
  setPaused(p: boolean): void {
    if (this.paused === p) return;
    this.paused = p;
    if (p) { this.stopLoop?.(); this.stopLoop = null; } else this.restart();
  }

  /** Loads the model (async). If end() happened meanwhile, the model is closed immediately. */
  async load(loader: () => Promise<PoseHandle>, video: HTMLVideoElement): Promise<void> {
    this.video = video;
    this.loader = loader;
    const h = await loader();
    if (this.ended) {
      h.close();
      return;
    }
    this.handle = h;
    this.restart();
  }

  /** (Re)start the frame loop on the current video element; call after every stream swap. Timing and detector state start over. */
  restart(): void {
    this.stopLoop?.();
    this.stopLoop = null;
    if (this.ended || this.paused || !this.handle || !this.video) return;
    this.clock.reset();
    this.scheduler.reset();
    this.pipeline = this.newPipeline();
    this.breathing = this.newBreathing();
    this.lastEstimate = null;
    this.errors = 0;
    this.lastGoodT = null;
    this.forceGap = false;
    const video = this.video;
    const chest = this.makeChestSource(video);
    this.stopLoop = this.subscribe(video, (now, meta) => this.onFrame(video, chest, now, meta));
  }

  end(): void {
    this.ended = true;
    this.stopLoop?.();
    this.stopLoop = null;
    this.handle?.close();
    this.handle = null;
  }

  private onFrame(video: HTMLVideoElement, chest: ChestImageSource, now: number, meta?: FrameMeta): void {
    if (this.ended || this.paused || !this.handle || this.recreating || video.videoWidth === 0 || (video.readyState ?? 4) < 2) return;
    try {
      if (this.process(video, chest, now, meta) && this.errors > 0) { this.errors = 0; this.hooks.onError?.(false); }
    } catch {
      this.errors++;
      this.hooks.onError?.(true);
      this.forceGap = true; // whatever was lost is unobserved time for the detectors
      if (this.errors >= MAX_CONSECUTIVE_ERRORS) void this.recreate(chest);
    }
  }

  /** Re-create the landmarker after repeated failures; frames are ignored meanwhile and count as a gap afterwards. */
  private async recreate(chest: ChestImageSource): Promise<void> {
    if (this.recreating || !this.loader) return;
    this.recreating = true;
    const old = this.handle;
    this.handle = null;
    try { old?.close(); } catch { /* already broken */ }
    try {
      const h = await this.loader();
      if (this.ended) { h.close(); return; }
      this.handle = h;
      this.errors = 1; // the error message clears after the first good frame
      this.forceGap = true;
      this.breathing.reset(chest);
      this.lastEstimate = null;
    } catch {
      this.errors = MAX_CONSECUTIVE_ERRORS - 1; // try again after the next failing frame
      this.handle = old; // keep the (broken) handle so the loop keeps trying
    } finally {
      this.recreating = false;
    }
  }

  private process(video: HTMLVideoElement, chest: ChestImageSource, now: number, meta?: FrameMeta): boolean {
    const t = this.clock.time(now, meta);
    if (t === null || !this.scheduler.shouldRunPose(now)) return false; // pose did not run: neither success nor failure
    const t0 = this.nowMs();
    const lm = this.handle!.detect(video, now);
    let gap = this.scheduler.gapBefore(t);
    if (this.forceGap) {
      this.forceGap = false;
      if (this.lastGoodT !== null) gap = Math.max(gap, t - this.lastGoodT - 1 / params.scheduling.pose_hz, 1e-3);
    }
    this.lastGoodT = t;
    if (gap > 0) { this.breathing.reset(chest); this.lastEstimate = null; }
    const step = this.pipeline.step(t, lm, video.videoWidth / video.videoHeight, gap);
    if (step.aspectChanged) { this.breathing.reset(chest); this.lastEstimate = null; }
    const r = this.breathing.update(t, step.state, step.held, chest);
    if (this.breathing.status !== 'ok') this.lastEstimate = null;
    else if (r.estimate) { this.lastEstimate = r.estimate; this.lastEstimateT = t; }
    else if (r.due) this.lastEstimate = null; // the attempt failed: do not keep showing an old reading
    else if (this.lastEstimate && t - this.lastEstimateT > this.breathing.estimator.maxAgeS) this.lastEstimate = null;
    this.scheduler.reportDuration(this.nowMs() - t0);
    this.hooks.onView({ t, state: step.state, held: step.held, breathing: { status: this.breathing.status, estimate: this.lastEstimate } });
    for (const e of step.events) this.hooks.onEvent(e);
    return true;
  }
}
