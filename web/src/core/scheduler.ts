// Frame scheduling for the browser pose chain (DOM-free, so it is unit-tested): pose runs at ~pose_hz (8 Hz; the
// fall detector is validated down to 3 fps), never queues work (if a frame's inference exceeded the budget, the next
// due pose frame is skipped instead), and the face landmarker runs on every frame only while the face-HR card is
// active. Also turns long timestamp jumps (tab hidden, stalled camera) into the `gap` the detectors need.
// Thresholds come from shared/params.json -> scheduling. Not a medical device.

export interface SchedulingParams {
  pose_hz: number; pose_budget_ms: number; gap_min_s: number;
  inactivity_demo_default_s: number; inactivity_demo_options_s: number[];
}

export class PoseScheduler {
  private readonly periodMs: number;
  private due = -Infinity; // time (ms) at which the next pose frame is due
  private skipNext = false;
  private lastPoseT: number | null = null;
  skipped = 0;

  constructor(private readonly p: SchedulingParams) {
    this.periodMs = 1000 / p.pose_hz;
  }

  /** Call for every presented video frame with its time (ms). True = run pose on this frame. After a late frame the
   *  schedule restarts from `now` (no catch-up burst). */
  shouldRunPose(nowMs: number): boolean {
    if (nowMs < this.due - 1e-6) return false;
    if (this.due === -Infinity || nowMs - this.due > this.periodMs) this.due = nowMs + this.periodMs; // first frame / late: restart
    else this.due += this.periodMs;
    if (this.skipNext) {
      this.skipNext = false;
      this.skipped++;
      return false;
    }
    return true;
  }

  /** Report how long the last pose inference took; over budget -> skip the next pose frame rather than queue. */
  reportDuration(ms: number): void {
    if (ms > this.p.pose_budget_ms) this.skipNext = true;
  }

  /** Seconds the camera was unobserved before a pose frame at tSeconds (0 for a normal cadence): a timestamp jump
   *  longer than gap_min_s counts as an outage and the usual frame period is not part of it. */
  gapBefore(tSeconds: number): number {
    const last = this.lastPoseT;
    this.lastPoseT = tSeconds;
    if (last === null) return 0;
    const dt = tSeconds - last;
    return dt > this.p.gap_min_s ? dt - this.periodMs / 1000 : 0;
  }

  /** New stream / restart: forget timing. */
  reset(): void {
    this.due = -Infinity;
    this.skipNext = false;
    this.lastPoseT = null;
  }

  /** The face landmarker runs on every frame, but only while its card is active. */
  static shouldRunFace(active: boolean): boolean {
    return active;
  }
}
