// Camera-breathing rows: one per estimator step. A new estimate is recorded with its value; while the estimator withholds
// one (nobody in view, moving, chest not visible, still collecting) a row with value null and the core's status as the
// reason is recorded once per step, so coverage counts that time like the other metrics do.
import type { PoseView } from '../capture/poseSession';
import type { SessionRecorder } from './recorder';

export class BreathCameraLogger {
  private lastEstT = -1;
  private lastRowS = -Infinity;

  constructor(private readonly rec: SessionRecorder, private readonly stepS: number) {}

  onView(v: PoseView): void {
    const est = v.breathing.estimate;
    const status = v.breathing.status;
    const now = this.rec.now();
    if (est && est.t !== this.lastEstT) {
      this.lastEstT = est.t;
      this.lastRowS = now;
      const reason = est.reported ? 'ok' : status !== 'ok' ? status : 'low_signal';
      this.rec.addEstimate({ metric: 'br_camera', value: est.value, quality: est.quality, reported: est.reported, reason, span_s: est.span_s });
    } else if (!est && now - this.lastRowS >= this.stepS) {
      this.lastRowS = now;
      this.rec.addEstimate({ metric: 'br_camera', value: null, quality: 0, reported: false, reason: status === 'ok' ? 'settling' : status });
    }
  }
}
