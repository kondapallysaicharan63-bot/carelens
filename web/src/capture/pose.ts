import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import type { Landmarks } from '../core/features';

const BASE = import.meta.env.BASE_URL;

export interface PoseHandle {
  /** 33 landmark rows [x, y, visibility] (x, y normalised by frame width / height) of the first person, or null. */
  detect(video: HTMLVideoElement, ts: number): Landmarks | null;
  delegate?: string;
  close(): void;
}

/** PoseLandmarker (lite) with the self-hosted wasm + model (same FilesetResolver path as the face model, so the CSP is unchanged). */
export async function createPoseLandmarker(): Promise<PoseHandle> {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm/vision`);
  const make = (delegate: 'GPU' | 'CPU') =>
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${BASE}models/pose_landmarker_lite.task`, delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
  let delegate = 'GPU';
  let landmarker: PoseLandmarker;
  try {
    landmarker = await make('GPU');
  } catch {
    delegate = 'CPU';
    landmarker = await make('CPU');
  }
  let last = -1;
  return {
    delegate,
    detect(video, ts) {
      if (ts <= last) ts = last + 1; // MediaPipe needs strictly increasing timestamps
      last = ts;
      const r = landmarker.detectForVideo(video, ts);
      const lm = r.landmarks[0];
      return lm ? lm.map((p) => [p.x, p.y, p.visibility ?? 1]) : null;
    },
    close: () => landmarker.close(),
  };
}
