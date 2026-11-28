import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { Pt } from './faceRoi';

const BASE = import.meta.env.BASE_URL;

export interface FaceHandle {
  /** 478 normalised landmarks of the first face, or null when no face is found. */
  detect(video: HTMLVideoElement, ts: number): Pt[] | null;
  delegate: string;
  close(): void;
}

/** FaceLandmarker with the self-hosted wasm + model (same FilesetResolver path as the pose model, so the CSP is unchanged). */
export async function createFaceLandmarker(): Promise<FaceHandle> {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm/vision`);
  const make = (delegate: 'GPU' | 'CPU') =>
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
    });
  let delegate = 'GPU';
  let landmarker: FaceLandmarker;
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
      return r.faceLandmarks.length > 0 ? (r.faceLandmarks[0] as Pt[]) : null;
    },
    close: () => landmarker.close(),
  };
}
