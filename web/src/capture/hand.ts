import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const BASE = import.meta.env.BASE_URL;

export interface HandDetection {
  /** 21 landmark rows [x, y], normalised by frame width / height. */
  lm: number[][];
  /** 1 for a hand the model returned: it already passed minHandDetectionConfidence / minHandPresenceConfidence (the handedness
   *  score only says left vs right, so it is NOT a presence measure and is not used). Scripted test providers may report less. */
  score: number;
}

export interface HandHandle {
  detect(video: HTMLVideoElement, ts: number): HandDetection | null;
  delegate?: string;
  close(): void;
}

/** HandLandmarker with the self-hosted wasm + model (same FilesetResolver path as the face and pose models, so the CSP is
 *  unchanged). Loaded only for the hand test and closed when it ends. */
export async function createHandLandmarker(): Promise<HandHandle> {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}wasm/vision`);
  const make = (delegate: 'GPU' | 'CPU') =>
    HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: `${BASE}models/hand_landmarker.task`, delegate },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  let delegate = 'GPU';
  let landmarker: HandLandmarker;
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
      if (!lm || lm.length < 21) return null;
      return { lm: lm.slice(0, 21).map((p) => [p.x, p.y]), score: 1 };
    },
    close: () => landmarker.close(),
  };
}
