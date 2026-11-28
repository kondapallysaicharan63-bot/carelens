import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

const BASE = import.meta.env.BASE_URL;

/** Smoke path: proves self-hosted wasm + model load under the CSP. Presence only for now. */
export async function createPoseSmoke(): Promise<{ detect: (v: HTMLVideoElement, ts: number) => boolean; delegate: string; close: () => void }> {
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
      return landmarker.detectForVideo(video, ts).landmarks.length > 0;
    },
    close: () => landmarker.close(),
  };
}
