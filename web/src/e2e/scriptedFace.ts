// TEST ONLY (end-to-end build, vite --mode e2e). Replays a scripted eye-aspect-ratio trace as if it were the face model: the test
// page supplies window.__E2E_FACE__ = { fps, ear: number[] } (one EAR value per script frame, replayed in real time from the first
// detect call, held at the last value after the end); the 478 landmarks are a frontal synthetic face whose eyes have that EAR.
// Without the script the real model is used, so other tests are unaffected. Never part of the production build.
import type { FaceHandle } from '../capture/face';
import type { Pt } from '../capture/faceRoi';
import params from '../../../shared/params.json';

export const E2E_FACE_MARKER = 'CARELENS_E2E_SCRIPTED_FACE';

interface Script { fps: number; ear: number[] }

const EYE_W = 0.085;                                    // eye width in frame-height units
const CENTRES = { left: [-0.075, -0.02], right: [0.075, -0.02] } as const;

export async function createScriptedFace(fallback: () => Promise<FaceHandle>): Promise<FaceHandle> {
  const script = (window as unknown as { __E2E_FACE__?: Script }).__E2E_FACE__;
  if (!script) return fallback();
  let t0: number | null = null;
  return {
    delegate: E2E_FACE_MARKER,
    detect(video, ts) {
      t0 ??= ts;
      const i = Math.min(script.ear.length - 1, Math.floor(((ts - t0) / 1000) * script.fps));
      const ear = script.ear[i]!;
      const aspect = video.videoWidth / video.videoHeight;
      const cx = 0.5 * aspect, cy = 0.5;
      const lm: Pt[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
      const put = (k: number, x: number, y: number): void => { lm[k] = { x: x / aspect, y }; };
      for (const side of ['left', 'right'] as const) {
        const [ox, oy] = CENTRES[side];
        const ex = cx + ox, ey = cy + oy, h = ear * EYE_W;
        const pts = [[ex - EYE_W / 2, ey], [ex - EYE_W / 6, ey - h / 2], [ex + EYE_W / 6, ey - h / 2], [ex + EYE_W / 2, ey], [ex + EYE_W / 6, ey + h / 2], [ex - EYE_W / 6, ey + h / 2]];
        params.blink.eye_landmarks[side].forEach((k, j) => put(k, pts[j]![0]!, pts[j]![1]!));
      }
      put(params.blink.pose_landmarks.nose, cx, cy + 0.05);
      put(params.blink.pose_landmarks.forehead, cx, cy - 0.2);
      put(params.blink.pose_landmarks.chin, cx, cy + 0.25);
      return lm;
    },
    close() { /* nothing to release */ },
  };
}
