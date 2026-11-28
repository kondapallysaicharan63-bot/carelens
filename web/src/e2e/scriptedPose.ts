// TEST ONLY (end-to-end build, vite --mode e2e). Replays a landmark fixture as if it were the pose model.
// The test page supplies the sequence through window.__E2E_POSE__ = { fps, landmarks }. Never part of the production build.
import type { PoseHandle } from '../capture/pose';
import type { Landmarks } from '../core/features';

export const E2E_MARKER = 'CARELENS_E2E_SCRIPTED_POSE';

/** `holdUntilPlay`: show the first frame until the test sets window.__E2E_PLAY__ (so a timed UI flow can start the motion at Go) */
interface Script { fps: number; landmarks: (Landmarks | null)[]; holdUntilPlay?: boolean }

export async function createScriptedPose(): Promise<PoseHandle> {
  const script = (window as unknown as { __E2E_POSE__?: Script }).__E2E_POSE__;
  if (!script) throw new Error(E2E_MARKER + ': window.__E2E_POSE__ is not set');
  let t0: number | null = null;
  return {
    delegate: E2E_MARKER,
    detect(_video, ts) {
      if (script.holdUntilPlay && !(window as unknown as { __E2E_PLAY__?: boolean }).__E2E_PLAY__) return script.landmarks[0] ?? null;
      t0 ??= ts;
      const i = Math.min(script.landmarks.length - 1, Math.floor(((ts - t0) / 1000) * script.fps));
      return script.landmarks[i] ?? null;
    },
    close() { /* nothing to release */ },
  };
}
