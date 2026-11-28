// TEST ONLY (end-to-end build, vite --mode e2e). Replays hand landmarks as if it were the hand model. The test page supplies the
// sequence through window.__E2E_HAND__ = { fps, landmarks, holdUntilPlay }; positions are interpolated in time between the scripted
// frames, so the replay is a continuous movement sampled at the real frame times. Never part of the production build.
import type { HandHandle } from '../capture/hand';

export const E2E_HAND_MARKER = 'CARELENS_E2E_SCRIPTED_HAND';

/** `holdUntilPlay`: show the first frame until the test sets window.__E2E_PLAY__ (the movement then starts at Go) */
interface Script { fps: number; landmarks: (number[][] | null)[]; holdUntilPlay?: boolean }

export async function createScriptedHand(): Promise<HandHandle> {
  const script = (window as unknown as { __E2E_HAND__?: Script }).__E2E_HAND__;
  if (!script) throw new Error(E2E_HAND_MARKER + ': window.__E2E_HAND__ is not set');
  let t0: number | null = null;
  return {
    delegate: E2E_HAND_MARKER,
    detect(_video, ts) {
      const first = script.landmarks[0];
      if (!first) return null;
      if (script.holdUntilPlay && !(window as unknown as { __E2E_PLAY__?: boolean }).__E2E_PLAY__) return { lm: first, score: 0.97 };
      t0 ??= ts;
      const pos = ((ts - t0) / 1000) * script.fps;
      const i = Math.min(script.landmarks.length - 1, Math.floor(pos));
      const a = script.landmarks[i], b = script.landmarks[Math.min(script.landmarks.length - 1, i + 1)];
      if (!a || !b) return null;
      const f = Math.min(1, Math.max(0, pos - i));
      return { lm: a.map((p, k) => [p[0]! + (b[k]![0]! - p[0]!) * f, p[1]! + (b[k]![1]! - p[1]!) * f]), score: 0.97 };
    },
    close() { /* nothing to release */ },
  };
}
