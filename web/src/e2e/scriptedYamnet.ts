// TEST ONLY (end-to-end build, vite --mode e2e). Replaces the YAMNet model with scripted scores so a test can raise a label on demand.
// The test page sets window.__E2E_YAMNET__ = { active: {} } and later assigns e.g. active = { 'Thump, thud': 0.9 } for a while.
// Without the global the app keeps the real model. Never part of the production build.
import type { YamnetHandle } from '../capture/yamnet';

export const E2E_MARKER = 'CARELENS_E2E_SCRIPTED_YAMNET';

interface Script { active: Record<string, number> }

/** The scripted provider when the test page supplied a script, else null (the real model is used). */
export function createScriptedYamnetIfSet(): YamnetHandle | null {
  const script = (window as unknown as { __E2E_YAMNET__?: Script }).__E2E_YAMNET__;
  if (!script) return null;
  return {
    classify(_window, labels) {
      return Object.fromEntries(labels.map((l) => [l, script.active[l] ?? 0]));
    },
    close() { /* nothing to release */ },
  };
}
