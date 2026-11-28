import { AudioClassifier, FilesetResolver } from '@mediapipe/tasks-audio';

const BASE = import.meta.env.BASE_URL;

export interface YamnetHandle {
  /** Scores of the requested labels for one 15600-sample (0.975 s) 16 kHz window; `tEnd` = end of the window in stream seconds (only the scripted test provider uses it). */
  classify(window: Float32Array, labels: readonly string[], tEnd?: number): Record<string, number>;
  close(): void;
}

/** YAMNet via MediaPipe AudioClassifier with the self-hosted wasm + model (same-origin, so the CSP is unchanged). */
export async function createYamnet(): Promise<YamnetHandle> {
  const fileset = await FilesetResolver.forAudioTasks(`${BASE}wasm/audio`);
  const clf = await AudioClassifier.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${BASE}models/yamnet.tflite` },
    maxResults: -1,
  });
  return {
    classify(window, labels) {
      const out: Record<string, number> = Object.fromEntries(labels.map((l) => [l, 0]));
      const res = clf.classify(window, 16000);
      for (const c of res[0]?.classifications[0]?.categories ?? []) if (c.categoryName in out) out[c.categoryName] = c.score;
      return out;
    },
    close: () => clf.close(),
  };
}
