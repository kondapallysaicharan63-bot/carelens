// Downloads the MediaPipe / YAMNet model files into public/models with pinned SHA-256 checks.
// Usage: npm run fetch-models   (no dependencies; Node >= 18)
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE = 'https://storage.googleapis.com/mediapipe-models';
const MODELS = [
  { file: 'face_landmarker.task', path: 'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    sha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff' },
  { file: 'pose_landmarker_lite.task', path: 'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    sha256: '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a' },
  { file: 'hand_landmarker.task', path: 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    sha256: 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1' },
  { file: 'yamnet.tflite', path: 'audio_classifier/yamnet/float32/1/yamnet.tflite',
    sha256: '4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf' },
];

const dir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');
mkdirSync(dir, { recursive: true });
const sha = (buf) => createHash('sha256').update(buf).digest('hex');

let failed = false;
for (const m of MODELS) {
  const dest = join(dir, m.file);
  if (existsSync(dest) && sha(readFileSync(dest)) === m.sha256) {
    console.log(`ok (cached)  ${m.file}`);
    continue;
  }
  const res = await fetch(`${BASE}/${m.path}`);
  if (!res.ok) {
    console.error(`FAILED ${m.file}: HTTP ${res.status}`);
    failed = true;
    continue;
  }
  const buf = Buffer.from(await res.arrayBuffer());
  const got = sha(buf);
  if (got !== m.sha256) {
    console.error(`CHECKSUM MISMATCH ${m.file}\n  expected ${m.sha256}\n  got      ${got}`);
    failed = true;
    continue;
  }
  writeFileSync(dest + '.part', buf);
  renameSync(dest + '.part', dest);
  console.log(`ok (fetched) ${m.file}`);
}
process.exit(failed ? 1 : 0);
