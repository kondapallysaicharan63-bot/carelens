// Copies the MediaPipe wasm runtimes from node_modules into public/wasm so that they are
// self-hosted (nothing is fetched from a CDN at runtime). Run automatically by dev/build.
import { cpSync, mkdirSync, existsSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const pairs = [
  ['node_modules/@mediapipe/tasks-vision/wasm', 'public/wasm/vision'],
  ['node_modules/@mediapipe/tasks-audio/wasm', 'public/wasm/audio'],
];
for (const [from, to] of pairs) {
  const src = join(root, from);
  if (!existsSync(src)) {
    console.error(`missing ${from} - run "npm ci" first`);
    process.exit(1);
  }
  rmSync(join(root, to), { recursive: true, force: true });
  mkdirSync(join(root, to), { recursive: true });
  // FilesetResolver requests only <name>_wasm_internal.{js,wasm} (verified in a browser);
  // the *_module_* variants (ES-module loader) are never requested, so they are not shipped.
  cpSync(src, join(root, to), { recursive: true, filter: (f) => !f.includes('_module_') });
  console.log(`wasm: ${from} -> ${to}`);
}
