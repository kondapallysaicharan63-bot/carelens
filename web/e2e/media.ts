// Synthetic fixture media for the end-to-end tests, generated at test time into a temp directory (never committed).
// Everything here is synthetic signal: no recordings or images of real people.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export const TMP = process.env.CARELENS_E2E_TMP ?? join(tmpdir(), 'carelens-e2e');
mkdirSync(TMP, { recursive: true });

/** small seeded PRNG (mulberry32) + Box-Muller normal */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function normal(r: () => number): number {
  return Math.sqrt(-2 * Math.log(r() || 1e-12)) * Math.cos(2 * Math.PI * r());
}

const clamp8 = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));

/** Writes a Y4M (4:2:0) clip; `rgb(frame, rnd)` returns the (r, g, b) of every pixel of that frame via `px`. */
function writeY4m(path: string, w: number, h: number, fps: number, frames: number, px: (f: number, rnd: () => number) => [number, number, number]): void {
  const header = Buffer.from(`YUV4MPEG2 W${w} H${h} F${fps}:1 Ip A1:1 C420jpeg\n`);
  const fsz = w * h * 3 / 2;
  const out = Buffer.alloc(header.length + frames * (6 + fsz));
  header.copy(out);
  let o = header.length;
  const rnd = rng(1234);
  const R = new Float32Array(w * h), G = new Float32Array(w * h), B = new Float32Array(w * h);
  for (let f = 0; f < frames; f++) {
    out.write('FRAME\n', o); o += 6;
    for (let i = 0; i < w * h; i++) { const [r, g, b] = px(f, rnd); R[i] = r; G[i] = g; B[i] = b; }
    const yo = o, uo = o + w * h, vo = uo + (w * h) / 4;
    for (let i = 0; i < w * h; i++) out[yo + i] = clamp8(16 + (65.481 * R[i]! + 128.553 * G[i]! + 24.966 * B[i]!) / 255);
    for (let y = 0; y < h / 2; y++) {
      for (let x = 0; x < w / 2; x++) {
        let r = 0, g = 0, b = 0;
        for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) { const k = (2 * y + dy) * w + 2 * x + dx; r += R[k]!; g += G[k]!; b += B[k]!; }
        r /= 4; g /= 4; b /= 4;
        out[uo + y * (w / 2) + x] = clamp8(128 + (-37.797 * r - 74.203 * g + 112 * b) / 255);
        out[vo + y * (w / 2) + x] = clamp8(128 + (112 * r - 93.786 * g - 18.214 * b) / 255);
      }
    }
    o += fsz;
  }
  writeFileSync(path, out);
}

/**
 * Fingertip clip: near-uniform red frames whose R and G brightness pulse at `bpm` (a whole number of beats per loop, so the
 * loop is seamless), with per-pixel noise. R ~ 200 +- 4, G ~ 50 +- 2: finger-like (red dominant, not clipped).
 */
export function writeFingertipY4m(name = 'fingertip-72bpm.y4m', bpm = 72, seconds = 40, fps = 30, w = 160, h = 120): string {
  const path = join(TMP, name);
  const frames = Math.round(seconds * fps);
  writeY4m(path, w, h, fps, frames, (f, rnd) => {
    // (called per pixel; the beat phase depends on the frame only)
    const s = Math.sin(2 * Math.PI * (bpm / 60) * (f / fps));
    return [200 + 4 * s + 3 * normal(rnd), 50 + 2 * s + 2 * normal(rnd), 40 + 2 * normal(rnd)];
  });
  return path;
}

/**
 * Beat times (s) of a known RR series: mean 850 ms, respiratory sinus arrhythmia (0.25 Hz sinusoid, 70 % of the variance)
 * plus AR(1) noise, scaled so the TRUE RMSSD equals `rmssdMs`. Returns the beats and the true RMSSD / SDNN (ms).
 */
export function rrSeries(seconds: number, rmssdMs: number, seed = 7, meanMs = 850): { beats: number[]; rmssd: number; sdnn: number } {
  const r = rng(seed);
  const n = Math.floor((seconds * 1000) / meanMs) + 8;
  const ph = 2 * Math.PI * r();
  let e = 0;
  const es: number[] = [], rsa: number[] = [];
  for (let k = 0; k < n; k++) { e = k === 0 ? normal(r) : 0.6 * e + normal(r); es.push(e); rsa.push(Math.sin(2 * Math.PI * 0.25 * k * (meanMs / 1000) + ph)); }
  const sd = (v: number[]): number => { const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };
  const mu = (v: number[]): number => v.reduce((a, b) => a + b, 0) / v.length;
  const se = sd(es), sr = sd(rsa), me = mu(es), mr = mu(rsa);
  const dev = es.map((x, i) => Math.sqrt(0.7) * ((rsa[i]! - mr) / sr) + Math.sqrt(0.3) * ((x - me) / se));
  const md = mu(dev);
  const d0 = dev.map((x) => x - md);
  const rmssd0 = Math.sqrt(mu(d0.slice(1).map((x, i) => (x - d0[i]!) ** 2)));
  const rr = d0.map((x) => meanMs + (rmssdMs / rmssd0) * x);
  const beats = [-r() * 0.5];
  for (const x of rr) beats.push(beats[beats.length - 1]! + x / 1000);
  const sdnn = Math.sqrt(rr.reduce((a, b) => a + (b - mu(rr)) ** 2, 0) / (rr.length - 1));
  return { beats, rmssd: rmssdMs, sdnn };
}

/**
 * Fingertip clip whose pulse follows a KNOWN RR series (see rrSeries): each beat is a systolic peak plus a smaller dicrotic
 * bump (the same template as carelens.ppg), R ~ 200 with ~8 counts peak to peak, G ~ 50, per-pixel noise. `seconds` is long
 * enough that the clip does not loop during the test (a loop would put a phase jump in the beat series).
 */
export function writeFingertipRrY4m(name: string, rmssdMs: number, seconds = 100, fps = 30, w = 80, h = 60): { path: string; rmssd: number; sdnn: number } {
  const { beats, rmssd, sdnn } = rrSeries(seconds + 10, rmssdMs);
  const path = join(TMP, name);
  const pulse = (tau: number): number => Math.exp(-0.5 * ((tau - 0.18) / 0.09) ** 2) + 0.4 * Math.exp(-0.5 * ((tau - 0.45) / 0.12) ** 2);
  const wave = new Float64Array(Math.round(seconds * fps));
  for (let f = 0; f < wave.length; f++) {
    let v = 0;
    for (const b0 of beats) { const tau = f / fps - b0; if (tau >= 0 && tau < 1.2) v += pulse(tau); }
    wave[f] = v / 1.4 - 0.25;
  }
  writeY4m(path, w, h, fps, wave.length, (f, rnd) => [200 + 8 * wave[f]! + 3 * normal(rnd), 50 + 1.0 * wave[f]! + 2 * normal(rnd), 40 + 2 * normal(rnd)]);
  return { path, rmssd, sdnn };
}

/** Plain mid-grey 4:3 clip with faint noise (a camera that sees "something"); used when the pose model is scripted. */
export function writeGreyY4m(name = 'grey.y4m', seconds = 4, fps = 15, w = 160, h = 120): string {
  const path = join(TMP, name);
  writeY4m(path, w, h, fps, Math.round(seconds * fps), (_f, rnd) => [120 + 4 * normal(rnd), 120 + 4 * normal(rnd), 120 + 4 * normal(rnd)]);
  return path;
}

/**
 * Breath-like noise at `bpm`: band-limited noise times a gain with a quiet inhale bump and a louder, longer exhale bump,
 * over a faint noise floor. 16-bit mono PCM WAV (the length is a whole number of breaths so it loops cleanly).
 */
export function writeBreathWav(name = 'breath-15bpm.wav', bpm = 15, seconds = 60, fs = 48000): string {
  const path = join(TMP, name);
  const n = Math.round(seconds * fs), T = 60 / bpm, r = rng(99);
  const pcm = Buffer.alloc(44 + n * 2);
  const bump = (x: number): number => (x > 0 && x < 1 ? 0.5 - 0.5 * Math.cos(2 * Math.PI * x) : 0); // 0..1..0 over x in (0, 1)
  let lp1 = 0, lp2 = 0, hp = 0, prev = 0;
  const aLo = 1 - Math.exp(-2 * Math.PI * 1800 / fs), aHi = Math.exp(-2 * Math.PI * 250 / fs);
  for (let i = 0; i < n; i++) {
    const ph = ((i / fs) % T);
    const gain = 0.004 + 0.12 * bump((ph - 0.2) / 1.2) + 0.40 * bump((ph - 1.6) / 1.8); // inhale 0.2-1.4 s, exhale 1.6-3.4 s (4 s cycle at 15/min)
    const w = normal(r);
    lp1 += aLo * (w - lp1); lp2 += aLo * (lp1 - lp2);
    hp = aHi * (hp + lp2 - prev); prev = lp2;
    const v = Math.max(-0.95, Math.min(0.95, hp * 2.2 * gain * 3));
    pcm.writeInt16LE(Math.round(v * 32767), 44 + 2 * i);
  }
  pcm.write('RIFF', 0); pcm.writeUInt32LE(36 + n * 2, 4); pcm.write('WAVEfmt ', 8); pcm.writeUInt32LE(16, 16); pcm.writeUInt16LE(1, 20);
  pcm.writeUInt16LE(1, 22); pcm.writeUInt32LE(fs, 24); pcm.writeUInt32LE(fs * 2, 28); pcm.writeUInt16LE(2, 32); pcm.writeUInt16LE(16, 34);
  pcm.write('data', 36); pcm.writeUInt32LE(n * 2, 40);
  writeFileSync(path, pcm);
  return path;
}

/** A shared/fixtures landmark fixture expanded to 33 rows per frame (unlisted landmarks: invisible), as the pose provider returns them. */
export function fixtureLandmarks(name: string): { fps: number; landmarks: (number[][] | null)[] } {
  const d = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'shared', 'fixtures', `${name}.json`), 'utf8'));
  const idx: number[] = d.input.landmark_indices;
  const landmarks = (d.input.landmarks as (number[][] | null)[]).map((frame) => {
    if (!frame) return null;
    const lm: number[][] = Array.from({ length: 33 }, () => [0, 0, 0.2]);
    idx.forEach((k, j) => { lm[k] = frame[j]!; });
    return lm;
  });
  return { fps: d.input.fps, landmarks };
}

/** The fall_sudden landmark fixture (see fixtureLandmarks). */
export const fallLandmarks = (): { fps: number; landmarks: (number[][] | null)[] } => fixtureLandmarks('fall_sudden');

/** The synthetic 30 s chair-stand fixture (8 full reps + 2 partial rises) and its scripted rep count. */
export function chairStandScript(): { fps: number; landmarks: (number[][] | null)[]; holdUntilPlay: boolean; reps: number } {
  const d = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'shared', 'fixtures', 'mobility_chair_stand_8.json'), 'utf8'));
  return { ...fixtureLandmarks('mobility_chair_stand_8'), holdUntilPlay: true, reps: d.expected.reps as number };
}

const HAND_TEMPLATE: [number, number][] = [
  [0, 0], [-0.45, -0.25], [-0.70, -0.55], [-0.95, -0.80], [-1.15, -1.00], [-0.32, -0.98], [-0.38, -1.40], [-0.40, -1.70], [-0.42, -1.95],
  [0, -1.00], [0, -1.50], [0, -1.85], [0, -2.15], [0.30, -0.95], [0.34, -1.35], [0.36, -1.65], [0.38, -1.90], [0.58, -0.85], [0.65, -1.15], [0.70, -1.35], [0.73, -1.55],
];

/**
 * Synthetic held-out hand (21 landmarks per frame, x/y normalised) for the scripted hand model: `hz` rhythmic movement of `ampRmsPct`
 * (RMS, % of hand size) along a diagonal axis, MediaPipe-like landmark jitter (0.4 % of frame height) and a slow drift. `hz` 0 = a
 * steady hand. The movement starts at Go (holdUntilPlay) and the script is longer than the 15 s test.
 */
export function handScript(hz: number, ampRmsPct = 2.5, seconds = 25, fps = 30, aspect = 4 / 3): { fps: number; landmarks: number[][][]; holdUntilPlay: boolean } {
  const r = rng(4242 + Math.round(hz * 10));
  const size = 0.24, jitter = 0.004, ph = 2 * Math.PI * r(), drift = [0.15, 0.4, 0.8].map(() => 2 * Math.PI * r());
  const n = Math.round(seconds * fps);
  const landmarks: number[][][] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fps;
    const s = hz > 0 ? Math.SQRT2 * (ampRmsPct / 100) * Math.sin(2 * Math.PI * hz * t + ph) : 0;
    const dx = s * Math.cos(Math.PI / 6) + 0.015 * Math.sin(2 * Math.PI * 0.15 * t + drift[0]!) + 0.01 * Math.sin(2 * Math.PI * 0.4 * t + drift[1]!);
    const dy = s * Math.sin(Math.PI / 6) + 0.015 * Math.sin(2 * Math.PI * 0.4 * t + drift[2]!);
    landmarks.push(HAND_TEMPLATE.map(([tx, ty]) => [
      (0.5 * aspect + size * (tx + dx) + jitter * normal(r)) / aspect,
      0.75 + size * (ty + dy) + jitter * normal(r),
    ]));
  }
  return { fps, landmarks, holdUntilPlay: true };
}
