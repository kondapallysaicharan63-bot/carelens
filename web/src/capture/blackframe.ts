export interface ImageDataLike {
  data: ArrayLike<number>; // RGBA
  width: number;
  height: number;
}
export interface LumaStats {
  mean: number;
  variance: number;
}

export const BLACK_MEAN_MAX = 8;
export const BLACK_VARIANCE_MAX = 4;

export function lumaStats(img: ImageDataLike): LumaStats {
  const n = img.width * img.height;
  if (n === 0) return { mean: 0, variance: 0 };
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const y = 0.2126 * (img.data[o] ?? 0) + 0.7152 * (img.data[o + 1] ?? 0) + 0.0722 * (img.data[o + 2] ?? 0);
    sum += y;
    sumSq += y * y;
  }
  const mean = sum / n;
  return { mean, variance: Math.max(0, sumSq / n - mean * mean) };
}

export function isBlackStats(s: LumaStats): boolean {
  return s.mean < BLACK_MEAN_MAX && s.variance < BLACK_VARIANCE_MAX;
}

/** True only if there is at least one sample and every sample is black. */
export function isBlackSequence(frames: ImageDataLike[]): boolean {
  return frames.length > 0 && frames.every((f) => isBlackStats(lumaStats(f)));
}

/** Samples the video for ~durationMs and reports whether it is sending black frames. */
export async function detectBlackVideo(
  video: HTMLVideoElement,
  durationMs = 1000,
  samples = 5,
): Promise<boolean> {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 48;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return false;
  const frames: ImageData[] = [];
  for (let i = 0; i < samples; i++) {
    await new Promise((r) => setTimeout(r, durationMs / samples));
    if (video.videoWidth === 0) continue;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  }
  return isBlackSequence(frames);
}
