/** Subset of VideoFrameMetadata we use (absent when falling back to requestAnimationFrame). */
export interface FrameMeta {
  mediaTime?: number;
  expectedDisplayTime?: number;
}

/** Calls cb for each presented video frame (requestVideoFrameCallback, else rAF). Returns stop(). */
export function onVideoFrame(video: HTMLVideoElement, cb: (nowMs: number, meta?: FrameMeta) => void): () => void {
  let stopped = false;
  const hasRvfc = 'requestVideoFrameCallback' in video;
  const tick = (now: number, meta?: FrameMeta) => {
    if (stopped) return;
    cb(now, meta);
    schedule();
  };
  const schedule = () => {
    if (hasRvfc) video.requestVideoFrameCallback((now, meta) => tick(now, meta));
    else requestAnimationFrame((now) => tick(now));
  };
  schedule();
  return () => {
    stopped = true;
  };
}
