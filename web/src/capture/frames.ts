/** Calls cb for each presented video frame (requestVideoFrameCallback, else rAF). Returns stop(). */
export function onVideoFrame(video: HTMLVideoElement, cb: (nowMs: number) => void): () => void {
  let stopped = false;
  const hasRvfc = 'requestVideoFrameCallback' in video;
  const tick = (now: number) => {
    if (stopped) return;
    cb(now);
    schedule();
  };
  const schedule = () => {
    if (hasRvfc) video.requestVideoFrameCallback((now) => tick(now));
    else requestAnimationFrame(tick);
  };
  schedule();
  return () => {
    stopped = true;
  };
}
