// "Heart rate (face)" (Tier 1): the frame loop that feeds the DOM-free rPPG core (src/core/rppg.ts) from
// FaceLandmarker landmarks, plus the card, guidance text and optional ROI outline. FaceLandmarker runs on every
// frame; the (smoke-test) pose presence loop is throttled to ~5 Hz so phones stay responsive (W3: keep that budget
// in mind when the real pose work lands). Wellbeing demonstration only. Not a medical device.
import params from '../../../shared/params.json';
import { FaceRPPG, assertRppgParams, type RppgParams, type RppgReason, type RppgUpdate } from '../core/rppg';
import { FrameClock } from '../capture/ppgSampler';
import type { FrameMeta } from '../capture/frames';
import { FaceRoiSampler, contentRect, eyeSpan, faceMotion, polysCentroid, roiPolygons, type Pt, type RoiPolys } from '../capture/faceRoi';
import { el } from './dom';

export const RPPG_PARAMS = assertRppgParams(params.rppg as RppgParams);

export const LIMITS_NOTE =
  'Camera heart-rate estimates are less reliable in low or uneven light, with movement, and can vary with skin tone.';
export const FINGERTIP_HINT = 'For a steadier reading, the fingertip mode is the stronger option.';

export function guidance(reason: RppgReason | 'idle' | 'loading'): string {
  switch (reason) {
    case 'loading':
      return 'Loading the face model…';
    case 'idle':
    case 'no_face':
      return 'Face the camera';
    case 'too_small':
      return 'Move closer';
    case 'motion':
      return 'Hold still';
    case 'skin_filter':
      return 'Use neutral, even light — very warm or coloured light hides the skin signal';
    case 'lighting':
      return 'Use steady, even light — avoid flicker and backlight';
    case 'settling':
      return 'Settling…';
    case 'low_signal':
      return 'Signal is weak. Hold still in steady, even light.';
    case 'disagree':
      return 'Hold still… the reading is not steady yet.';
    case 'gaps':
      return 'Hold still… the camera dropped some frames; keep this tab in the foreground.';
    case 'ok':
      return 'Good signal. Keep still.';
  }
}

export interface FaceHandle {
  detect(video: HTMLVideoElement, ts: number): Pt[] | null;
  close(): void;
}

export interface FaceHooks {
  onUpdate(u: RppgUpdate): void;
  /** every frame: the ROI polygons in normalised image coordinates, or null when no face */
  onOutline(polys: RoiPolys | null, videoW: number, videoH: number): void;
}

type Subscribe = (video: HTMLVideoElement, cb: (nowMs: number, meta?: FrameMeta) => void) => () => void;

/** Owns the face model, ROI sampler and core for one capture. Rebind with restart() after a stream swap; end() is safe at any time. */
export class FaceHrSession {
  private readonly core = new FaceRPPG(RPPG_PARAMS);
  private readonly sampler = new FaceRoiSampler(RPPG_PARAMS);
  private readonly clock = new FrameClock();
  private handle: FaceHandle | null = null;
  private stopLoop: (() => void) | null = null;
  private ended = false;
  private video: HTMLVideoElement | null = null;
  private prev: Pt | null = null;
  private prevT = 0;

  constructor(private readonly subscribe: Subscribe, private readonly hooks: FaceHooks) {}

  get isEnded(): boolean {
    return this.ended;
  }

  async load(loader: () => Promise<FaceHandle>, video: HTMLVideoElement): Promise<void> {
    this.video = video;
    const h = await loader();
    if (this.ended) { h.close(); return; }
    this.handle = h;
    this.restart();
  }

  restart(): void {
    this.stopLoop?.();
    this.stopLoop = null;
    if (this.ended || !this.handle || !this.video) return;
    this.core.reset(); this.clock.reset(); this.prev = null;
    const video = this.video;
    this.stopLoop = this.subscribe(video, (now, meta) => this.onFrame(video, now, meta));
  }

  end(): void {
    this.ended = true;
    this.stopLoop?.();
    this.stopLoop = null;
    this.handle?.close();
    this.handle = null;
  }

  private onFrame(video: HTMLVideoElement, now: number, meta?: FrameMeta): void {
    if (this.ended || !this.handle || video.videoWidth === 0 || video.readyState < 2) return;
    const t = this.clock.time(now, meta);
    if (t === null) return;
    const lm = this.handle.detect(video, now);
    let u: RppgUpdate | null;
    if (!lm) {
      this.prev = null;
      this.hooks.onOutline(null, video.videoWidth, video.videoHeight);
      u = this.core.update(t, null, 0, 0, 0);
    } else {
      const polys = roiPolygons(lm, RPPG_PARAMS.roi_landmarks);
      this.hooks.onOutline(polys, video.videoWidth, video.videoHeight);
      const aspect = video.videoWidth / video.videoHeight;
      const c = polysCentroid(polys);
      const motion = faceMotion(this.prev, c, t - this.prevT, eyeSpan(lm, aspect), aspect);
      this.prev = c; this.prevT = t;
      const s = this.sampler.sample(video, polys);
      if (!s) return; // empty mask (polygons fully outside the frame)
      u = this.core.update(t, s.rgb, s.px, motion, s.brightness, s.skinFrac);
    }
    if (u) this.hooks.onUpdate(u);
  }
}

/** Draws the ROI polygons on an overlay canvas that covers the video element; polygons are mapped into the
 *  displayed video content rectangle (the video is letterboxed by object-fit: contain). No-op while hidden. */
export function drawOutline(canvas: HTMLCanvasElement, polys: RoiPolys | null, videoW: number, videoH: number): void {
  if (canvas.hidden) return;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w === 0 || h === 0) return;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  if (!polys) return;
  const r = contentRect(w, h, videoW, videoH);
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#ffffff';
  ctx.shadowColor = '#000000';
  ctx.shadowBlur = 3;
  for (const poly of [polys.forehead, polys.left_cheek, polys.right_cheek]) {
    ctx.beginPath();
    poly.forEach((p, i) => (i === 0 ? ctx.moveTo(r.x + p.x * r.w, r.y + p.y * r.h) : ctx.lineTo(r.x + p.x * r.w, r.y + p.y * r.h)));
    ctx.closePath();
    ctx.stroke();
  }
}

export interface FaceCard {
  root: HTMLElement;
  /** overlay canvas: append it inside the (position: relative) video preview */
  overlay: HTMLCanvasElement;
  hooks: FaceHooks;
  setStatus(text: string): void;
}

export function buildFaceCard(meter: (q: number | null) => HTMLElement): FaceCard {
  const bpm = el('p', { class: 'value muted', id: 'face-bpm' }, 'No reading yet');
  const meterSlot = el('div', { class: 'meter-slot' }, meter(null));
  const guide = el('p', { class: 'guidance', id: 'face-guidance', role: 'status' }, guidance('loading'));
  const toggle = el('input', { type: 'checkbox', id: 'face-outline-toggle' });
  const overlay = document.createElement('canvas');
  overlay.className = 'roi-overlay';
  overlay.setAttribute('aria-hidden', 'true');
  overlay.hidden = true;
  toggle.addEventListener('change', () => { overlay.hidden = !toggle.checked; });
  const root = el(
    'article',
    { class: 'card face-card', 'aria-labelledby': 'h-face' },
    el('div', { class: 'card-head' }, el('h3', { id: 'h-face' }, 'Heart rate (face)'), el('span', { class: 'tag' }, 'Tier 1')),
    bpm,
    meterSlot,
    guide,
    el('label', { class: 'toggle' }, toggle, el('span', {}, 'Show the measured face areas')),
    el('p', { class: 'note', id: 'face-limits' }, LIMITS_NOTE),
    el('p', { class: 'note' }, FINGERTIP_HINT),
  );
  let lastReason: RppgReason | null = null;
  const hooks: FaceHooks = {
    onOutline: (polys, vw, vh) => drawOutline(overlay, polys, vw, vh),
    onUpdate(u) {
      lastReason = u.reason;
      if (u.reported && u.bpm !== null) {
        bpm.textContent = `${u.bpm.toFixed(0)} bpm`; // rounding happens here, not in the core
        bpm.classList.remove('muted');
      } else {
        bpm.textContent = 'No reading yet';
        bpm.classList.add('muted');
      }
      meterSlot.replaceChildren(meter(u.snr !== null ? u.quality : null));
      guide.textContent = guidance(u.reason);
      root.dataset.reason = lastReason;
    },
  };
  return { root, overlay, hooks, setStatus: (text) => { guide.textContent = text; } };
}
