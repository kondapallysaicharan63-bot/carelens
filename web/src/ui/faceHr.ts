// "Heart rate (face)" (Tier 1): the frame loop that feeds the DOM-free rPPG core (src/core/rppg.ts) from
// FaceLandmarker landmarks, plus the card, guidance text and optional ROI outline. FaceLandmarker runs on every
// frame; the (smoke-test) pose presence loop is throttled to ~5 Hz so phones stay responsive (W3: keep that budget
// in mind when the real pose work lands). Wellbeing demonstration only. Not a medical device.
import params from '../../../shared/params.json';
import { FaceRPPG, assertRppgParams, type RppgParams, type RppgReason, type RppgUpdate } from '../core/rppg';
import { FrameClock } from '../capture/ppgSampler';
import type { FrameMeta } from '../capture/frames';
import { FaceRoiSampler, contentRect, eyeSpan, faceMotion, polysCentroid, roiPolygons, type Pt, type RoiPolys } from '../capture/faceRoi';
import { PoseScheduler } from '../core/scheduler';
import { seedAndSubscribe } from './controls';
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

export type FaceLoadState = 'loading' | 'ready' | 'failed';

/** The guidance line is derived from (load state, active, last reason), never overwritten piecemeal. */
export function faceGuidanceText(load: FaceLoadState, active: boolean, reason: RppgReason | null, failMsg = ''): string {
  if (load === 'failed') return 'Face model failed to load' + (failMsg ? ': ' + failMsg : '');
  if (!active) return 'Face heart rate is off';
  if (load === 'loading') return guidance('loading');
  return guidance(reason ?? 'idle');
}

export interface FaceHandle {
  detect(video: HTMLVideoElement, ts: number): Pt[] | null;
  close(): void;
}

export interface FaceHooks {
  onUpdate(u: RppgUpdate): void;
  /** every frame: the ROI polygons in normalised image coordinates, or null when no face */
  onOutline(polys: RoiPolys | null, videoW: number, videoH: number): void;
  /** frame processing threw (true) / recovered (false) */
  onError?(failing: boolean): void;
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
  private active = true;

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

  /** The face landmarker only runs while the face-HR card is active (PoseScheduler.shouldRunFace); pausing drops the window. */
  setActive(active: boolean): void {
    if (this.active === active) return;
    this.active = active;
    if (!active) {
      this.core.reset(); this.clock.reset(); this.prev = null;
      this.hooks.onOutline(null, 0, 0);
    }
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

  private errors = 0;

  private onFrame(video: HTMLVideoElement, now: number, meta?: FrameMeta): void {
    try {
      this.processFrame(video, now, meta);
      if (this.errors > 0) { this.errors = 0; this.hooks.onError?.(false); }
    } catch {
      this.errors++;
      this.core.reset(); this.clock.reset(); this.prev = null; // the frames lost are not data
      this.hooks.onError?.(true);
    }
  }

  private processFrame(video: HTMLVideoElement, now: number, meta?: FrameMeta): void {
    if (this.ended || !this.handle || !PoseScheduler.shouldRunFace(this.active) || video.videoWidth === 0 || video.readyState < 2) return;
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
  /** register the handler for the "Measure face heart rate" switch (default on); it is seeded with the current state */
  onActiveChange(cb: (active: boolean) => void): void;
  /** overlay canvas: append it inside the (position: relative) video preview */
  overlay: HTMLCanvasElement;
  hooks: FaceHooks;
  setLoadState(state: FaceLoadState, failMsg?: string): void;
  /** a processing error is being retried */
  setError(failing: boolean): void;
}

export function buildFaceCard(meter: (q: number | null) => HTMLElement): FaceCard {
  const bpm = el('p', { class: 'value muted', id: 'face-bpm' }, 'No reading yet');
  const meterSlot = el('div', { class: 'meter-slot' }, meter(null));
  const guide = el('p', { class: 'guidance', id: 'face-guidance', role: 'status' }, guidance('loading'));
  const toggle = el('input', { type: 'checkbox', id: 'face-outline-toggle' });
  const activeToggle = el('input', { type: 'checkbox', id: 'face-active-toggle' });
  activeToggle.checked = true;
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
    el('label', { class: 'toggle' }, activeToggle, el('span', {}, 'Measure face heart rate (uses more battery)')),
    el('label', { class: 'toggle' }, toggle, el('span', {}, 'Show the measured face areas')),
    el('p', { class: 'note', id: 'face-limits' }, LIMITS_NOTE),
    el('p', { class: 'note' }, FINGERTIP_HINT),
  );
  let lastReason: RppgReason | null = null;
  let load: FaceLoadState = 'loading';
  let failMsg = '';
  let failing = false;
  const renderGuide = () => {
    guide.textContent = failing ? 'Camera processing error — restarting detector' : faceGuidanceText(load, activeToggle.checked, lastReason, failMsg);
  };
  const hooks: FaceHooks = {
    onOutline: (polys, vw, vh) => drawOutline(overlay, polys, vw, vh),
    onError: (f) => { failing = f; renderGuide(); },
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
      root.dataset.reason = u.reason;
      renderGuide();
    },
  };
  const activeSubs: (() => void)[] = [];
  activeToggle.addEventListener('change', () => {
    if (!activeToggle.checked) { bpm.textContent = 'No reading yet'; bpm.classList.add('muted'); meterSlot.replaceChildren(meter(null)); }
    else lastReason = null;
    renderGuide();
    for (const h of activeSubs) h();
  });
  return {
    root, overlay, hooks,
    setLoadState: (s, m = '') => { load = s; failMsg = m; renderGuide(); },
    setError: (f) => { failing = f; renderGuide(); },
    onActiveChange: (cb) => seedAndSubscribe(() => activeToggle.checked, (h) => activeSubs.push(h), cb),
  };
}
