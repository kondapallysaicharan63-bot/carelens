// "Blinks & eyes-closed time" (Experimental, W9): blinks per minute, the share of the last minute the eyes were almost shut and
// long closures, from FaceLandmarker eye points. The face model is the one the face-heart-rate card uses (FaceHrSession, one
// session, one model); this card only switches the blink path on (default off) and shows the DOM-free core's numbers
// (src/core/blink.ts). Wellbeing trend only: no norms, no interpretation. Not a medical device.
import params from '../../../shared/params.json';
import { BlinkTracker, assertBlinkParams, type BlinkParams, type BlinkState, type FacePt } from '../core/blink';
import type { FaceFrame } from './faceHr';
import { blinkSummary } from '../export/wellbeing';
import type { MobilitySummaryFields } from '../export/mobility';
import { seedAndSubscribe } from './controls';
import { el } from './dom';

export const BLINK_PARAMS = assertBlinkParams(params.blink as unknown as BlinkParams);
export const BLINK_NOTE = 'Experimental. Screens, reading, dry air and light all change blink rate. Compare with your own earlier results.';
export const BLINK_FACE_HINT = 'Face the camera; glasses glare can affect this';

export type BlinkLoad = 'loading' | 'ready' | 'failed';

/** The guidance line is derived from (switch, face model state, last reason), never overwritten piecemeal. */
export function blinkGuidance(active: boolean, load: BlinkLoad, reason: BlinkState['reason'] | null, bp: BlinkParams = BLINK_PARAMS, failMsg = ''): string {
  if (!active) return 'Blink measurement is off. Switch it on to measure while you face the camera.';
  if (load === 'failed') return 'The face model failed to load' + (failMsg ? ': ' + failMsg : '');
  if (load === 'loading') return 'Loading the face model…';
  switch (reason) {
    case null:
    case 'no_face': return BLINK_FACE_HINT;
    case 'low_fps': return `This camera sends too few frames per second for this (about ${Math.round(bp.min_rate_hz)} or more are needed). No number is shown rather than a guess.`;
    case 'head_turned': return 'Face the camera: with your head turned the eye measurement is not reliable';
    case 'calibrating': return `Learning your open-eye level (about ${Math.round(bp.baseline_s)} s), then measuring. Look at the camera and blink as usual.`;
    case 'ok': return 'Measuring over the last minute. Keep facing the camera.';
  }
}

export interface BlinkCardDeps {
  bp?: BlinkParams;
  meter: (q: number | null) => HTMLElement;
  /** the session's summary row is refreshed (about once a second while measuring) */
  record(row: MobilitySummaryFields): void;
  /** the camera's declared frame rate (track.getSettings().frameRate), null when unknown */
  getDeclaredFps?: () => number | null;
}

export interface BlinkCard {
  root: HTMLElement;
  /** one distinct camera frame from the shared face session */
  onFrame(f: FaceFrame): void;
  /** registers the handler for the switch (default off); it is seeded with the current state */
  onActiveChange(cb: (active: boolean) => void): void;
  setLoadState(load: BlinkLoad, failMsg?: string): void;
  /** the stream was swapped: start the calibration and the window over */
  reset(): void;
}

const MIN_RENDER_GAP_S = 0.5;

export function buildBlinkCard(deps: BlinkCardDeps): BlinkCard {
  const bp = deps.bp ?? BLINK_PARAMS;
  let tracker = new BlinkTracker(bp);
  let load: BlinkLoad = 'loading';
  let failMsg = '';
  let lastReason: BlinkState['reason'] | null = null;
  let lastRenderT = -Infinity;
  let lastRecordT = -Infinity;
  let haveGood = false;

  const toggle = el('input', { type: 'checkbox', id: 'blink-active-toggle' });
  const value = el('p', { class: 'value muted', id: 'blink-value', 'aria-live': 'polite' }, 'No reading yet');
  const closed = el('p', { class: 'note', id: 'blink-closed' });
  const longC = el('p', { class: 'note', id: 'blink-long' });
  const meterSlot = el('div', { class: 'meter-slot' }, deps.meter(null));
  const guide = el('p', { class: 'guidance', id: 'blink-guidance', role: 'status' });
  const root = el('article', { class: 'card blink-card', id: 'blink-card', 'aria-labelledby': 'h-blink' },
    el('div', { class: 'card-head' }, el('h3', { id: 'h-blink' }, 'Blinks & eyes-closed time'), el('span', { class: 'tag exp' }, 'Experimental')),
    el('label', { class: 'toggle' }, toggle, el('span', {}, 'Measure blinks and eyes-closed time (uses the face model and more battery)')),
    value, closed, longC, meterSlot, guide,
    el('p', { class: 'note', id: 'blink-note' }, BLINK_NOTE),
    el('p', { class: 'note' }, 'Eyes-closed time is the share of the last minute your eyes were almost shut. Not a medical test.'));
  const subs: (() => void)[] = [];

  function show(s: BlinkState | null): void {
    const shown = s !== null && s.reported;
    lastReason = s ? s.reason : null;
    if (shown) {
      value.textContent = `${s!.blinks_per_min.toFixed(0)} blinks per minute`; // rounding happens here, not in the core
      value.classList.remove('muted');
      closed.textContent = `Eyes closed: ${(100 * s!.perclos).toFixed(0)} % of the last minute`;
      longC.textContent = `Long closures (over ${(bp.blink_max_ms / 1000).toFixed(1)} s): ${s!.long_closures}`;
      value.setAttribute('data-bpm', s!.blinks_per_min.toFixed(2));
      value.setAttribute('data-closed-pct', (100 * s!.perclos).toFixed(1));
      value.setAttribute('data-long', String(s!.long_closures));
      value.setAttribute('data-window', s!.window_s.toFixed(1));
    } else {
      value.textContent = 'No reading yet';
      value.classList.add('muted');
      closed.textContent = '';
      longC.textContent = '';
      for (const k of ['data-bpm', 'data-closed-pct', 'data-long', 'data-window']) value.removeAttribute(k);
    }
    value.setAttribute('data-reason', s ? s.reason : 'idle');
    meterSlot.replaceChildren(deps.meter(shown ? s!.coverage : null));
    guide.textContent = blinkGuidance(toggle.checked, load, lastReason, bp, failMsg);
  }

  function renderIdle(): void { show(null); }
  renderIdle();

  toggle.addEventListener('change', () => {
    tracker = new BlinkTracker(bp);
    lastRenderT = lastRecordT = -Infinity;
    lastReason = null;
    renderIdle();
    for (const h of subs) h();
  });

  return {
    root,
    onActiveChange: (cb) => seedAndSubscribe(() => toggle.checked, (h) => subs.push(h), cb),
    setLoadState(l, m = '') { load = l; failMsg = m; guide.textContent = blinkGuidance(toggle.checked, load, lastReason, bp, failMsg); },
    reset() { tracker = new BlinkTracker(bp); lastRenderT = lastRecordT = -Infinity; show(null); },
    onFrame(f) {
      if (!toggle.checked) return;
      let s = tracker.update(f.t, f.lm as ArrayLike<FacePt> | null, f.aspect);
      if (!f.reliable) {
        // Distinct camera frames could not be told apart from repeated display callbacks: trust the declared rate, withhold when unknown.
        const fps = deps.getDeclaredFps?.() ?? null;
        if (fps === null || !(fps >= bp.min_rate_hz)) s = { ...s, reason: 'low_fps', reported: false };
      }
      if (f.t - lastRenderT >= MIN_RENDER_GAP_S || s.reason !== lastReason) { lastRenderT = f.t; show(s); }
      if (f.t - lastRecordT >= 1) {
        lastRecordT = f.t;
        if (s.reported) { haveGood = true; deps.record(blinkSummary(s)); }
        else if (!haveGood) deps.record(blinkSummary(s)); // never overwrite a reported row with a later withheld one
      }
    },
  };
}
