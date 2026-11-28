// "Heart rate (fingertip)" mode: guidance text, 30 s measurement, live waveform, and the frame loop that
// feeds the DOM-free core (src/core/ppg.ts). The core runs on the main thread: one ~2k-point FFT per
// second is far below frame time, so a Web Worker would add complexity without benefit.
// Wellbeing demonstration only. Not a medical device.
import params from '../../../shared/params.json';
import { ContactPPG, assertContactPpgParams, type ContactPpgParams, type PpgReason, type PpgUpdate } from '../core/ppg';
import { FrameClock, RoiSampler } from '../capture/ppgSampler';
import type { FrameMeta } from '../capture/frames';
import { el } from './dom';

export const PPG_PARAMS = assertContactPpgParams(params.contact_ppg as ContactPpgParams);
export const MEASURE_S = 30;
export const MIN_REPORTED_FOR_RESULT = 5;

export const INSTRUCTIONS =
  'Cover the back camera and flash with your fingertip; rest your hand; hold still for 30 s.';
export const LAPTOP_NOTE =
  'No flash available on this camera. Cover the webcam with your fingertip near a bright light — weaker signal.';

export function guidance(reason: PpgReason | 'idle', settleLeftS = 0): string {
  switch (reason) {
    case 'idle':
    case 'no_finger':
      return 'Cover the camera and flash fully with your fingertip.';
    case 'saturated':
      return 'Press more lightly.';
    case 'settling':
      return `Hold still… ${Math.max(0, Math.ceil(settleLeftS))} s`;
    case 'low_signal':
      return 'Signal is weak. Rest your hand, press lightly and hold still.';
    case 'disagree':
      return 'Hold still… the reading is not steady yet.';
    case 'gaps':
      return 'Hold still… the camera dropped some frames; keep this tab in the foreground.';
    case 'ok':
      return 'Good signal. Keep holding still.';
  }
}

export interface MeasurementState {
  elapsedS: number;
  windows: number;
  reported: number;
  done: boolean;
  /** median of reported bpm; null if fewer than MIN_REPORTED_FOR_RESULT reported windows */
  medianBpm: number | null;
}

/** 30 s measurement: starts at the first update with a finger on the camera; restarts if the finger is lost. */
export class MeasurementTracker {
  private t0: number | null = null;
  private bpms: number[] = [];
  private windows = 0;
  private result: MeasurementState | null = null;
  private epoch: number | null = null;

  push(u: Pick<PpgUpdate, 't' | 'reason' | 'reported' | 'bpm' | 'contact_epoch'>): MeasurementState {
    if (this.result) return this.result;
    if (this.epoch !== null && u.contact_epoch !== this.epoch) { this.t0 = null; this.bpms = []; this.windows = 0; } // contact restarted
    this.epoch = u.contact_epoch;
    if (u.reason === 'no_finger' || u.reason === 'saturated') {
      this.t0 = null; this.bpms = []; this.windows = 0;
      return { elapsedS: 0, windows: 0, reported: 0, done: false, medianBpm: null };
    }
    if (this.t0 === null) this.t0 = u.t;
    this.windows++;
    if (u.reported && u.bpm !== null) this.bpms.push(u.bpm);
    const elapsedS = Math.min(MEASURE_S, u.t - this.t0);
    const done = u.t - this.t0 >= MEASURE_S;
    const s = [...this.bpms].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    const med = s.length < MIN_REPORTED_FOR_RESULT ? null : s.length % 2 ? s[m]! : 0.5 * (s[m - 1]! + s[m]!);
    const st = { elapsedS, windows: this.windows, reported: s.length, done, medianBpm: med === null ? null : Math.round(med * 10) / 10 };
    if (done) this.result = st;
    return st;
  }

  reset(): void {
    this.t0 = null; this.bpms = []; this.windows = 0; this.result = null; this.epoch = null;
  }
}

/** SVG polyline points for the last samples: mean removed, scaled to +-1 (with a floor so noise is not blown up). */
export function sparklinePoints(values: number[], w: number, h: number, minScale = 0.5): string {
  if (values.length < 2) return '';
  let m = 0;
  for (const v of values) m += v;
  m /= values.length;
  let mx = minScale;
  for (const v of values) mx = Math.max(mx, Math.abs(v - m));
  return values
    .map((v, i) => `${((i / (values.length - 1)) * w).toFixed(1)},${(h / 2 - ((v - m) / mx) * (h / 2 - 2)).toFixed(1)}`)
    .join(' ');
}

const SVG_NS = 'http://www.w3.org/2000/svg';

export interface FingertipHooks {
  onUpdate(u: PpgUpdate, m: MeasurementState, settleLeftS: number): void;
  onWave(points: string): void;
}

/** Owns the sampler, clock, core and measurement for one capture. end() is safe at any time. */
export class FingertipSession {
  private readonly core = new ContactPPG(PPG_PARAMS);
  private readonly sampler = new RoiSampler();
  private readonly clock = new FrameClock();
  private readonly meas = new MeasurementTracker();
  private stop: (() => void) | null = null;
  private wave: number[] = [];
  private lastDraw = 0;

  constructor(
    private readonly subscribe: (video: HTMLVideoElement, cb: (nowMs: number, meta?: FrameMeta) => void) => () => void,
    private readonly hooks: FingertipHooks,
    private readonly waveSamples = 150, // ~5 s at 30 fps
  ) {}

  start(video: HTMLVideoElement): void {
    this.stop?.();
    this.clock.reset();
    this.stop = this.subscribe(video, (now, meta) => this.onFrame(video, now, meta));
  }

  /** Re-arm after a stream swap (clock and measurement start fresh). */
  restart(video: HTMLVideoElement): void {
    this.core.reset(); this.meas.reset(); this.wave = [];
    this.start(video);
  }

  end(): void {
    this.stop?.();
    this.stop = null;
  }

  private onFrame(video: HTMLVideoElement, now: number, meta?: FrameMeta): void {
    const t = this.clock.time(now, meta);
    if (t === null) return;
    const s = this.sampler.sample(video);
    if (!s) return;
    this.wave.push(s.r);
    if (this.wave.length > this.waveSamples) this.wave.shift();
    if (now - this.lastDraw > 90) {
      this.lastDraw = now;
      this.hooks.onWave(sparklinePoints(this.wave, 300, 60));
    }
    const u = this.core.update(t, s.r, s.g, s.b, s.clipFrac);
    if (!u) return;
    const m = this.meas.push(u);
    this.hooks.onUpdate(u, m, PPG_PARAMS.settle_s - this.core.contactSeconds(t));
  }
}

export interface FingertipView {
  root: HTMLElement;
  video: HTMLVideoElement;
  status: HTMLElement;
  setNoTorchNote(show: boolean): void;
  hooks: FingertipHooks;
}

/** Builds the live panel: instruction, preview, waveform, BPM (only when reported), quality, guidance, 30 s result. */
export function buildFingertipView(
  meter: (q: number | null) => HTMLElement,
): FingertipView {
  const video = el('video', { playsinline: '', muted: '', autoplay: '', 'aria-label': 'Camera preview (your fingertip)' });
  video.muted = true;
  video.playsInline = true;
  const status = el('p', { class: 'status', role: 'status' }, 'Opening the camera…');
  const noTorch = el('p', { class: 'note', id: 'fp-laptop-note' }, LAPTOP_NOTE);
  noTorch.hidden = true;
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 300 60');
  svg.setAttribute('class', 'wave');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Live pulse waveform');
  const poly = document.createElementNS(SVG_NS, 'polyline');
  poly.setAttribute('fill', 'none');
  poly.setAttribute('stroke', 'currentColor');
  poly.setAttribute('stroke-width', '2');
  poly.setAttribute('vector-effect', 'non-scaling-stroke');
  svg.append(poly);
  const bpm = el('p', { class: 'value muted', id: 'fp-bpm' }, 'No reading yet');
  const meterSlot = el('div', { class: 'meter-slot' }, meter(null));
  const guide = el('p', { class: 'guidance', id: 'fp-guidance', role: 'status' }, guidance('idle'));
  const prog = el('progress', { max: String(MEASURE_S), value: '0', 'aria-label': 'Measurement progress' });
  const progText = el('p', { class: 'note', id: 'fp-progress' }, `Measurement: 0 / ${MEASURE_S} s`);
  const result = el('p', { class: 'value', id: 'fp-result' });
  result.hidden = true;
  const root = el(
    'section',
    { class: 'panel live fingertip', 'aria-labelledby': 'h-fp' },
    el('div', { class: 'card-head' }, el('h2', { id: 'h-fp' }, 'Heart rate (fingertip)'), el('span', { class: 'tag' }, 'Tier 1')),
    el('p', { class: 'instructions', id: 'fp-instructions' }, INSTRUCTIONS),
    noTorch,
    status,
    el('div', { class: 'preview small' }, video),
    svg,
    bpm,
    meterSlot,
    guide,
    prog,
    progText,
    result,
    el('p', { class: 'note' }, 'A wellbeing trend from the camera, not a medical measurement. Nothing leaves this device.'),
  );
  const hooks: FingertipHooks = {
    onWave: (points) => poly.setAttribute('points', points),
    onUpdate(u, m, settleLeft) {
      if (u.reported && u.bpm !== null) {
        bpm.textContent = `${u.bpm.toFixed(0)} bpm`; // rounding happens here, not in the core
        bpm.classList.remove('muted');
      } else {
        bpm.textContent = 'No reading yet';
        bpm.classList.add('muted');
      }
      meterSlot.replaceChildren(meter(u.snr !== null ? u.quality : null));
      guide.textContent = guidance(u.reason, settleLeft);
      prog.value = m.elapsedS;
      progText.textContent = `Measurement: ${Math.floor(m.elapsedS)} / ${MEASURE_S} s (${m.reported} readings)`;
      if (m.done) {
        result.hidden = false;
        result.textContent =
          m.medianBpm !== null
            ? `Result: about ${m.medianBpm.toFixed(0)} bpm (median of ${m.reported} readings in ${m.windows} windows). A wellbeing trend, not a diagnosis.`
            : `Not enough clean signal for a result (${m.reported} of ${m.windows} windows). Press Stop and try again, resting your hand.`;
      }
    },
  };
  return { root, video, status, setNoTorchNote: (show) => { noTorch.hidden = !show; }, hooks };
}
