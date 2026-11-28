// Tier 1 audio cards: "Breathing (sound)" (+ live input level) and "Cough sounds". The DOM-free core does the work
// (src/core/audio.ts); this file only shows it. Wellbeing wording only; no diagnosis. Not a medical device.
import type { BreathSoundReason, BreathSoundUpdate, LabelEvent } from '../core/audio';
import type { AudioHooks } from '../capture/audioSession';
import { el } from './dom';
import { formatClock } from './poseCards';

export const BREATH_SOUND_HINT = 'Hold the phone 10–20 cm from your nose and mouth, breathe normally, don’t talk.';
export const BREATH_SOUND_NOTE = 'Breathing from sound is a wellbeing trend only. It needs a quiet room and a close microphone.';
export const COUGH_NOTE = 'Counts sounds the model recognises as coughs — wellbeing trend only.';
export const COUGH_BOUT_NOTE = 'A run of coughs less than about a second apart counts once.';

export function breathSoundGuidance(reason: BreathSoundReason | 'idle' | 'off' | 'loading'): string {
  switch (reason) {
    case 'off': return 'Microphone is off. Turn it on at the start screen to use this.';
    case 'idle':
    case 'loading': return BREATH_SOUND_HINT;
    case 'no_mic': return 'No microphone signal — check that the microphone is connected and not muted.';
    case 'too_quiet': return 'Too quiet — move the microphone closer to your nose and mouth.';
    case 'clipping': return 'Too loud or distorted — hold the microphone a little further away.';
    case 'talking': return 'Talking detected — stay quiet and breathe normally.';
    case 'settling': return 'Collecting a steady signal (about 20 seconds)…';
    case 'ambiguous': return 'Breathing sounds are ambiguous — breathe a little more audibly on the out-breath, or use the camera breathing card.';
    case 'low_signal': return `Signal is weak. ${BREATH_SOUND_HINT}`;
    case 'ok': return 'Good signal. Keep breathing normally.';
  }
}

export function coughLabel(count: number): string {
  return `Cough sounds this session: ${count}`;
}

/** DOM-free state of the two cards (unit-tested): what to show for the breathing and cough cards. */
export class AudioCardState {
  dead = false;
  lastBreath: BreathSoundUpdate | null = null;
  count = 0;
  lastWallMs: number | null = null;
  ticks: number[] = [];
  t = 0;

  /** a new session starts: nothing from the previous one may linger */
  reset(): void {
    this.dead = false; this.lastBreath = null; this.count = 0; this.lastWallMs = null; this.ticks = []; this.t = 0;
  }
  setBreath(u: BreathSoundUpdate): void { this.lastBreath = u; }
  setNoMic(dead: boolean): void { this.dead = dead; }
  setCough(r: { count: number; last: LabelEvent | null; ticks: number[]; t: number }, nowMs: number): void {
    if (r.count > this.count && r.last) this.lastWallMs = nowMs - (r.t - r.last.t_start) * 1000;
    this.count = r.count; this.ticks = r.ticks; this.t = r.t;
  }
  get breath(): { text: string | null; quality: number | null; reason: BreathSoundReason | 'idle'; guidance: string } {
    const u = this.lastBreath;
    const reason = this.dead ? 'no_mic' : u ? u.reason : 'idle';
    const reported = !!u && u.reported && !this.dead;
    return { text: reported ? `${u!.bpm!.toFixed(0)} breaths/min` : null, quality: u && !this.dead ? u.quality : null, reason, guidance: breathSoundGuidance(reason) };
  }
}

export interface AudioCards {
  breathing: HTMLElement;
  cough: HTMLElement;
  hooks: AudioHooks;
  /** call whenever a new AudioSession starts (also after a failed start): clears all state from the previous one */
  reset(): void;
}

function head(title: string, id: string): HTMLElement {
  return el('div', { class: 'card-head' }, el('h3', { id }, title), el('span', { class: 'tag' }, 'Tier 1'));
}

/** Placeholder cards when the microphone is off. */
export function buildAudioOffCards(): { breathing: HTMLElement; cough: HTMLElement } {
  return {
    breathing: el('article', { class: 'card', 'aria-labelledby': 'h-brs' }, head('Breathing (sound)', 'h-brs'),
      el('p', { class: 'value muted' }, 'Microphone is off'), el('p', { class: 'note' }, 'Turn on the microphone at the start screen. ' + BREATH_SOUND_NOTE)),
    cough: el('article', { class: 'card', 'aria-labelledby': 'h-cough' }, head('Cough sounds', 'h-cough'),
      el('p', { class: 'value muted' }, 'Microphone is off'), el('p', { class: 'note' }, COUGH_NOTE)),
  };
}

export function buildAudioCards(meter: (q: number | null) => HTMLElement, now: () => number = () => Date.now()): AudioCards {
  // ---- Breathing (sound)
  const value = el('p', { class: 'value muted', id: 'brs-value' }, 'No reading yet');
  const meterSlot = el('div', { class: 'meter-slot' }, meter(null));
  const guide = el('p', { class: 'guidance', id: 'brs-guidance', role: 'status' }, breathSoundGuidance('idle'));
  const levelBar = el('div', { class: 'level-bar', id: 'mic-level-bar' });
  const levelFill = el('span', { class: 'level-fill' });
  levelBar.append(levelFill);
  const levelText = el('span', { class: 'level-text', id: 'mic-level-text' }, 'Input level: waiting for sound…');
  const level = el('div', { class: 'level', id: 'mic-level', role: 'meter', 'aria-label': 'Microphone input level', 'aria-valuemin': '0', 'aria-valuemax': '1', 'aria-valuenow': '0' }, levelBar, levelText);
  const breathing = el('article', { class: 'card', 'aria-labelledby': 'h-brs' }, head('Breathing (sound)', 'h-brs'), value, meterSlot, guide, level,
    el('p', { class: 'note' }, BREATH_SOUND_NOTE));

  // ---- Cough sounds
  const countEl = el('p', { class: 'value', id: 'cough-count' }, coughLabel(0));
  const lastEl = el('p', { class: 'note', id: 'cough-last' }, 'Last detected: —');
  const statusEl = el('p', { class: 'note', id: 'cough-status', role: 'status' }, 'Loading the sound model…');
  const timeline = el('div', { class: 'timeline', id: 'cough-timeline', role: 'img', 'aria-label': 'Timeline of detected cough sounds this session' });
  const cough = el('article', { class: 'card', 'aria-labelledby': 'h-cough' }, head('Cough sounds', 'h-cough'), countEl, lastEl, timeline, statusEl,
    el('p', { class: 'note' }, COUGH_NOTE + ' ' + COUGH_BOUT_NOTE));

  const state = new AudioCardState();
  const renderBreath = (): void => {
    const v = state.breath;
    if (v.text) { value.textContent = v.text; value.classList.remove('muted'); }
    else { value.textContent = 'No reading yet'; value.classList.add('muted'); }
    meterSlot.replaceChildren(meter(v.quality));
    guide.textContent = v.guidance;
    breathing.dataset.reason = v.reason;
  };
  const renderCough = (): void => {
    countEl.textContent = coughLabel(state.count);
    lastEl.textContent = state.lastWallMs !== null ? `Last detected: ${formatClock(state.lastWallMs)}` : 'Last detected: —';
    timeline.replaceChildren(...state.ticks.map((ts) => {
      const tick = el('span', { class: 'tick' });
      tick.style.left = `${Math.min(99, Math.max(0, (ts / Math.max(state.t, 1)) * 100)).toFixed(1)}%`;
      return tick;
    }));
  };

  const hooks: AudioHooks = {
    onLevel(rms, peak) {
      const db = 20 * Math.log10(Math.max(rms, 1e-5));
      const frac = Math.min(1, Math.max(0, (db + 60) / 60)); // -60 dBFS .. 0 dBFS
      levelFill.style.width = `${(frac * 100).toFixed(0)}%`;
      level.setAttribute('aria-valuenow', frac.toFixed(2));
      levelText.textContent = rms < 1e-4 ? 'Input level: silent' : `Input level: ${db.toFixed(0)} dBFS${peak >= 0.99 ? ' (clipping)' : ''}`;
    },
    onBreath(u) { state.setBreath(u); renderBreath(); },
    onNoMic(d) { state.setNoMic(d); renderBreath(); },
    onCough(r) { state.setCough(r, now()); renderCough(); },
    onYamnet(ys, msg) {
      statusEl.textContent = ys === 'loading' ? 'Loading the sound model…' : ys === 'ready' ? 'Listening on this device. Nothing is recorded or sent.'
        : 'Sound model failed to load' + (msg ? ': ' + msg : '');
      cough.dataset.state = ys;
    },
  };
  const reset = (): void => {
    state.reset();
    renderBreath(); renderCough();
    levelFill.style.width = '0%';
    level.setAttribute('aria-valuenow', '0');
    levelText.textContent = 'Input level: waiting for sound…';
    hooks.onYamnet('loading');
  };
  return { breathing, cough, hooks, reset };
}
