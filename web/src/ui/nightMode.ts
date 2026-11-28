// Night mode view (microphone near the bed): dims the page, shows a minimal live view with the three sound results, warns to keep
// the phone plugged in, and shows whether the screen is being kept awake. Wellbeing wording only. Not a medical device.
import { PLUG_TEXT, WAKE_TEXT, type WakeStatus } from '../capture/wakeLock';
import { el } from './dom';

export const NIGHT_TITLE = 'Night mode';
export const NIGHT_HINT = 'Put the phone near the bed with the microphone facing you. Sounds are analysed on this device only; nothing is recorded or sent.';

export interface Choices { camera: boolean; mic: boolean }

/**
 * Camera / microphone ticks around night mode. Night mode needs the microphone: ticking it forces the microphone on (and locked) and
 * starts camera-off (the camera may be ticked again); unticking it restores exactly what was chosen before.
 */
export class NightChoices {
  private prev: Choices | null = null;
  /** `on`: night mode ticked; `cur`: the current ticks. Returns the ticks to show and whether the microphone tick is locked. */
  apply(on: boolean, cur: Choices): Choices & { micLocked: boolean } {
    if (on) {
      this.prev ??= { ...cur };
      return { camera: false, mic: true, micLocked: true };
    }
    const p = this.prev ?? cur;
    this.prev = null;
    return { ...p, micLocked: false };
  }
}

export const NIGHT_NO_MIC = 'No microphone is available, so night mode cannot listen. Allow the microphone and start again.';
export const NIGHT_WAITING = 'Starting the microphone…';
export const NIGHT_LISTENING = 'Listening… results appear here.';

export interface NightPanel {
  root: HTMLElement;
  setWake(s: WakeStatus): void;
  setSummary(text: string | null): void;
}

/** Dims the whole page (CSS, html[data-night]) and, on exit, restores it. */
export function setNightClass(root: HTMLElement, on: boolean, details = false): void {
  if (on) root.setAttribute('data-night', 'on'); else root.removeAttribute('data-night');
  if (on && details) root.setAttribute('data-night-details', 'on'); else root.removeAttribute('data-night-details');
}

export function buildNightPanel(root: HTMLElement): NightPanel {
  const plug = el('p', { class: 'night-plug', id: 'night-plug' }, PLUG_TEXT);
  const wake = el('p', { class: 'note', id: 'night-wake', role: 'status' }, WAKE_TEXT.idle);
  const summary = el('p', { class: 'night-summary', id: 'night-summary', role: 'status' }, NIGHT_WAITING);
  const toggle = el('button', { type: 'button', class: 'btn ghost', id: 'night-details', 'aria-pressed': 'false' }, 'Show all cards');
  toggle.addEventListener('click', () => {
    const on = root.getAttribute('data-night-details') !== 'on';
    setNightClass(root, true, on);
    toggle.textContent = on ? 'Hide cards' : 'Show all cards';
    toggle.setAttribute('aria-pressed', String(on));
  });
  const panel = el('section', { class: 'panel night', id: 'night-panel', 'aria-labelledby': 'h-night' }, el('h2', { id: 'h-night' }, NIGHT_TITLE), plug, el('p', { class: 'note' }, NIGHT_HINT), summary, wake, toggle);
  return { root: panel, setWake: (s) => { wake.textContent = WAKE_TEXT[s]; wake.dataset.status = s; }, setSummary: (t) => { if (t) summary.textContent = t; } };
}
