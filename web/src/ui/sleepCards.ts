// Tier 2 (Experimental) sound cards: "Snoring sounds", "Possible pauses in breathing sounds", "Talking time (wellbeing trend)".
// The DOM-free core does the work (src/core/audio.ts, sleepSounds.ts); this file only shows it. Wellbeing wording only: counts and
// minutes, never a condition. Not a sleep test. Not a medical device.
import type { LabelView } from '../core/audio';
import type { PauseView } from '../core/sleepSounds';
import type { SoundsView } from '../capture/audioSession';
import { el } from './dom';

export const SNORING_TITLE = 'Snoring sounds';
export const PAUSES_TITLE = 'Possible pauses in breathing sounds';
export const TALKING_TITLE = 'Talking time (wellbeing trend)';
export const SNORING_NOTE = 'Counts sounds the model recognises as snoring — wellbeing trend only. Phone placement and room noise change it.';
export const PAUSES_NOTE = 'Counts gaps in breathing sounds after regular breathing. Phone placement, bedding and movement all affect it. Not a sleep test — talk to a clinician if you’re worried about your sleep.';
export const TALKING_NOTE = 'Time the model hears speech — a wellbeing trend, not a measure of anything else. Nothing is recorded or transcribed.';
export const PAUSES_WAIT = 'Waiting for regular breathing sounds — place the phone near the bed and stay quiet.';
export const MIN_REGULAR_MIN = 30;
export const PAUSES_NO_MODEL = 'Pause counting needs the sound model, which didn\'t load.';
export const SOUNDS_NO_MODEL = 'Needs the sound model, which didn\'t load.';

const min1 = (s: number): string => (s / 60).toFixed(1);

export function snoringText(v: LabelView | null): string {
  return v ? `Snoring sounds this session: ${v.count} · ${min1(v.total_s)} min` : 'Snoring sounds this session: 0 · 0.0 min';
}
export function snoringHours(v: LabelView | null): string {
  return v && v.per_hour.length ? 'By session hour: ' + v.per_hour.map((n, i) => `${i + 1}: ${n}`).join(' · ') : 'By session hour: —';
}
export function pausesText(v: PauseView | null): string { return `Possible pauses this session: ${v ? v.count : 0}`; }
/** The per-hour figure is withheld below 30 min of regular breathing sounds; the card says how much it has. */
export function pausesRate(v: PauseView | null): string {
  if (!v || v.regular_s <= 0) return PAUSES_WAIT;
  if (v.rate_per_h === null) return `Per-hour figure withheld until ${MIN_REGULAR_MIN} min of regular breathing sounds (${min1(v.regular_s)} min so far).`;
  return `About ${v.rate_per_h.toFixed(1)} per hour of regular breathing sounds (${min1(v.regular_s)} min).`;
}
export function pausesDurations(v: PauseView | null): string {
  return v && v.pauses.length ? 'Lengths: ' + v.pauses.slice(-10).map((p) => `${p.duration_s.toFixed(0)} s`).join(', ') + (v.pauses.length > 10 ? ' (latest 10)' : '') : 'Lengths: —';
}
export function talkingText(v: LabelView | null): string { return `Talking time: ${v ? min1(v.total_s) : '0.0'} min`; }
export function talkingHours(v: LabelView | null): string {
  return v && v.per_hour_s.length ? 'By session hour: ' + v.per_hour_s.map((s, i) => `${i + 1}: ${min1(s)} min`).join(' · ') : 'By session hour: —';
}

function head(title: string, id: string): HTMLElement {
  return el('div', { class: 'card-head' }, el('h3', { id }, title), el('span', { class: 'tag exp' }, 'Experimental'));
}

export interface SleepCards {
  snoring: HTMLElement;
  pauses: HTMLElement;
  talking: HTMLElement;
  onSounds(v: SoundsView): void;
  /** the sound model's state: when it failed to load nothing is counted, and the cards say so instead of showing zeros */
  setModel(state: 'loading' | 'ready' | 'failed'): void;
  /** a new audio session starts: nothing from the previous one may linger */
  reset(): void;
}

export function buildSleepCards(): SleepCards {
  const snoreCount = el('p', { class: 'value', id: 'snore-count' }, snoringText(null));
  const snoreHours = el('p', { class: 'note', id: 'snore-hours' }, snoringHours(null));
  const snoring = el('article', { class: 'card', 'aria-labelledby': 'h-snore' }, head(SNORING_TITLE, 'h-snore'), snoreCount, snoreHours, el('p', { class: 'note' }, SNORING_NOTE));

  const pauseCount = el('p', { class: 'value', id: 'pause-count' }, pausesText(null));
  const pauseRate = el('p', { class: 'guidance', id: 'pause-rate', role: 'status' }, pausesRate(null));
  const pauseLens = el('p', { class: 'note', id: 'pause-lengths' }, pausesDurations(null));
  const pauses = el('article', { class: 'card', 'aria-labelledby': 'h-pause' }, head(PAUSES_TITLE, 'h-pause'), pauseCount, pauseRate, pauseLens, el('p', { class: 'note', id: 'pause-note' }, PAUSES_NOTE));

  const talkMin = el('p', { class: 'value', id: 'talk-minutes' }, talkingText(null));
  const talkHours = el('p', { class: 'note', id: 'talk-hours' }, talkingHours(null));
  const talking = el('article', { class: 'card', 'aria-labelledby': 'h-talk' }, head(TALKING_TITLE, 'h-talk'), talkMin, talkHours, el('p', { class: 'note' }, TALKING_NOTE));

  let failed = false;
  const render = (v: SoundsView | null): void => {
    if (failed) {
      snoreCount.textContent = talkMin.textContent = SOUNDS_NO_MODEL;
      snoreHours.textContent = talkHours.textContent = '';
      pauseCount.textContent = 'Possible pauses this session: —';
      pauseRate.textContent = PAUSES_NO_MODEL;
      pauseLens.textContent = '';
      for (const c of [snoring, pauses, talking]) c.dataset.model = 'failed';
      return;
    }
    for (const c of [snoring, pauses, talking]) delete c.dataset.model;
    snoreCount.textContent = snoringText(v?.snoring ?? null);
    snoreHours.textContent = snoringHours(v?.snoring ?? null);
    pauseCount.textContent = pausesText(v?.pauses ?? null);
    pauseRate.textContent = pausesRate(v?.pauses ?? null);
    pauseLens.textContent = pausesDurations(v?.pauses ?? null);
    talkMin.textContent = talkingText(v?.talking ?? null);
    talkHours.textContent = talkingHours(v?.talking ?? null);
    talkMin.dataset.seconds = String(Math.round(v?.talking.total_s ?? 0));
    pauseCount.dataset.count = String(v?.pauses.count ?? 0);
    snoreCount.dataset.count = String(v?.snoring.count ?? 0);
  };
  render(null);
  return {
    snoring, pauses, talking, onSounds: render,
    setModel: (st) => { failed = st === 'failed'; render(null); },
    reset: () => { failed = false; render(null); },
  };
}
