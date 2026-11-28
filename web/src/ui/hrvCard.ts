// "Heart-rate variability" card (Experimental). Active only in fingertip mode after >= min_duration_s of steady contact.
// RMSSD and SDNN are shown as plain numbers with their quality, never as a stress, recovery or health score.
import type { HrvReason, HrvResult } from '../core/hrv';
import { el } from './dom';

export const HRV_NOTE =
  'Experimental. Short recordings at camera frame rates give rough values; trends over many days mean more than a single number. Camera timing adds noise, so RMSSD tends to read higher than a chest strap or ECG — compare trends, not single numbers. Not a stress or health score.';
export const HRV_OFF_TEXT = 'Available in fingertip mode only.';

export function hrvWithheldText(reason: HrvReason): string {
  switch (reason) {
    case 'too_many_artifacts': return 'Too many irregular beats for a steady value. Rest your hand, press lightly and keep still.';
    case 'unsteady': return 'The signal was not steady enough. Rest your hand, press lightly and keep still.';
    case 'low_fps': return 'This camera runs too slowly for heart-rate variability.';
    case 'collecting': return 'Collecting more beats…';
    case 'too_short':
    case 'ok': return '';
  }
}

export interface HrvCardText { value: string; muted: boolean; detail: string | null; quality: string | null }

const holdText = (segS: number, min: number): string => `Hold steady for ${min} s after the signal settles to see this (${Math.floor(Math.min(segS, min))} / ${min} s).`;

/** Card text for one state (rounding happens here, not in the core). `res` null: segment shorter than `minDurationS`.
 *  `last`: the last reported result, kept on screen until a new measurement reports. */
export function hrvCardText(res: HrvResult | null, segS: number, minDurationS: number, last: HrvResult | null = null): HrvCardText {
  const shown = res?.reported ? res : last;
  if (shown && shown.reported && shown.rmssd_ms !== null && shown.sdnn_ms !== null) {
    const stale = shown !== res;
    return {
      value: `RMSSD ${shown.rmssd_ms.toFixed(0)} ms, SDNN ${shown.sdnn_ms.toFixed(0)} ms`, muted: false,
      detail: `${shown.n_kept} beats over ${Math.round(shown.duration_s)} s. A wellbeing trend only.${stale ? ' Last completed measurement.' : ''}`,
      quality: `Quality: ${Math.round(shown.kept_frac * 100)} % of beats kept.`,
    };
  }
  if (res === null) return { value: holdText(segS, minDurationS), muted: true, detail: null, quality: null };
  const soft = res.reason === 'low_fps' || res.reason === 'unsteady' || res.reason === 'collecting';
  return { value: hrvWithheldText(res.reason), muted: true, detail: `${Math.round(res.duration_s)} s ${res.reason === 'collecting' ? 'so far' : 'of contact'}.`, quality: soft ? null : `Quality: ${Math.round(res.kept_frac * 100)} % of beats kept.` };
}

export interface HrvCard {
  /** the Experimental section wrapping the card */
  section: HTMLElement;
  update(res: HrvResult | null, segmentS: number, last?: HrvResult | null): void;
}

export function buildHrvCard(minDurationS: number): HrvCard {
  const value = el('p', { class: 'value muted', id: 'fp-hrv' }, holdText(0, minDurationS));
  const detail = el('p', { class: 'note', id: 'fp-hrv-detail' });
  const quality = el('p', { class: 'note', id: 'fp-hrv-quality' });
  detail.hidden = true;
  quality.hidden = true;
  const card = el(
    'article',
    { class: 'card', id: 'fp-hrv-card' },
    el('div', { class: 'card-head' }, el('h3', {}, 'Heart-rate variability'), el('span', { class: 'tag exp' }, 'Experimental')),
    value,
    detail,
    quality,
    el('p', { class: 'note' }, HRV_NOTE),
  );
  const section = el(
    'section',
    { class: 'results experimental', 'aria-labelledby': 'h-fp-exp' },
    el('h2', { id: 'h-fp-exp' }, 'Experimental'),
    el('p', { class: 'sub' }, 'Tier 2: early, less reliable features. Treat as curiosity, not information.'),
    el('div', { class: 'grid' }, card),
  );
  return {
    section,
    update(res, segS, last = null) {
      const t = hrvCardText(res, segS, minDurationS, last);
      value.textContent = t.value;
      value.classList.toggle('muted', t.muted);
      detail.hidden = t.detail === null;
      detail.textContent = t.detail ?? '';
      quality.hidden = t.quality === null;
      quality.textContent = t.quality ?? '';
    },
  };
}
