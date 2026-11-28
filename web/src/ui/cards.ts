import { el } from './dom';
import { qualityMeter } from './quality';

export interface Card {
  root: HTMLElement;
  setValue(text: string, quality?: number | null): void;
}

export function makeCard(title: string, tier: 1 | 2, note: string, value = 'Not available yet'): Card {
  const valueEl = el('p', { class: 'value muted' }, value);
  const meterSlot = el('div', { class: 'meter-slot' }, qualityMeter(null));
  const root = el(
    'article',
    { class: 'card' },
    el('div', { class: 'card-head' }, el('h3', {}, title), el('span', { class: tier === 1 ? 'tag' : 'tag exp' }, tier === 1 ? 'Tier 1' : 'Experimental')),
    valueEl,
    meterSlot,
    el('p', { class: 'note' }, note),
  );
  return {
    root,
    setValue(text, quality = null) {
      valueEl.textContent = text;
      valueEl.classList.remove('muted');
      meterSlot.replaceChildren(qualityMeter(quality));
    },
  };
}

export const TIER1: [string, string][] = [
  ['Heart rate (fingertip)', 'Phone back camera with torch. Wellbeing trend only.'],
  ['Heart rate (face)', 'Contactless, needs steady light and a still face.'],
  ['Breathing rate', 'From chest motion, when still.'],
  ['Breathing (sound)', 'From breath sounds with a close microphone.'],
  ['Fall and inactivity check-in', 'Suggests a check-in; it does not call anyone.'],
  ['Cough sounds', 'On-device sound classification.'],
];
export const TIER2: [string, string][] = [
  ['Heart-rate variability', 'Fingertip only. Experimental.'],
  ['Sit-to-stand and walking speed', 'Experimental movement trends.'],
  ['Hand tremor', 'Experimental; not a screening test.'],
  ['Blink rate and eye closure', 'Experimental.'],
  ['Snoring', 'Experimental.'],
];
