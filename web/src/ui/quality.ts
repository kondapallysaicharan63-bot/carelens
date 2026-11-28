import { el } from './dom';

export type QualityLevel = 'none' | 'weak' | 'ok' | 'good';

export const QUALITY_WEAK_BELOW = 0.34;
export const QUALITY_GOOD_FROM = 0.67;

export function qualityLevel(q: number | null | undefined): QualityLevel {
  if (q === null || q === undefined || Number.isNaN(q)) return 'none';
  const v = Math.min(1, Math.max(0, q));
  if (v < QUALITY_WEAK_BELOW) return 'weak';
  if (v < QUALITY_GOOD_FROM) return 'ok';
  return 'good';
}

const LABEL: Record<QualityLevel, string> = {
  none: 'no signal yet',
  weak: 'weak',
  ok: 'ok',
  good: 'good',
};
const BARS: Record<QualityLevel, number> = { none: 0, weak: 1, ok: 2, good: 3 };

export function qualityText(q: number | null | undefined): string {
  return `Signal quality: ${LABEL[qualityLevel(q)]}`;
}

/** Meter with 3 segments (shape) plus a text label, so it never relies on colour alone. */
export function qualityMeter(q: number | null | undefined): HTMLElement {
  const level = qualityLevel(q);
  const bars = el('span', { class: 'qbars', 'aria-hidden': 'true' });
  for (let i = 0; i < 3; i++) bars.append(el('span', { class: i < BARS[level] ? 'qbar on' : 'qbar' }));
  const wrap = el('div', {
    class: `quality q-${level}`,
    role: 'meter',
    'aria-label': 'Signal quality',
    'aria-valuemin': '0',
    'aria-valuemax': '1',
    'aria-valuetext': LABEL[level],
  });
  if (level !== 'none') wrap.setAttribute('aria-valuenow', String(Math.min(1, Math.max(0, q as number))));
  wrap.append(bars, el('span', { class: 'qtext' }, qualityText(q)));
  return wrap;
}
