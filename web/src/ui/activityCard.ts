// "Activity" (Experimental, W9): minutes active, still and away this session and a small posture-time bar, from the pose chain
// (the DOM-free ActivityTimeline / ActivitySummary in src/core/events.ts, ports of the Python reference). Wellbeing trend only:
// no targets, no interpretation. Nothing is stored beyond the session summary the export offers. Not a medical device.
import params from '../../../shared/params.json';
import type { PoseView } from '../capture/poseSession';
import { ActivitySummary, ActivityTimeline, POSTURES, naiveWallMs, type ActivitySummaryResult } from '../core/events';
import { activitySummaryRow } from '../export/wellbeing';
import type { MobilitySummaryFields } from '../export/mobility';
import { el } from './dom';

export const ACTIVITY_NOTE = 'Experimental. Active means moving while in view; still means in view without moving much. Compare with your own earlier days.';
export const ACTIVITY_PARAMS = { still: params.common.still_motion_bodysizes_s, maxStep: params.events.timeline.max_step_s, minReport: params.activity.min_report_s };

export const POSTURE_LABEL: Record<string, string> = { sitting: 'Sitting', standing: 'Standing', walking: 'Walking', lying: 'Lying', absent: 'Away' };

const fmtMin = (s: number): string => (s / 60).toFixed(1);

/** Plain text for the card (rounding happens here, not in the core). */
export function activityText(r: ActivitySummaryResult): { headline: string; longest: string } {
  if (!r.reported) return { headline: 'Collecting this session’s activity…', longest: '' };
  return {
    headline: `Active ${fmtMin(r.active_s)} min · Still ${fmtMin(r.still_s)} min · Away ${fmtMin(r.absent_s)} min`,
    longest: `Longest still stretch: ${fmtMin(r.longest_still_s)} min`,
  };
}

/** Accessible description of the posture bar: every posture with its share of the session. */
export function postureBarText(totals: Record<string, number>): string {
  const sum = POSTURES.reduce((n, p) => n + (totals[p] ?? 0), 0);
  if (!(sum > 0)) return 'Posture time: no data yet';
  return 'Posture time: ' + POSTURES.filter((p) => (totals[p] ?? 0) > 0).map((p) => `${POSTURE_LABEL[p]!.toLowerCase()} ${Math.round((100 * (totals[p] ?? 0)) / sum)} %`).join(', ');
}

const SVG_NS = 'http://www.w3.org/2000/svg';

export interface ActivityCard {
  root: HTMLElement;
  onView(v: Pick<PoseView, 't' | 'state'>): void;
  /** a new camera stream or a model restart: the clock may start over, so the next step is not measured against the old one */
  breakStream(): void;
}

export function buildActivityCard(deps: { record(row: MobilitySummaryFields): void; now?: () => number }): ActivityCard {
  const now = deps.now ?? (() => Date.now());
  // hour buckets need a wall-clock anchor: the first observed frame time is 'now' (re-anchored after a new stream)
  const anchor = (t: number): number => naiveWallMs(new Date(now() - t * 1000));
  let anchored = false;
  const timeline = new ActivityTimeline(0, ACTIVITY_PARAMS.maxStep);
  const summary = new ActivitySummary(ACTIVITY_PARAMS.still, ACTIVITY_PARAMS.maxStep, ACTIVITY_PARAMS.minReport);
  let lastT = -Infinity;
  let lastSec = -1;

  const headline = el('p', { class: 'value muted', id: 'activity-value', 'aria-live': 'polite' }, 'Starts when the camera is on.');
  const longest = el('p', { class: 'note', id: 'activity-longest' });
  const svg = document.createElementNS?.(SVG_NS, 'svg') ?? el('div');
  svg.setAttribute('class', 'posture-bar');
  svg.setAttribute('viewBox', '0 0 100 8');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img');
  svg.setAttribute('id', 'activity-bar');
  svg.setAttribute('aria-label', postureBarText({}));
  const barText = el('p', { class: 'note', id: 'activity-bar-text' }, postureBarText({}));
  const root = el('article', { class: 'card', id: 'activity-card', 'aria-labelledby': 'h-activity' },
    el('div', { class: 'card-head' }, el('h3', { id: 'h-activity' }, 'Activity'), el('span', { class: 'tag exp' }, 'Experimental')),
    headline, longest, svg, barText,
    el('p', { class: 'note', id: 'activity-note' }, ACTIVITY_NOTE));

  function drawBar(totals: Record<string, number>): void {
    const sum = POSTURES.reduce((n, p) => n + (totals[p] ?? 0), 0);
    const kids: Element[] = [];
    let x = 0;
    if (sum > 0 && document.createElementNS) {
      for (const p of POSTURES) {
        const w = (100 * (totals[p] ?? 0)) / sum;
        if (!(w > 0)) continue;
        const r = document.createElementNS(SVG_NS, 'rect');
        r.setAttribute('x', x.toFixed(2)); r.setAttribute('y', '0'); r.setAttribute('width', w.toFixed(2)); r.setAttribute('height', '8');
        r.setAttribute('class', `pb pb-${p}`);
        kids.push(r);
        x += w;
      }
    }
    svg.replaceChildren(...kids);
    const text = postureBarText(totals);
    svg.setAttribute('aria-label', text);
    barText.textContent = text;
  }

  function render(): void {
    const r = summary.toDict();
    const t = activityText(r);
    headline.textContent = t.headline;
    headline.classList.toggle('muted', !r.reported);
    longest.textContent = t.longest;
    headline.setAttribute('data-active-s', r.active_s.toFixed(1));
    headline.setAttribute('data-still-s', r.still_s.toFixed(1));
    headline.setAttribute('data-away-s', r.absent_s.toFixed(1));
    headline.setAttribute('data-reported', String(r.reported));
    drawBar(timeline.totals());
    if (r.total_s > 0) deps.record(activitySummaryRow(r));
  }

  return {
    root,
    breakStream() { timeline.breakStream(); summary.breakStream(); lastT = -Infinity; anchored = false; },
    onView(v) {
      if (v.t < lastT) this.breakStream(); // the clock started over (new stream): do not measure against the old time
      lastT = v.t;
      if (!anchored) { timeline.startNaiveMs = anchor(v.t); anchored = true; }
      timeline.update(v.state);
      summary.update(v.state);
      const sec = Math.floor(summary.totalS);
      if (sec !== lastSec) { lastSec = sec; render(); }
    },
  };
}
