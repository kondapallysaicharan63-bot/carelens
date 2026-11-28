// Tier 1 pose cards: "In view", "Posture", "Breathing (camera)", "Fall and inactivity check-in" (+ banner).
// The DOM-free core does the work (src/core/*); this file only shows it. Wellbeing wording only. Not a medical device.
import params from '../../../shared/params.json';
import type { PoseView } from '../capture/poseSession';
import type { PoseEvent } from '../core/events';
import type { Posture } from '../core/features';
import { el } from './dom';
import { seedAndSubscribe } from './controls';
import { EventLog } from './eventLog';

export const BREATHING_HINT = 'Sit still facing the camera with your chest visible';
export const BREATHING_NOTE = 'Camera breathing estimates are a wellbeing trend only. Clothing, movement and low light make them less reliable.';
export const ERROR_TEXT = 'Camera processing error — restarting detector';
export const CHECKIN_NOTE = 'Suggests a check-in; it does not call anyone, send anything or make a sound.';

export function postureLabel(p: Posture): string {
  switch (p) {
    case 'standing': return 'standing';
    case 'walking': return 'standing (walking)';
    case 'sitting': return 'sitting';
    case 'lying': return 'lying';
    default: return '—';
  }
}

export function breathingGuidance(status: PoseView['breathing']['status'], reported: boolean, haveEstimate: boolean): string {
  if (reported) return 'Good signal. Keep still.';
  switch (status) {
    case 'absent': return BREATHING_HINT;
    case 'moving': return 'Hold still';
    case 'no_chest': return 'Show your shoulders and chest to the camera';
    case 'too_small': return 'Move closer so your chest is visible';
    case 'held': return 'Looking for you again…';
    default: return haveEstimate ? `Signal is weak. ${BREATHING_HINT}` : 'Collecting a steady signal (about 30 seconds)…';
  }
}

export function formatClock(ms: number): string {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export interface PoseCards {
  inView: HTMLElement;
  posture: HTMLElement;
  breathing: HTMLElement;
  checkins: HTMLElement;
  /** place above the video preview */
  banner: HTMLElement;
  hooks: { onView(v: PoseView): void; onEvent(e: PoseEvent): void; onError(failing: boolean): void };
  setStatus(text: string): void;
  /** registers the handler and seeds it with the current selection (seconds) */
  onInactivityChange(cb: (seconds: number) => void): void;
  log: EventLog;
}

function head(title: string, id: string): HTMLElement {
  return el('div', { class: 'card-head' }, el('h3', { id }, title), el('span', { class: 'tag' }, 'Tier 1'));
}

export function buildPoseCards(meter: (q: number | null) => HTMLElement, now: () => number = () => Date.now()): PoseCards {
  const log = new EventLog();

  // ---- In view
  const inViewValue = el('p', { class: 'value muted', id: 'inview-value' }, 'In view: —');
  const inViewNote = el('p', { class: 'note', id: 'inview-note' }, 'Loading the pose model…');
  const inView = el('article', { class: 'card', 'aria-labelledby': 'h-inview' }, head('In view', 'h-inview'), inViewValue, inViewNote);

  // ---- Posture
  const postureValue = el('p', { class: 'value muted', id: 'posture-value' }, 'Posture: —');
  const posture = el('article', { class: 'card', 'aria-labelledby': 'h-posture' }, head('Posture', 'h-posture'), postureValue,
    el('p', { class: 'note' }, 'Standing, sitting or lying, from body landmarks on this device. Nothing is recorded.'));

  // ---- Breathing (camera)
  const brValue = el('p', { class: 'value muted', id: 'br-value' }, 'No reading yet');
  const brMeter = el('div', { class: 'meter-slot' }, meter(null));
  const brGuide = el('p', { class: 'guidance', id: 'br-guidance', role: 'status' }, BREATHING_HINT);
  const breathing = el('article', { class: 'card', 'aria-labelledby': 'h-br' }, head('Breathing (camera)', 'h-br'), brValue, brMeter, brGuide,
    el('p', { class: 'note' }, BREATHING_NOTE));

  // ---- Check-ins
  const sel = el('select', { id: 'inactivity-select', 'aria-label': 'Check-in after no movement for' });
  for (const s of params.scheduling.inactivity_demo_options_s) {
    const o = el('option', { value: String(s) }, `${s / 60} min`);
    o.selected = s === params.scheduling.inactivity_demo_default_s;
    sel.append(o);
  }
  const listEl = el('ol', { class: 'event-list', id: 'event-list' });
  const empty = el('p', { class: 'note', id: 'event-empty' }, 'No check-in suggestions this session.');
  const checkins = el('article', { class: 'card', 'aria-labelledby': 'h-checkin' }, head('Fall and inactivity check-in', 'h-checkin'),
    el('label', { class: 'field' }, el('span', {}, 'Suggest a check-in after no movement for'), sel),
    el('p', { class: 'note' }, `Demo setting: the same inactivity detector as the full system (default ${params.events.inactivity.seconds / 60} min) with a shorter time so you can try it.`),
    empty, listEl, el('p', { class: 'note' }, CHECKIN_NOTE));

  // ---- Banner
  const bannerText = el('p', { class: 'banner-text', id: 'banner-text' });
  const ackBtn = el('button', { type: 'button', class: 'btn', id: 'banner-ack' }, 'Acknowledge');
  const banner = el('div', { class: 'banner', id: 'checkin-banner', role: 'alert' }, bannerText, ackBtn);
  banner.hidden = true;

  const render = (): void => {
    listEl.replaceChildren(
      ...[...log.entries].reverse().map((e) => {
        const btn = el('button', { type: 'button', class: 'btn ghost', 'data-ack': String(e.id) }, e.acknowledged ? 'Acknowledged' : 'Acknowledge');
        btn.disabled = e.acknowledged;
        btn.addEventListener('click', () => { if (log.acknowledge(e.id)) render(); });
        return el('li', { class: e.acknowledged ? 'event done' : 'event' }, el('span', { class: 'event-time' }, formatClock(e.at)), el('span', { class: 'event-text' }, e.text), btn);
      }),
    );
    empty.hidden = log.entries.length > 0;
    const pend = log.pending;
    banner.hidden = pend === null;
    if (pend) bannerText.textContent = `${pend.text}${log.pendingCount > 1 ? ` (${log.pendingCount} waiting)` : ''}`;
  };
  ackBtn.addEventListener('click', () => { const p = log.pending; if (p && log.acknowledge(p.id)) render(); });

  let lastPosture: Posture = 'absent';
  const hooks: PoseCards['hooks'] = {
    onView(v) {
      const present = v.state.present;
      inViewValue.textContent = `In view: ${present ? 'yes' : 'no'}`;
      inViewValue.classList.remove('muted');
      inViewNote.textContent = present ? (v.held ? 'Briefly lost, still counting you as in view.' : 'Detected on this device.') : 'No person detected.';
      if (present) lastPosture = v.state.posture;
      postureValue.textContent = `Posture: ${present ? postureLabel(lastPosture) : '—'}`;
      postureValue.classList.toggle('muted', !present);
      const est = v.breathing.estimate;
      const reported = !!est && est.reported;
      if (reported) { brValue.textContent = `${est!.value.toFixed(0)} breaths/min`; brValue.classList.remove('muted'); }
      else { brValue.textContent = 'No reading yet'; brValue.classList.add('muted'); }
      brMeter.replaceChildren(meter(est ? est.quality : null));
      brGuide.textContent = breathingGuidance(v.breathing.status, reported, !!est);
      breathing.dataset.state = v.breathing.status;
    },
    onEvent(e) {
      log.add(e, now());
      render();
    },
    onError(failing) {
      if (!failing) return; // the next view overwrites the messages
      inViewNote.textContent = ERROR_TEXT;
      brGuide.textContent = ERROR_TEXT;
    },
  };
  render();
  return {
    inView, posture, breathing, checkins, banner, hooks, log,
    setStatus: (t) => { inViewNote.textContent = t; },
    onInactivityChange: (cb) => seedAndSubscribe(() => Number(sel.value), (h) => sel.addEventListener('change', h), cb),
  };
}
