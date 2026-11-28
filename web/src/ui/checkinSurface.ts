// The check-in banner and event list, shared by the camera cards (falls, inactivity) and the sound cards (distress sounds), so a
// microphone-only session (night mode) has them too. Wellbeing wording only; it suggests a check-in and nothing else.
import { el } from './dom';
import { EventLog } from './eventLog';

export interface CheckinSurface {
  log: EventLog;
  /** place above the live view; shows the newest unacknowledged suggestion with an Acknowledge button */
  banner: HTMLElement;
  /** the event list (newest first) and its "none yet" note */
  list: HTMLElement;
  empty: HTMLElement;
  /** call after the log changed */
  refresh(): void;
}

export const formatClock = (ms: number): string => {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};

export function buildCheckinSurface(log: EventLog = new EventLog()): CheckinSurface {
  const listEl = el('ol', { class: 'event-list', id: 'event-list' });
  const empty = el('p', { class: 'note', id: 'event-empty' }, 'No check-in suggestions this session.');
  const bannerText = el('p', { class: 'banner-text', id: 'banner-text' });
  const ackBtn = el('button', { type: 'button', class: 'btn', id: 'banner-ack' }, 'Acknowledge');
  const banner = el('div', { class: 'banner', id: 'checkin-banner', role: 'alert' }, bannerText, ackBtn);
  banner.hidden = true;
  const refresh = (): void => {
    listEl.replaceChildren(
      ...[...log.visible].reverse().map((e) => {
        const btn = el('button', { type: 'button', class: 'btn ghost', 'data-ack': String(e.id) }, e.acknowledged ? 'Acknowledged' : 'Acknowledge');
        btn.disabled = e.acknowledged;
        btn.addEventListener('click', () => { if (log.acknowledge(e.id)) refresh(); });
        return el('li', { class: e.acknowledged ? 'event done' : 'event', 'data-kind': e.kind }, el('span', { class: 'event-time' }, formatClock(e.at)), el('span', { class: 'event-text' }, e.text), btn);
      }),
    );
    empty.hidden = log.visible.length > 0;
    const pend = log.pending;
    banner.hidden = pend === null;
    if (pend) bannerText.textContent = `${pend.text}${log.pendingCount > 1 ? ` (${log.pendingCount} waiting)` : ''}`;
  };
  ackBtn.addEventListener('click', () => { const p = log.pending; if (p && log.acknowledge(p.id)) refresh(); });
  refresh();
  return { log, banner, list: listEl, empty, refresh };
}
