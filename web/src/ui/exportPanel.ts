// "Your data" panel: download this session as JSON / CSV (file download only), optional save of a small summary on
// this device (IndexedDB, off by default), and "Delete all saved data". Nothing is ever sent anywhere.
import { LIMIT_TEXT, SessionRecorder, type SessionSummary } from '../export/recorder';
import { deleteAllSavedData, listSummaries, saveSummary } from '../export/saved';
import { el } from './dom';

/** File download through a Blob object URL (no network request; works under the page's CSP). */
export function downloadText(name: string, text: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = el('a', { href: url, download: name });
  a.style.display = 'none';
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

const METRIC_LABEL: Record<string, string> = { hr_fingertip: 'HR fingertip', hr_face: 'HR face', br_camera: 'Breathing (camera)', br_sound: 'Breathing (sound)' };
const UNIT: Record<string, string> = { hr_fingertip: 'bpm', hr_face: 'bpm', br_camera: '/min', br_sound: '/min' };

export function summaryLine(s: SessionSummary): string {
  const meds = Object.entries(s.medians).map(([m, v]) => `${METRIC_LABEL[m] ?? m}: ${v === null || v === undefined ? 'no reported reading' : `${v} ${UNIT[m] ?? ''}`} (median)`);
  const mins = Math.floor(s.duration_s / 60), sec = s.duration_s % 60;
  return `${s.started_at.slice(0, 16).replace('T', ' ')} · ${mins} min ${sec} s · ${meds.length ? meds.join(' · ') : 'no estimates'}`;
}

export interface ExportPanel {
  root: HTMLElement;
  /** the session whose data the buttons act on (the latest one, also after Stop) */
  setRecorder(r: SessionRecorder | null): void;
  /** call when a session ends: saves its summary if the checkbox is on */
  sessionEnded(r: SessionRecorder): Promise<void>;
  refresh(): Promise<void>;
}

export function buildExportPanel(): ExportPanel {
  let rec: SessionRecorder | null = null;
  const jsonBtn = el('button', { type: 'button', class: 'btn', id: 'dl-json' }, 'Download JSON');
  const csvBtn = el('button', { type: 'button', class: 'btn', id: 'dl-csv' }, 'Download CSV');
  const status = el('p', { class: 'note', id: 'data-status', role: 'status' }, 'No session yet.');
  const save = el('input', { type: 'checkbox', id: 'save-summary' });
  const list = el('ul', { class: 'saved-list', id: 'saved-list' });
  const empty = el('p', { class: 'note', id: 'saved-empty' }, 'No saved summaries.');
  const del = el('button', { type: 'button', class: 'btn ghost', id: 'delete-all' }, 'Delete all saved data');
  const confirmText = el('span', {}, 'Delete all saved summaries and remembered settings on this device?');
  const yes = el('button', { type: 'button', class: 'btn', id: 'delete-confirm' }, 'Yes, delete');
  const no = el('button', { type: 'button', class: 'btn ghost', id: 'delete-cancel' }, 'Cancel');
  const confirmRow = el('div', { class: 'btnrow', id: 'delete-confirm-row', role: 'alertdialog', 'aria-label': 'Confirm delete' }, confirmText, yes, no);
  confirmRow.hidden = true;

  const sync = (): void => {
    jsonBtn.disabled = csvBtn.disabled = !rec || rec.isEmpty;
    status.textContent = !rec ? 'No session yet.' : rec.isEmpty ? 'Nothing recorded yet in this session.' : `This session: ${rec.estimates.length} estimates, ${rec.events.length} events${rec.summaries.length ? `, ${rec.summaries.length} HRV ${rec.summaries.length === 1 ? 'summary' : 'summaries'}` : ''} (kept in memory only).${rec.truncated ? ' ' + LIMIT_TEXT + ': later rows were not recorded.' : ''}`;
  };
  const stamp = (r: SessionRecorder): string => r.toJSON().started_at.slice(0, 19).replace(/[:T]/g, '-');
  jsonBtn.addEventListener('click', () => { if (rec) { sync(); downloadText(`carelens-session-${stamp(rec)}.json`, JSON.stringify(rec.toJSON(), null, 2), 'application/json'); } });
  csvBtn.addEventListener('click', () => { if (rec) downloadText(`carelens-estimates-${stamp(rec)}.csv`, rec.toCsv(), 'text/csv'); });

  const refresh = async (): Promise<void> => {
    let items: SessionSummary[] = [];
    try { items = await listSummaries(); } catch { /* storage unavailable */ }
    list.replaceChildren(...items.map((s) => el('li', {}, summaryLine(s))));
    empty.hidden = items.length > 0;
  };
  del.addEventListener('click', () => { confirmRow.hidden = false; del.hidden = true; });
  no.addEventListener('click', () => { confirmRow.hidden = true; del.hidden = false; });
  yes.addEventListener('click', async () => {
    try { await deleteAllSavedData(); } catch { /* ignore */ }
    save.checked = false;
    confirmRow.hidden = true;
    del.hidden = false;
    await refresh();
    status.textContent = 'All saved data deleted.';
  });

  const root = el(
    'section', { class: 'panel', id: 'your-data', 'aria-labelledby': 'h-data' },
    el('h2', { id: 'h-data' }, 'Your data'),
    el('p', { class: 'note' }, 'Estimates and check-in events stay in this page’s memory. You can download them as a file; nothing is uploaded, and no video or sound is included.'),
    el('div', { class: 'btnrow' }, jsonBtn, csvBtn),
    status,
    el('label', { class: 'toggle' }, save, el('span', {}, 'Save a summary of each session on this device (off by default)')),
    el('h3', {}, 'Saved summaries'), empty, list,
    del, confirmRow,
  );
  sync();
  void refresh();
  setInterval(sync, 1000); // the recorder fills in the background: keep the buttons and counts current
  return {
    root,
    setRecorder(r) { rec = r; sync(); },
    async sessionEnded(r) {
      r.finish();
      sync();
      if (save.checked && !r.isEmpty) {
        try { await saveSummary(r.summary()); } catch { status.textContent = 'Could not save on this device (storage unavailable).'; }
        await refresh();
      }
    },
    refresh,
  };
}
