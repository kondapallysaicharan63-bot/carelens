// "Rhythmic hand movement" test (Experimental, W8): hold a hand out to the camera for 15 s; the hand model loads only for this test and
// is closed afterwards. Flow: instructions + comfort line with an explicit "I'm ready" -> live hand-in-view check -> 3-2-1 countdown ->
// 15 s capture -> result card. Wellbeing trend only: no norms, no interpretation. "No clear rhythmic hand movement" is the expected
// result for most people and is worded neutrally.
import params from '../../../shared/params.json';
import type { HandHandle } from '../capture/hand';
import { HandSession, type FrameSubscriber } from '../capture/handSession';
import { analyseTremor, forceLowFps, type HandFrame, type TremorParams, type TremorResult } from '../core/tremor';
import { tremorSummary } from '../export/tremor';
import type { MobilitySummaryFields } from '../export/mobility';
import { el } from './dom';

export const TREMOR_INSTRUCTIONS = 'Hold your hand out in front of the camera, palm facing the camera, fingers spread, about 40–60 cm away, for 15 seconds. Rest your elbow if you need to.';
export const TREMOR_COMFORT = 'Only do this if it is comfortable. Rest your elbow on a table or armrest, and stop if your arm tires or you feel unwell.';
export const TREMOR_NOTE = 'Experimental. Many things affect this — caffeine, tiredness, cold, holding position. Compare with your own earlier results.';
export const TREMOR_NO_RHYTHM = 'No clear rhythmic hand movement detected';
export const TREMOR_PARAMS = params.tremor as unknown as TremorParams;

export interface TremorText { headline: string; details: string[]; withheld: boolean }

const n1 = (x: number): string => x.toFixed(1);

/** Plain numbers for the result card (rounding happens here, not in the core). */
export function tremorResultText(r: TremorResult, tp: TremorParams): TremorText {
  switch (r.reason) {
    case 'ok': return { headline: `Rhythmic hand movement: ${n1(r.freq_hz!)} Hz, amplitude ${n1(r.amp_pct!)} % of hand size`, details: [], withheld: false };
    case 'no_rhythm': return { headline: TREMOR_NO_RHYTHM, details: [], withheld: false };
    case 'no_hand': return { headline: 'No result', details: ['No hand was seen. Hold your hand out, palm facing the camera and fingers spread, then try again.'], withheld: true };
    case 'too_short': return { headline: 'No result', details: [`Your hand was in view for too short a time (at least ${tp.min_window_s} s is needed). Keep it in view for the whole ${tp.test_s} s and try again.`], withheld: true };
    case 'low_fps': return { headline: 'No result', details: [`This camera sends too few frames per second for this test (about ${Math.round(tp.min_rate_hz)} or more are needed). No value is shown rather than a wrong one.`], withheld: true };
    case 'moving': return { headline: 'No result', details: ['Your hand or arm moved too much. Rest your elbow, hold still and try again.'], withheld: true };
  }
}

export interface TremorDeps {
  tp: TremorParams;
  record(row: MobilitySummaryFields): void;
  subscribe: FrameSubscriber;
  loadHand: () => Promise<HandHandle>;
  getVideo: () => HTMLVideoElement | null;
  /** the camera's declared frame rate (track.getSettings().frameRate), null when unknown */
  getDeclaredFps?: () => number | null;
  /** true while the hand test needs the whole CPU (placement, countdown, capture): the caller pauses the pose and face models and
   *  any movement test, and resumes them on false */
  onActive?(active: boolean): void;
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (h: unknown) => void;
}

export interface TremorTest {
  root: HTMLElement;
  /** the camera is off: the card shows why it cannot start; any test in progress is abandoned (no result recorded) */
  disable(text: string): void;
}

type Step = 'intro' | 'safety' | 'placement' | 'countdown' | 'running' | 'result';
/** a hand counts as "in view" if one of the last frames within this many seconds had it */
const IN_VIEW_WINDOW_S = 0.5;

export function buildTremorTest(deps: TremorDeps): TremorTest {
  const tp = deps.tp;
  const setT = deps.setTimeout ?? ((fn, ms) => window.setTimeout(fn, ms));
  const clearT = deps.clearTimeout ?? ((h) => window.clearTimeout(h as number));
  let step: Step = 'intro';
  let session: HandSession | null = null;
  let frames: HandFrame[] = [];
  let aspect = 4 / 3;
  let lastSeenT = -Infinity;
  let lastT = 0;
  let goT: number | null = null;
  let goPending = false;
  let timer: unknown = null;
  let disabled: string | null = null;
  let busy = false;

  const introBtn = el('button', { type: 'button', class: 'btn', id: 'tm-pick' }, 'Start the hand test');
  const intro = el('div', { class: 'mv-pick', id: 'tm-intro' }, el('p', { class: 'note' }, 'A 15-second test of rhythmic hand movement, using the camera on this device.'), el('div', { class: 'btnrow' }, introBtn));
  const disabledNote = el('p', { class: 'note', id: 'tm-disabled' });
  disabledNote.hidden = true;

  const ready = el('button', { type: 'button', class: 'btn primary', id: 'tm-ready' }, "I'm ready");
  const safetyBack = el('button', { type: 'button', class: 'btn ghost', id: 'tm-safety-back' }, 'Back');
  const safety = el('div', { class: 'mv-step', id: 'tm-safety' },
    el('h4', {}, 'Rhythmic hand movement'), el('p', { class: 'guidance', id: 'tm-instructions' }, TREMOR_INSTRUCTIONS), el('p', { class: 'note', id: 'tm-comfort' }, TREMOR_COMFORT),
    el('p', { class: 'note' }, 'The hand model is loaded for this test only and closed afterwards. Nothing leaves your device.'), el('div', { class: 'btnrow' }, ready, safetyBack));

  const loadNote = el('p', { class: 'note', id: 'tm-load' });
  const hand = el('li', { id: 'tm-chk-hand' });
  const start = el('button', { type: 'button', class: 'btn primary', id: 'tm-start' }, 'Start countdown');
  const placeBack = el('button', { type: 'button', class: 'btn ghost', id: 'tm-place-back' }, 'Back');
  const placement = el('div', { class: 'mv-step', id: 'tm-placement' }, el('p', { class: 'guidance' }, TREMOR_INSTRUCTIONS), loadNote,
    el('ul', { class: 'mv-checks', 'aria-live': 'polite' }, hand), el('div', { class: 'btnrow' }, start, placeBack));

  const count = el('p', { class: 'value', id: 'tm-countdown', role: 'status', 'aria-live': 'assertive' });
  const countdown = el('div', { class: 'mv-step', id: 'tm-count' }, el('p', { class: 'note' }, 'Hold your hand out. The test starts at Go.'), count);

  const liveTimer = el('p', { class: 'value', id: 'tm-timer' });
  const liveHand = el('p', { class: 'note', id: 'tm-live-hand', 'aria-live': 'polite' });
  const finish = el('button', { type: 'button', class: 'btn', id: 'tm-finish' }, 'Finish now');
  const cancel = el('button', { type: 'button', class: 'btn ghost', id: 'tm-cancel' }, 'Stop');
  const live = el('div', { class: 'mv-step', id: 'tm-live' }, el('p', { class: 'guidance' }, 'Hold your hand steady, palm to the camera.'), liveTimer, liveHand, el('div', { class: 'btnrow' }, finish, cancel));

  const resValue = el('p', { class: 'value', id: 'tm-result-value' });
  const resDetail = el('div', { id: 'tm-result-detail' });
  const again = el('button', { type: 'button', class: 'btn primary', id: 'tm-again' }, 'Try again');
  const done = el('button', { type: 'button', class: 'btn', id: 'tm-done' }, 'Done');
  const result = el('div', { class: 'mv-step', id: 'tm-result', role: 'group', 'aria-label': 'Hand test result' }, resValue, resDetail,
    el('p', { class: 'note', id: 'tm-note' }, TREMOR_NOTE), el('div', { class: 'btnrow' }, again, done));

  const steps: Record<Step, HTMLElement> = { intro, safety, placement, countdown, running: live, result };
  const root = el('article', { class: 'card', id: 'tremor-test' },
    el('div', { class: 'card-head' }, el('h3', {}, 'Hand movement'), el('span', { class: 'tag exp' }, 'Experimental')),
    disabledNote, intro, safety, placement, countdown, live, result, el('p', { class: 'note' }, 'Not a medical test. A rough wellbeing trend from hand landmarks.'));

  introBtn.addEventListener('click', () => go('safety'));
  safetyBack.addEventListener('click', () => go('intro'));
  ready.addEventListener('click', () => go('placement'));
  placeBack.addEventListener('click', () => go('safety'));
  start.addEventListener('click', () => go('countdown'));
  finish.addEventListener('click', () => complete());
  cancel.addEventListener('click', () => go('intro'));
  again.addEventListener('click', () => go('safety'));
  done.addEventListener('click', () => go('intro'));

  function clearTimer(): void { if (timer !== null) { clearT(timer); timer = null; } }
  function closeModel(): void { session?.end(); session = null; }

  function go(next: Step): void {
    clearTimer();
    step = next;
    const wantBusy = next === 'placement' || next === 'countdown' || next === 'running';
    if (wantBusy !== busy) { busy = wantBusy; deps.onActive?.(busy); }
    for (const [k, node] of Object.entries(steps)) node.hidden = k !== next;
    if (next === 'intro' || next === 'safety' || next === 'result') closeModel(); // the hand model lives only during the test
    if (next === 'intro' || next === 'safety') { frames = []; goT = null; goPending = false; lastSeenT = -Infinity; }
    if (next === 'placement') { frames = []; lastSeenT = -Infinity; void openModel(); updatePlacement(); }
    if (next === 'countdown') runCountdown(3);
    if (next === 'running') { goT = null; goPending = true; liveTimer.textContent = `${tp.test_s.toFixed(1)} s`; liveHand.textContent = ''; }
  }

  async function openModel(): Promise<void> {
    const video = deps.getVideo();
    if (!video) { loadNote.textContent = 'The camera is not ready.'; return; }
    closeModel();
    if (disabled === null) disabledNote.hidden = true; // a load error from an earlier try is out of date now
    const s = new HandSession(deps.subscribe, onFrame);
    session = s;
    loadNote.textContent = 'Loading the hand model…';
    try {
      await s.load(deps.loadHand, video);
      if (!s.isEnded) loadNote.textContent = '';
    } catch (err) {
      if (s.isEnded) return;
      closeModel();
      go('intro');
      disabledNote.textContent = 'The hand model failed to load: ' + (err instanceof Error ? err.message : String(err));
      disabledNote.hidden = false;
    }
  }

  function runCountdown(n: number): void {
    count.textContent = n > 0 ? String(n) : 'Go!';
    if (n === 0) { go('running'); return; }
    timer = setT(() => runCountdown(n - 1), 1000);
  }

  const inView = (): boolean => lastT - lastSeenT <= IN_VIEW_WINDOW_S;

  function updatePlacement(): void {
    const ok = inView();
    hand.textContent = `Hand: ${ok ? 'in view' : 'not in view yet'}`;
    hand.className = ok ? 'ok' : 'warn';
    start.disabled = !ok || session === null;
  }

  function complete(): void {
    if (step !== 'running') return;
    const t0 = goT ?? lastT;
    let r = analyseTremor(frames.filter((f) => f.t >= t0), aspect, tp);
    // Frames could not be told apart from repeated display callbacks: trust the camera's declared rate, and withhold when unknown.
    if (session && !session.timingReliable) {
      const fps = deps.getDeclaredFps?.() ?? null;
      if (fps === null || !(fps >= tp.min_rate_hz)) r = forceLowFps(r);
    }
    const t = tremorResultText(r, tp);
    resValue.textContent = t.headline;
    resValue.classList.toggle('muted', t.withheld);
    resValue.setAttribute('data-reported', String(r.reported));
    resValue.setAttribute('data-reason', r.reason);
    if (r.reported) resValue.setAttribute('data-freq', r.freq_hz!.toFixed(2)); else resValue.removeAttribute('data-freq');
    resDetail.replaceChildren(...t.details.map((d) => el('p', { class: 'note' }, d)));
    deps.record(tremorSummary(r));
    go('result');
  }

  function onFrame(f: HandFrame, a: number): void {
    lastT = f.t;
    aspect = a;
    if (f.lm !== null && f.score >= tp.min_hand_score) lastSeenT = f.t;
    if (disabled !== null) return;
    if (step === 'placement') { updatePlacement(); return; }
    if (step !== 'countdown' && step !== 'running') return;
    frames.push(f);
    if (step !== 'running') return;
    if (goPending) { goT = f.t; goPending = false; }
    const el0 = f.t - goT!;
    liveTimer.textContent = `${Math.max(0, tp.test_s - el0).toFixed(1)} s`;
    liveHand.textContent = inView() ? 'Hand in view' : 'Hand not in view: bring it back into view';
    if (el0 >= tp.test_s - 1e-9) complete();
  }

  go('intro');
  return {
    root,
    disable(text) {
      disabled = text;
      disabledNote.textContent = text;
      disabledNote.hidden = false;
      introBtn.disabled = true;
      go('intro'); // abandons any test in progress (no result is recorded) and closes the model
    },
  };
}
