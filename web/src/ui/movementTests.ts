// "Movement tests" (Experimental, W7): 30 s chair stand, five-times sit-to-stand, Timed Up and Go, walking speed, from pose
// landmarks on this device. Flow per test: safety screen with an explicit "I'm ready" -> placement guidance with a live
// visibility check -> 3-2-1 countdown -> live counter/timer -> result card. Wellbeing trends only: no norms, no
// interpretation, no pass/fail. The height (walking speed scale) lives in memory unless the user opted into saving.
import params from '../../../shared/params.json';
import {
  RepMachine, blankSample, chairStand30s, sampleFromLandmarks, sitToStand5x, tug, walkingSpeed, PHASES,
  type MobilityParams, type MobilityResult, type MobilityTest, type Sample,
} from '../core/mobility';
import type { Landmarks } from '../core/features';
import type { PoseView } from '../capture/poseSession';
import { mobilitySummary, type MobilitySummaryFields } from '../export/mobility';
import { safeGet, safeSet } from '../storage';
import { el } from './dom';

export const SAFETY_TEXT = "Only do this if you can do it safely. Use a sturdy chair without wheels against a wall, have someone nearby, stop if you feel unwell or unsteady.";
export const TREND_TEXT = 'A wellbeing trend. Compare with your own earlier results, not with other people.';
export const PLACEMENT_TEXT = 'Place the camera so it sees you from the side, 2–3 m away, with your whole body in view (hips, knees and ankles).';
export const WALK_PLACEMENT_TEXT = " Turns and walking speed are detected from sideways movement, so walk across the camera's view (side view), not towards it.";
export const MOVEMENT_NOTE = 'Experimental. Rough trends from body landmarks; results depend on camera position and light. Not a medical test.';

export const TEST_INFO: Record<MobilityTest, { title: string; blurb: string; live: string }> = {
  chair_stand_30s: { title: '30-second chair stand', blurb: 'Stand up and sit down as many times as you can in 30 seconds, arms folded.', live: 'Stand up fully, then sit down again. Keep going until time is up.' },
  sit_to_stand_5x: { title: 'Five sit-to-stands', blurb: 'Stand up and sit down five times, as quickly as is comfortable.', live: 'Stand up fully five times.' },
  tug: { title: 'Timed Up and Go', blurb: 'Stand up, walk about 3 m, turn, walk back and sit down.', live: 'Stand, walk away, turn, come back and sit down.' },
  walking_speed: { title: 'Walking speed', blurb: 'Walk across the view at your normal pace. Your height scales the result.', live: 'Walk across the view at your normal pace.' },
};
const ORDER: MobilityTest[] = ['chair_stand_30s', 'sit_to_stand_5x', 'tug', 'walking_speed'];
const PHASE_LABEL: Record<(typeof PHASES)[number], string> = { rise: 'Stand up', walk_out: 'Walk away', turn_1: 'Turn', walk_back: 'Walk back', turn_2: 'Turn to sit', sit: 'Sit down' };

export function reasonText(reason: string): string {
  switch (reason) {
    case 'out_of_view': return 'Your legs were not in view often enough. Step back so hips, knees and ankles are visible, then try again.';
    case 'not_seated_at_start': return 'The test starts seated. Sit on the chair before the countdown ends, then try again.';
    case 'no_rise': return 'No stand-up was seen. Try again.';
    case 'no_turn': return 'No turn was seen. Walk away from the chair, turn around, and come back across the view.';
    case 'not_seated_at_end': return 'The test ends when you sit down again. Try again and sit at the end.';
    case 'incomplete': return 'Not all stands were seen in time. Try again.';
    case 'not_side_on': return 'The walk was not side-on to the camera. Walk across the view rather than towards the camera.';
    case 'no_walk': return 'No walking was seen. Walk across the view.';
    case 'too_short': return 'The walk was too short to measure. Walk a little further across the view.';
    case 'bad_height': return 'Enter your height in centimetres to scale the walk.';
    default: return 'No result this time. Try again.';
  }
}

export interface ResultText { headline: string; details: string[]; withheld: boolean }

const n1 = (x: number): string => x.toFixed(1);
const n2 = (x: number): string => x.toFixed(2);

/** Plain numbers for the result card (rounding happens here, not in the core). */
export function resultText(r: MobilityResult, mp: MobilityParams): ResultText {
  if (!r.reported) return { headline: 'No result', details: [reasonText(r.reason)], withheld: true };
  const q = `Legs in view in ${Math.round(r.quality * 100)} % of frames.`;
  switch (r.test) {
    case 'chair_stand_30s': {
      const d = [q];
      if (r.rep_in_progress) d.push('A last stand was still under way when time ran out and is not counted.');
      return { headline: `${r.reps} full stands in ${mp.chair_stand_s} s`, details: d, withheld: false };
    }
    case 'sit_to_stand_5x': return { headline: `${mp.sts_reps} sit-to-stands took ${n1(r.time_s!)} s`, details: [`Per stand: ${r.rep_times_s!.map(n1).join(', ')} s.`, q], withheld: false };
    case 'tug': return { headline: `Timed Up and Go: ${n1(r.time_s!)} s`, details: [PHASES.map((p) => `${PHASE_LABEL[p]} ${n1(r.phases_s![p])} s`).join(' · '), q], withheld: false };
    case 'walking_speed': {
      const d = [`Approximate: depends on the height you entered and a side-on view; perspective is not corrected.`, q];
      if (r.cadence_spm !== null) d.unshift(`About ${Math.round(r.cadence_spm)} steps per minute.`);
      return { headline: `About ${n2(r.speed_m_s!)} m/s (approx.)`, details: d, withheld: false };
    }
  }
}

export interface Placement { hips: boolean; knees: boolean; ankles: boolean; ok: boolean }

/** Live check: is at least one hip, knee and ankle visible? */
export function placementStatus(lm: Landmarks | null, mp: MobilityParams): Placement {
  const seen = (idx: number[]): boolean => lm !== null && idx.some((i) => lm[i]![2]! >= mp.min_visibility);
  const hips = seen([23, 24]), knees = seen([25, 26]), ankles = seen([27, 28]);
  return { hips, knees, ankles, ok: hips && knees && ankles };
}

/** Live progress of one test, DOM-free. The live machine sees exactly the frames the core test function uses (chair stand:
 *  from Go; the others: from `sts_early_s` before Go), so the live count and the final result always agree. */
export class LiveProgress {
  readonly machine: RepMachine;
  private seatsSeen = 0;
  private okCount = 0;
  private lastOkT = -Infinity;

  constructor(readonly test: MobilityTest, private readonly mp: MobilityParams, private readonly heightM: () => number) {
    this.machine = new RepMachine(mp);
  }

  private accepts(s: Sample, goT: number): boolean {
    if (this.test === 'walking_speed') return false;
    return s.t >= goT - (this.test === 'chair_stand_30s' ? 0 : this.mp.sts_early_s);
  }

  /** Feed buffered frames from before Go (the clock origin is approximated by `now`). */
  prime(samples: Sample[], now: number): void {
    for (const s of samples) if (this.test !== 'chair_stand_30s' && this.accepts(s, now)) this.machine.update(s);
  }

  /** One frame (already appended to `samples`). Returns the finished result when the test is over, else null. */
  step(samples: Sample[], s: Sample, goT: number): MobilityResult | null {
    const mp = this.mp, el0 = s.t - goT;
    if (s.ok) { this.okCount++; this.lastOkT = s.t; }
    if (this.accepts(s, goT)) this.machine.update(s);
    switch (this.test) {
      case 'chair_stand_30s': return el0 >= mp.chair_stand_s ? chairStand30s(samples, goT, mp) : null;
      case 'sit_to_stand_5x': return this.machine.stands.length >= mp.sts_reps || el0 >= mp.sts_max_s ? sitToStand5x(samples, goT, mp) : null;
      case 'tug': {
        // a sit only ends the test when a turn came before it (a stand-up, wobble and sit-down is not the walk)
        if (this.machine.seats.length > this.seatsSeen) {
          this.seatsSeen = this.machine.seats.length;
          const r = tug(samples, goT, mp);
          if (r.reported) return r;
        }
        return el0 >= mp.tug_max_s ? tug(samples, goT, mp) : null;
      }
      case 'walking_speed': {
        const left = this.okCount >= mp.walk.min_samples && s.t - this.lastOkT >= 2 * mp.walk.gap_s; // walked across and left the view
        return el0 >= mp.walk_window_s || left ? walkingSpeed(samples, this.heightM(), mp) : null;
      }
    }
  }

  elapsedText(goT: number, t: number): string {
    const el0 = t - goT;
    const left = this.test === 'chair_stand_30s' ? this.mp.chair_stand_s - el0 : this.test === 'walking_speed' ? this.mp.walk_window_s - el0 : el0;
    return `${Math.max(0, left).toFixed(1)} s`;
  }

  countText(): string {
    if (this.test === 'chair_stand_30s') return `Stands: ${this.machine.seats.length}`;
    if (this.test === 'sit_to_stand_5x') return `Stands: ${Math.min(this.machine.stands.length, this.mp.sts_reps)} / ${this.mp.sts_reps}`;
    return '';
  }
}

export interface MovementDeps {
  mp: MobilityParams;
  record(row: MobilitySummaryFields): void;
  /** true when the user opted into saving summaries on this device (the height may then be remembered) */
  heightSaving(): boolean;
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (h: unknown) => void;
}

export interface MovementTests {
  root: HTMLElement;
  /** feed every pose frame (PoseView.state.landmarks are tracker-smoothed, frame-height units) */
  onView(v: PoseView): void;
  /** the camera is off: the card shows why it cannot start */
  disable(text: string): void;
}

type Step = 'pick' | 'safety' | 'placement' | 'countdown' | 'running' | 'result';
const DEFAULT_HEIGHT_CM = 165;

export function buildMovementTests(deps: MovementDeps): MovementTests {
  const mp = deps.mp;
  const setT = deps.setTimeout ?? ((fn, ms) => window.setTimeout(fn, ms));
  const clearT = deps.clearTimeout ?? ((h) => window.clearTimeout(h as number));
  let step: Step = 'pick';
  let test: MobilityTest = 'chair_stand_30s';
  let samples: Sample[] = [];
  let liveP = new LiveProgress(test, mp, () => heightCm / 100);
  let goT: number | null = null;
  let goPending = false;
  let lastT = 0;
  let timer: unknown = null;
  let heightCm = Number(safeGet('height_cm')) || DEFAULT_HEIGHT_CM;
  let lastLm: Landmarks | null = null;
  let disabled: string | null = null;

  const pick = el('div', { class: 'mv-pick', id: 'mv-pick' });
  const pickBtns = ORDER.map((t) => {
    const b = el('button', { type: 'button', class: 'btn', id: `mv-pick-${t}` }, TEST_INFO[t].title);
    b.addEventListener('click', () => { test = t; go('safety'); });
    return b;
  });
  pick.append(el('p', { class: 'note' }, 'Pick a test. Each one starts with a safety check.'), el('div', { class: 'btnrow' }, ...pickBtns));
  const disabledNote = el('p', { class: 'note', id: 'mv-disabled' });
  disabledNote.hidden = true;

  const safetyTitle = el('h4', { id: 'mv-safety-title' });
  const blurb = el('p', { class: 'note', id: 'mv-blurb' });
  const ready = el('button', { type: 'button', class: 'btn primary', id: 'mv-ready' }, "I'm ready");
  const safetyBack = el('button', { type: 'button', class: 'btn ghost', id: 'mv-safety-back' }, 'Back');
  const safety = el('div', { class: 'mv-step', id: 'mv-safety', role: 'group', 'aria-labelledby': 'mv-safety-title' },
    safetyTitle, el('p', { class: 'guidance', id: 'mv-safety-text' }, SAFETY_TEXT), blurb,
    el('div', { class: 'btnrow' }, ready, safetyBack));
  ready.addEventListener('click', () => go('placement'));
  safetyBack.addEventListener('click', () => go('pick'));

  const heightInput = el('input', { type: 'number', id: 'mv-height', min: String(Math.round(mp.walk.height_min_m * 100)), max: String(Math.round(mp.walk.height_max_m * 100)), step: '1', inputmode: 'numeric' });
  heightInput.value = String(heightCm);
  const heightRow = el('label', { class: 'field', id: 'mv-height-row' }, el('span', {}, 'Your height (cm), used only to scale the walk. Kept in memory unless you chose to save summaries.'), heightInput);
  const placementText = el('p', { class: 'guidance', id: 'mv-placement-text' });
  const checks = { hips: el('li', { id: 'mv-chk-hips' }), knees: el('li', { id: 'mv-chk-knees' }), ankles: el('li', { id: 'mv-chk-ankles' }) };
  const start = el('button', { type: 'button', class: 'btn primary', id: 'mv-start' }, 'Start countdown');
  const placeBack = el('button', { type: 'button', class: 'btn ghost', id: 'mv-place-back' }, 'Back');
  const placement = el('div', { class: 'mv-step', id: 'mv-placement' }, placementText,
    el('ul', { class: 'mv-checks', 'aria-live': 'polite' }, checks.hips, checks.knees, checks.ankles), heightRow, el('div', { class: 'btnrow' }, start, placeBack));
  placeBack.addEventListener('click', () => go('safety'));
  start.addEventListener('click', () => go('countdown'));

  const heightValid = (): boolean => { const m = heightCm / 100; return Number.isFinite(m) && m >= mp.walk.height_min_m && m <= mp.walk.height_max_m; };
  heightInput.addEventListener('input', () => {
    heightCm = Number(heightInput.value);
    if (deps.heightSaving() && heightValid()) safeSet('height_cm', String(heightCm));
    updatePlacement(lastLm);
  });

  const count = el('p', { class: 'value', id: 'mv-countdown', role: 'status', 'aria-live': 'assertive' });
  const countdown = el('div', { class: 'mv-step', id: 'mv-count' }, el('p', { class: 'note' }, 'Get into position. The test starts at Go.'), count);

  const liveInfo = el('p', { class: 'guidance', id: 'mv-live-text' });
  const liveTimer = el('p', { class: 'value', id: 'mv-timer' });
  const liveCount = el('p', { class: 'value', id: 'mv-reps' });
  const finish = el('button', { type: 'button', class: 'btn', id: 'mv-finish' }, 'Finish now');
  const cancel = el('button', { type: 'button', class: 'btn ghost', id: 'mv-cancel' }, 'Stop');
  const live = el('div', { class: 'mv-step', id: 'mv-live' }, liveInfo, liveTimer, liveCount, el('div', { class: 'btnrow' }, finish, cancel));
  finish.addEventListener('click', () => complete());
  cancel.addEventListener('click', () => go('pick'));

  const resTitle = el('h4', { id: 'mv-result-title' });
  const resValue = el('p', { class: 'value', id: 'mv-result-value' });
  const resDetail = el('div', { id: 'mv-result-detail' });
  const again = el('button', { type: 'button', class: 'btn primary', id: 'mv-again' }, 'Try again');
  const other = el('button', { type: 'button', class: 'btn', id: 'mv-other' }, 'Another test');
  const result = el('div', { class: 'mv-step', id: 'mv-result', role: 'group', 'aria-labelledby': 'mv-result-title' }, resTitle, resValue, resDetail,
    el('p', { class: 'note', id: 'mv-trend' }, TREND_TEXT), el('div', { class: 'btnrow' }, again, other));
  again.addEventListener('click', () => go('safety'));
  other.addEventListener('click', () => go('pick'));

  const steps: Record<Step, HTMLElement> = { pick, safety, placement, countdown, running: live, result };
  const root = el('article', { class: 'card', id: 'movement-tests' },
    el('div', { class: 'card-head' }, el('h3', {}, 'Movement tests'), el('span', { class: 'tag exp' }, 'Experimental')),
    disabledNote, pick, safety, placement, countdown, live, result, el('p', { class: 'note' }, MOVEMENT_NOTE));

  function clearTimer(): void { if (timer !== null) { clearT(timer); timer = null; } }

  function go(next: Step): void {
    clearTimer();
    step = next;
    for (const [k, node] of Object.entries(steps)) node.hidden = k !== next;
    if (next === 'pick') { samples = []; goT = null; goPending = false; }
    if (next === 'safety') { safetyTitle.textContent = TEST_INFO[test].title; blurb.textContent = TEST_INFO[test].blurb; }
    if (next === 'placement') {
      samples = [];
      placementText.textContent = PLACEMENT_TEXT + (test === 'tug' || test === 'walking_speed' ? WALK_PLACEMENT_TEXT : '');
      heightRow.hidden = test !== 'walking_speed';
      updatePlacement(lastLm);
    }
    if (next === 'countdown') runCountdown(3);
    if (next === 'running') {
      liveP = new LiveProgress(test, mp, () => heightCm / 100);
      goT = null;
      goPending = true;
      liveP.prime(samples, lastT);
      liveInfo.textContent = TEST_INFO[test].live;
      liveTimer.textContent = '';
      liveCount.textContent = liveP.countText();
      finish.hidden = test === 'chair_stand_30s' || test === 'sit_to_stand_5x';
    }
  }

  function runCountdown(n: number): void {
    count.textContent = n > 0 ? String(n) : 'Go!';
    if (n === 0) { go('running'); return; }
    timer = setT(() => runCountdown(n - 1), 1000);
  }

  function updatePlacement(lm: Landmarks | null): void {
    const p = placementStatus(lm, mp);
    const row = (li: HTMLElement, name: string, ok: boolean): void => { li.textContent = `${name}: ${ok ? 'in view' : 'not in view yet'}`; li.className = ok ? 'ok' : 'warn'; };
    row(checks.hips, 'Hips', p.hips); row(checks.knees, 'Knees', p.knees); row(checks.ankles, 'Ankles', p.ankles);
    start.disabled = !p.ok || (test === 'walking_speed' && !heightValid());
  }

  function compute(): MobilityResult {
    const go0 = goT ?? lastT;
    switch (test) {
      case 'chair_stand_30s': return chairStand30s(samples, go0, mp);
      case 'sit_to_stand_5x': return sitToStand5x(samples, go0, mp);
      case 'tug': return tug(samples, go0, mp);
      case 'walking_speed': return walkingSpeed(samples, heightCm / 100, mp);
    }
  }

  function complete(r: MobilityResult = compute()): void {
    if (step !== 'running') return;
    const t = resultText(r, mp);
    resTitle.textContent = TEST_INFO[test].title;
    resValue.textContent = t.headline;
    resValue.classList.toggle('muted', t.withheld);
    resValue.setAttribute('data-reported', String(r.reported));
    if (r.test === 'chair_stand_30s' && r.reported) resValue.setAttribute('data-reps', String(r.reps)); else resValue.removeAttribute('data-reps');
    resDetail.replaceChildren(...t.details.map((d) => el('p', { class: 'note' }, d)));
    deps.record(mobilitySummary(r));
    go('result');
  }

  function onView(v: PoseView): void {
    lastT = v.t;
    const lm = !v.held && v.state.present ? v.state.landmarks : null;
    if (lm) lastLm = lm; else if (!v.held) lastLm = null;
    if (disabled !== null) return;
    if (step === 'placement') { updatePlacement(lm ?? (v.held ? lastLm : null)); return; }
    if (step !== 'countdown' && step !== 'running') return;
    const s = v.held ? blankSample(v.t, true) : lm ? sampleFromLandmarks(v.t, lm, 1, mp) : blankSample(v.t);
    samples.push(s);
    if (step !== 'running') return;
    if (goPending) { goT = v.t; goPending = false; }
    const r = liveP.step(samples, s, goT!);
    liveTimer.textContent = liveP.elapsedText(goT!, v.t);
    liveCount.textContent = liveP.countText();
    if (r) complete(r);
  }

  go('pick');
  return {
    root,
    onView,
    disable(text) {
      disabled = text;
      disabledNote.textContent = text;
      disabledNote.hidden = false;
      for (const b of pickBtns) b.disabled = true;
      go('pick'); // abandon any test in progress: no result is recorded
    },
  };
}

export const MOVEMENT_PARAMS = params.mobility as unknown as MobilityParams;
