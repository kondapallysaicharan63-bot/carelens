// W7 mobility tests: TS port checks on the shared synthetic fixtures (bit-level parity is in test/parity), withheld reasons,
// incremental tracker, summary export mapping and UI wording. Synthetic data only. Not a medical device.
import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import params from '../../shared/params.json';
import {
  PHASES, RepMachine, SampleTracker, blankSample, sampleFromLandmarks, assertMobilityParams, chairStand30s, mobilityErrors, runMobilityFixture, sitToStand5x, trackSamples, tug, walkingSpeed,
  type MobilityParams, type MobilityResult,
} from '../src/core/mobility';
import type { Sample } from '../src/core/mobility';
import { expandLandmarks } from '../src/core/pipeline';
import { compareOutputs, type Fixture } from '../src/core/parity';
import type { PoseView } from '../src/capture/poseSession';
import { mobilitySummary } from '../src/export/mobility';
import { SessionRecorder } from '../src/export/recorder';
import { validateSession } from '../src/export/schema';
import { LiveProgress, PLACEMENT_TEXT, SAFETY_TEXT, WALK_PLACEMENT_TEXT, buildMovementTests, TREND_TEXT, placementStatus, reasonText, resultText } from '../src/ui/movementTests';

const MP: MobilityParams = params.mobility;
const fx = (name: string): any => JSON.parse(readFileSync(`../shared/fixtures/${name}.json`, 'utf8'));
const run = (input: any): any => runMobilityFixture(input, params as any);

describe('mobility params', () => {
  it('validate and reject inconsistent sets (mirror of carelens.params.mobility_errors)', () => {
    expect(mobilityErrors(MP)).toEqual([]);
    for (const bad of [{ ...MP, rise_deg: 100 }, { ...MP, stand_deg: 90 }, { ...MP, stand_hip_frac: 1.5 }, { ...MP, sts_reps: 1 }, { ...MP, chair_stand_s: 0 },
      { ...MP, walk: { ...MP.walk, height_min_m: 3 } }, { ...MP, tug: { ...MP.tug, turn_speed_frac: 1 } }]) {
      expect(mobilityErrors(bad)).not.toEqual([]);
      expect(() => assertMobilityParams(bad)).toThrow();
    }
  });
});

describe('chair stand', () => {
  it.each([[8, 'mobility_chair_stand_8'], [12, 'mobility_chair_stand_12'], [15, 'mobility_chair_stand_15'], [12, 'mobility_chair_stand_dropouts']])('counts %i full reps, not the 2 partial rises (%s)', (n, name) => {
    const r = run(fx(name).input);
    expect(r.reported).toBe(true);
    expect(r.reps).toBe(n);
    expect(r.partial_rises).toBe(2);
    expect(r.rep_times_s).toHaveLength(n);
  });

  it.each([[2, 7.5], [3, 5]])('counts every rep when the pose rate drops (every %ith frame, %f Hz) or timing is irregular', (step, _hz) => {
    const inp = fx('mobility_chair_stand_8').input;
    const sub = (keep: (i: number) => boolean) => {
      const idx = inp.landmarks.map((_: unknown, i: number) => i).filter(keep);
      return { ...inp, t: idx.map((i: number) => i / inp.fps), landmarks: idx.map((i: number) => inp.landmarks[i]) };
    };
    expect(run(sub((i) => i % step === 0)).reps).toBe(8);
    expect(run(sub((i) => i === 0 || (i % step === 0) !== (i % 7 === 3))).reps).toBeGreaterThanOrEqual(7); // irregular: some frames missing, some extra
  });

  it('withholds when no frames fall in the window', () => {
    const inp = fx('mobility_chair_stand_8').input;
    expect(run({ ...inp, go_t: 500 })).toMatchObject({ reported: false, reason: 'out_of_view', reps: null });
  });
});

describe('five-times sit-to-stand, TUG and walking', () => {
  it('sts: reports the time with five per-rep durations that add up', () => {
    const r = run(fx('mobility_sts_5x').input);
    expect(r.reported).toBe(true);
    expect(r.stands).toBe(5);
    expect(r.rep_times_s).toHaveLength(5);
    expect(r.rep_times_s.reduce((a: number, b: number) => a + b, 0)).toBeCloseTo(r.time_s, 9);
  });

  it('tug: phases sum to the total; missing turn is withheld with its reason and no numbers', () => {
    const r = run(fx('mobility_tug').input);
    expect(r.reported).toBe(true);
    expect(Object.keys(r.phases_s)).toEqual([...PHASES]);
    expect((Object.values(r.phases_s) as number[]).reduce((a, b) => a + b, 0)).toBeCloseTo(r.time_s, 9);
    expect(run(fx('mobility_tug_no_turn').input)).toMatchObject({ reported: false, reason: 'no_turn', time_s: null, phases_s: null });
  });

  it.each([['0p6', 0.6], ['1p0', 1.0], ['1p4', 1.4]])('walking speed %s within 10 %% of %f m/s', (tag, v) => {
    const r = run(fx(`mobility_walk_${tag}`).input);
    expect(r.reported).toBe(true);
    expect(Math.abs(r.speed_m_s - v) / v).toBeLessThan(0.1);
    expect(Math.abs(r.cadence_spm - 108)).toBeLessThan(10);
  });

  it('walking towards the camera is not side-on; bad heights are refused; speed scales with height', () => {
    expect(run(fx('mobility_walk_toward_camera').input)).toMatchObject({ reported: false, reason: 'not_side_on', speed_m_s: null });
    const inp = fx('mobility_walk_1p0').input;
    expect(run({ ...inp, height_m: 0.5 }).reason).toBe('bad_height');
    expect(run({ ...inp, height_m: 1.8 }).speed_m_s).toBeCloseTo((run(inp).speed_m_s * 1.8) / 1.65, 6);
  });
});

describe('tracker and machine details', () => {
  it('incremental SampleTracker equals the batch trackSamples', () => {
    const inp = fx('mobility_chair_stand_dropouts').input;
    const frames = inp.landmarks.map((f: number[][] | null) => (f ? expandLandmarks(f, inp.landmark_indices) : null));
    const ts: number[] = frames.map((_: unknown, i: number) => i / inp.fps);
    const batch = trackSamples(ts, frames, inp.aspect, params.tracking, MP);
    const st = new SampleTracker(params.tracking, MP);
    const inc = frames.map((lm: any, i: number) => st.step(ts[i]!, lm, inp.aspect));
    expect(inc).toEqual(batch);
    expect(batch.some((s) => s.held)).toBe(true);
    expect(batch.filter((s) => s.held).every((s) => !s.ok)).toBe(true);
  });

  it('an empty sample series never throws and is withheld', () => {
    expect(chairStand30s([], 0, MP).reason).toBe('out_of_view');
    expect(sitToStand5x([], 0, MP).reason).toBe('out_of_view');
    expect(tug([], 0, MP).reason).toBe('out_of_view');
    expect(walkingSpeed([], 1.65, MP).reason).toBe('out_of_view');
    expect(new RepMachine(MP).inProgress).toBe(false);
  });
});

describe('export mapping and wording', () => {
  const results: MobilityResult[] = [
    run(fx('mobility_chair_stand_8').input), run(fx('mobility_sts_5x').input), run(fx('mobility_tug').input), run(fx('mobility_walk_1p0').input),
    run(fx('mobility_tug_no_turn').input), run(fx('mobility_walk_toward_camera').input),
  ];

  it('every result maps to a valid summaries[] row; withheld rows carry no numbers', () => {
    const rec = new SessionRecorder(['camera'], () => 1_700_000_000_000, 'Chrome/130');
    for (const r of results) rec.addMobilitySummary(mobilitySummary(r));
    const json = rec.toJSON();
    expect(validateSession(json)).toEqual([]);
    expect(json.summaries!.map((s) => s.kind)).toEqual(['chair_stand_30s', 'sit_to_stand_5x', 'tug', 'walking_speed', 'tug', 'walking_speed']);
    expect(json.summaries![0]).toMatchObject({ reported: true, reps: 8 });
    expect(json.summaries![4]).toMatchObject({ reported: false, reason: 'no_turn' });
    expect(json.summaries![4]).not.toHaveProperty('time_s');
    expect(JSON.stringify(json)).not.toMatch(/height/);
  });

  it('a reported row without its headline number is invalid', () => {
    const rec = new SessionRecorder(['camera']);
    const row = rec.addMobilitySummary({ kind: 'tug', reported: true, reason: 'ok' })!;
    expect(validateSession(rec.toJSON()).join()).toContain('time_s');
    row.time_s = 11;
    expect(validateSession(rec.toJSON())).toEqual([]);
  });

  it('uses wellbeing-trend wording only: no norms, risk scores or diagnosis', () => {
    expect(SAFETY_TEXT).toBe('Only do this if you can do it safely. Use a sturdy chair without wheels against a wall, have someone nearby, stop if you feel unwell or unsteady.');
    expect(TREND_TEXT).toBe('A wellbeing trend. Compare with your own earlier results, not with other people.');
    const texts = [SAFETY_TEXT, TREND_TEXT, PLACEMENT_TEXT, ...results.flatMap((r) => { const t = resultText(r, MP); return [t.headline, ...t.details]; }),
      ...['out_of_view', 'not_seated_at_start', 'no_rise', 'no_turn', 'not_seated_at_end', 'incomplete', 'not_side_on', 'no_walk', 'too_short', 'bad_height'].map(reasonText)];
    for (const t of texts) expect(t).not.toMatch(/fall risk|risk of|diagnos|normal|abnormal|below average|above average|pass|fail|frail|sarcopenia|healthy range/i);
    expect(resultText(results[0]!, MP).headline).toBe('8 full stands in 30 s');
    expect(resultText(results[3]!, MP).headline).toMatch(/^About \d\.\d\d m\/s \(approx\.\)$/);
  });

  it('placement check needs a hip, knee and ankle in view', () => {
    const lm = (vis: number) => Array.from({ length: 33 }, () => [0.5, 0.5, vis]);
    expect(placementStatus(null, MP).ok).toBe(false);
    expect(placementStatus(lm(0.9), MP)).toEqual({ hips: true, knees: true, ankles: true, ok: true });
    const some = lm(0.9); some[27]![2] = 0.1; some[28]![2] = 0.1;
    expect(placementStatus(some, MP)).toMatchObject({ ankles: false, ok: false });
  });
});


describe('review fixes', () => {
  it('parity tolerances are scoped per fixture family: 80 ms cough onset fails, mobility times allow 0.1 s', () => {
    const cough: Fixture = fx('audio_cough_single');
    const e = JSON.parse(JSON.stringify(cough.expected));
    expect(compareOutputs(cough, e)).toEqual([]);
    e.events[0].t_start += 0.08;
    expect(compareOutputs(cough, e).length).toBeGreaterThan(0);
    const hrv: Fixture = fx('hrv_clean_30ms');
    const h = JSON.parse(JSON.stringify(hrv.expected));
    h.hrv.duration_s += 0.08;
    expect(compareOutputs(hrv, h).length).toBeGreaterThan(0);
    const mob: Fixture = fx('mobility_sts_5x');
    const m = JSON.parse(JSON.stringify(mob.expected));
    m.time_s += 0.08;
    expect(compareOutputs(mob, m)).toEqual([]);
    m.time_s += 0.08;
    expect(compareOutputs(mob, m).length).toBeGreaterThan(0);
  });

  it('walking: a single 3 s crossing between absent frames is reported (quality over the in-view span)', () => {
    const r = run(fx('mobility_walk_crossing_3s').input);
    expect(r.reported).toBe(true);
    expect(Math.abs(r.speed_m_s - 1.0)).toBeLessThan(0.1);
    expect(r.quality).toBeGreaterThan(0.9);
  });

  it('sts_reps is bounded to [2, 50] and the 5x per-rep list is capped like the chair stand', () => {
    expect(mobilityErrors({ ...MP, sts_reps: 51 })).not.toEqual([]);
    expect(mobilityErrors({ ...MP, sts_reps: 50 })).toEqual([]);
    const row = mobilitySummary({ test: 'sit_to_stand_5x', reported: true, reason: 'ok', time_s: 9, rep_times_s: Array(150).fill(1), stands: 150, quality: 1 });
    expect(row.rep_times_s).toHaveLength(100);
  });

  it('placement text tells the user turns come from sideways movement (side view, not towards the camera)', () => {
    expect(WALK_PLACEMENT_TEXT).toMatch(/sideways movement/);
    expect(WALK_PLACEMENT_TEXT).toMatch(/\(side view\), not towards it/);
  });
});

// ---- live progress and the UI flow (a minimal fake DOM: the project has no jsdom)
class FakeEl {
  attrs: Record<string, string> = {}; kids: (FakeEl | string)[] = []; listeners: Record<string, (() => void)[]> = {};
  hidden = false; disabled = false; className = ''; value = ''; private text = '';
  constructor(readonly tag: string, private readonly reg: Map<string, FakeEl>) {}
  setAttribute(k: string, v: string): void { this.attrs[k] = v; if (k === 'id') this.reg.set(v, this); }
  getAttribute(k: string): string | null { return this.attrs[k] ?? null; }
  removeAttribute(k: string): void { delete this.attrs[k]; }
  append(...c: (FakeEl | string)[]): void { this.kids.push(...c); }
  replaceChildren(...c: (FakeEl | string)[]): void { this.kids = c; this.text = ''; }
  addEventListener(ev: string, fn: () => void): void { (this.listeners[ev] ??= []).push(fn); }
  click(): void { if (!this.disabled) (this.listeners.click ?? []).forEach((f) => f()); }
  get classList() { return { toggle: () => undefined, remove: () => undefined, add: () => undefined }; }
  get textContent(): string { return this.text + this.kids.map((k) => (typeof k === 'string' ? k : k.textContent)).join(''); }
  set textContent(v: string) { this.text = v; this.kids = []; }
}
const reg = new Map<string, FakeEl>();
const $ = (id: string): FakeEl => reg.get(id)!;

describe('movement tests flow', () => {
  beforeEach(() => { reg.clear(); (globalThis as any).document = { createElement: (t: string) => new FakeEl(t, reg) }; });
  afterEach(() => { delete (globalThis as any).document; });

  const frames = (name: string) => {
    const inp = fx(name).input;
    return (inp.landmarks as (number[][] | null)[]).map((f, i) => {
      const lm = f ? expandLandmarks(f, inp.landmark_indices).map((r) => [r[0]! * inp.aspect, r[1]!, r[2]!]) : null;
      return { t: i / inp.fps, lm };
    });
  };
  const view = (t: number, lm: number[][] | null): PoseView => ({ t, held: false, state: { present: lm !== null, landmarks: lm } as any, breathing: { status: 'ok', estimate: null } as any });

  function setup() {
    const rows: any[] = [];
    const queue: (() => void)[] = [];
    const ui = buildMovementTests({ mp: MP, record: (r) => rows.push(r), heightSaving: () => false, setTimeout: (fn) => queue.push(fn), clearTimeout: () => { queue.length = 0; } });
    return { ui, rows, queue };
  }

  it('chair stand: live count equals the final result, which is recorded once', () => {
    const { ui, rows, queue } = setup();
    const fr = frames('mobility_chair_stand_8');
    $('mv-pick-chair_stand_30s').click();
    expect($('mv-safety-text').textContent).toBe(SAFETY_TEXT);
    $('mv-ready').click();
    expect($('mv-start').disabled).toBe(true); // nothing in view yet
    for (const f of fr.slice(0, 5)) ui.onView(view(f.t, f.lm));
    expect($('mv-start').disabled).toBe(false);
    $('mv-start').click();
    while (queue.length) queue.shift()!();
    expect($('mv-live').hidden).toBe(false);
    let lastLive = '';
    for (const f of fr) { ui.onView(view(f.t, f.lm)); if (!$('mv-result').hidden) break; lastLive = $('mv-reps').textContent; }
    expect($('mv-result').hidden).toBe(false);
    const reps = Number($('mv-result-value').getAttribute('data-reps'));
    const samples = fr.map((f) => (f.lm ? sampleFromLandmarks(f.t, f.lm, 1, MP) : blankSample(f.t)));
    expect(reps).toBe(chairStand30s(samples, 0, MP).reps);
    expect(reps).toBe(8);
    expect(lastLive).toBe(`Stands: ${reps}`);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ kind: 'chair_stand_30s', reported: true, reps });
  });

  it('disable() abandons a running test, records nothing and ignores later frames', () => {
    const { ui, rows, queue } = setup();
    const fr = frames('mobility_chair_stand_8');
    $('mv-pick-chair_stand_30s').click(); $('mv-ready').click();
    for (const f of fr.slice(0, 5)) ui.onView(view(f.t, f.lm));
    $('mv-start').click();
    while (queue.length) queue.shift()!();
    for (const f of fr.slice(0, 40)) ui.onView(view(f.t, f.lm));
    ui.disable('The camera is off.');
    expect($('mv-pick').hidden).toBe(false);
    expect($('mv-live').hidden).toBe(true);
    expect($('mv-disabled').hidden).toBe(false);
    expect($('mv-pick-tug').disabled).toBe(true);
    for (const f of fr.slice(40)) ui.onView(view(f.t, f.lm));
    expect(rows).toEqual([]);
    expect($('mv-result').hidden).toBe(true);
  });

  it('live chair stand uses the core window: no frames before Go are fed', () => {
    const lp = new LiveProgress('chair_stand_30s', MP, () => 1.65);
    const fr = frames('mobility_chair_stand_8').map((f) => (f.lm ? sampleFromLandmarks(f.t, f.lm, 1, MP) : blankSample(f.t)));
    lp.prime(fr, 3.0);
    expect(lp.machine.state).toBe('init');
    const buf: typeof fr = [];
    let res = null;
    for (const s of fr.filter((x) => x.t >= 1.0)) { buf.push(s); res = lp.step(buf, s, 1.0); if (res) break; }
    expect(res).toMatchObject({ reported: true, reps: 8 });
    expect(lp.countText()).toBe('Stands: 8');
  });

  it('TUG live: the sit at the far end without a turn does not end the test; the real return and sit does', () => {
    const run1 = (name: string) => {
      const inp = fx(name).input;
      const fr = (inp.landmarks as (number[][] | null)[]).map((f, i) => ({ t: i / inp.fps, lm: f ? expandLandmarks(f, inp.landmark_indices).map((r) => [r[0]! * inp.aspect, r[1]!, r[2]!]) : null }));
      const lp = new LiveProgress('tug', MP, () => 1.65);
      const buf: Sample[] = [];
      let done: any = null, doneAt = -1;
      for (const f of fr) {
        const s = f.lm ? sampleFromLandmarks(f.t, f.lm, 1, MP) : blankSample(f.t);
        buf.push(s);
        const r = lp.step(buf, s, inp.go_t);
        if (r) { done = r; doneAt = f.t; break; }
      }
      return { done, doneAt, seats: lp.machine.seats.length, end: fr[fr.length - 1]!.t };
    };
    const noTurn = run1('mobility_tug_no_turn');
    expect(noTurn.seats).toBeGreaterThanOrEqual(1);   // it did sit at the far end ...
    expect(noTurn.done).toBeNull();                    // ... but that did not end the test
    const ok = run1('mobility_tug');
    expect(ok.done).toMatchObject({ reported: true });
    expect(ok.doneAt).toBeLessThan(ok.end);
  });

  it('walking live ends once the person has walked across and left the view', () => {
    const inp = fx('mobility_walk_crossing_3s').input;
    const lp = new LiveProgress('walking_speed', MP, () => 1.65);
    const buf: Sample[] = [];
    let done: any = null, at = 0;
    (inp.landmarks as (number[][] | null)[]).forEach((f, i) => {
      if (done) return;
      const t = i / inp.fps;
      const s = f ? sampleFromLandmarks(t, expandLandmarks(f, inp.landmark_indices).map((r) => [r[0]! * inp.aspect, r[1]!, r[2]!]), 1, MP) : blankSample(t);
      buf.push(s);
      done = lp.step(buf, s, 0); at = t;
    });
    expect(done).not.toBeNull();
    expect(at).toBeLessThan(MP.walk_window_s);
  });
});
