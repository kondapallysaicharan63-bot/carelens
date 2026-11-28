import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import params from '../../shared/params.json';
import { BreathCameraLogger } from '../src/export/breathCamera';
import { CoughRecorder } from '../src/export/cough';
import { LabelEventTracker, detectLabelEvents } from '../src/core/audio';
import { CSV_HEADER, MAX_PER_METRIC, SessionRecorder, estimatesCsv, isoWithOffset, uaFamily } from '../src/export/recorder';
import type { PoseView } from '../src/capture/poseSession';
import { clearLocalPrefs } from '../src/export/saved';
import { EVENT_KINDS, METRICS, MODES, SUMMARY_KINDS, UA_FAMILIES, validateSession } from '../src/export/schema';
import { summaryLine } from '../src/ui/exportPanel';

const schema = JSON.parse(readFileSync('../shared/session.schema.json', 'utf8'));
const example = () => JSON.parse(readFileSync('../shared/session.example.json', 'utf8'));

describe('session summaries (HRV)', () => {
  const mod = (f: (s: any) => void) => { const s = JSON.parse(readFileSync('../shared/session.example.json', 'utf8')); f(s); return validateSession(s); };
  it('are optional and validated', () => {
    expect(mod((s) => delete s.summaries)).toEqual([]);
    for (const f of [(r: any) => (r.kind = 'stress'), (r: any) => (r.rmssd_ms = -1), (r: any) => (r.kept_frac = 1.5), (r: any) => (r.n_kept = 2.5),
      (r: any) => (r.foo = 1), (r: any) => delete r.sdnn_ms, (r: any) => delete r.reason, (r: any) => (r.reported = 'yes')]) {
      expect(mod((s) => f(s.summaries[0])), String(f)).not.toEqual([]);
    }
    expect(mod((s) => { Object.assign(s.summaries[0], { reported: false, reason: 'too_short' }); for (const k of ['rmssd_ms', 'sdnn_ms', 'mean_hr_bpm']) delete s.summaries[0][k]; })).toEqual([]);
  });
});

describe('session schema validator', () => {
  it('accepts the shared example and matches the JSON schema file', () => {
    expect(validateSession(example())).toEqual([]);
    expect(schema.$defs.estimate.properties.metric.enum).toEqual([...METRICS]);
    expect(schema.$defs.event.properties.kind.enum).toEqual([...EVENT_KINDS]);
    expect(schema.properties.modes.items.enum).toEqual([...MODES]);
    expect(schema.$defs.summary.properties.kind.enum).toEqual([...SUMMARY_KINDS]);
    expect(Object.keys(schema.$defs.summary.properties).sort()).toEqual(['absent_min', 'active_min', 'amp_pct', 'blinks_per_min', 'cadence_spm', 'duration_s', 'eyes_closed_pct', 'freq_hz', 'kept_frac', 'kind', 'long_closures', 'longest_still_min', 'mean_hr_bpm', 'n_beats', 'n_kept', 'p2p_pct', 'phases_s', 'quality', 'reason', 'rep_times_s', 'reported', 'reps', 'rmssd_ms', 'sdnn_ms', 'speed_m_s', 'still_min', 't', 'time_s', 'wall']);
    expect(schema.properties.device.properties.user_agent_family.enum).toEqual([...UA_FAMILIES]);
  });
  const bad: [string, (s: any) => void, string][] = [
    ['missing field', (s) => delete s.started_at, 'missing "started_at"'],
    ['version', (s) => (s.schema_version = 2), 'schema_version'],
    ['no offset', (s) => (s.started_at = '2026-11-20T10:00:00.000'), 'started_at'],
    ['device label', (s) => (s.device.label = 'Pixel 7'), 'unexpected field "label"'],
    ['device id', (s) => (s.device_id = 'x'), 'unexpected field "device_id"'],
    ['media', (s) => (s.video = 'data:'), 'unexpected field "video"'],
    ['metric', (s) => (s.estimates[0].metric = 'spo2'), 'estimates[0].metric'],
    ['quality', (s) => (s.estimates[0].quality = 2), 'estimates[0].quality'],
    ['value type', (s) => (s.estimates[0].value = '72'), 'estimates[0].value'],
    ['event kind', (s) => (s.events[0].kind = 'diagnosis'), 'events[0].kind'],
    ['not an array', (s) => (s.events = {}), 'events: must be an array'],
  ];
  it.each(bad)('rejects: %s', (_n, mutate, needle) => {
    const s = example();
    mutate(s);
    expect(validateSession(s).join('\n')).toContain(needle);
  });
  it('parity with the Python validator: integers, booleans, reported rows, truncated', () => {
    const ok = (txt: string) => validateSession(JSON.parse(txt));
    const base = JSON.stringify(example());
    expect(ok(base.replace('"schema_version":1', '"schema_version":1.0'))).toEqual([]); // JSON 1.0 is the number 1
    const mod = (f: (s: any) => void) => { const s = example(); f(s); return validateSession(s); };
    expect(mod((s) => { s.params_version = 7.0; s.tz_offset_min = -480.0; })).toEqual([]);
    expect(mod((s) => (s.schema_version = true))).not.toEqual([]);
    expect(mod((s) => (s.params_version = true))).not.toEqual([]);
    expect(mod((s) => (s.tz_offset_min = false))).not.toEqual([]);
    expect(mod((s) => (s.params_version = 7.5))).not.toEqual([]);
    expect(mod((s) => (s.truncated = true))).toEqual([]);
    expect(mod((s) => (s.truncated = 'yes'))).not.toEqual([]);
    expect(mod((s) => { s.estimates[0].reported = true; s.estimates[0].value = null; }).join()).toContain('reported rows need a value');
  });
  it('rejects non-objects without throwing', () => {
    expect(validateSession(null)).toHaveLength(1);
    expect(validateSession([])).toHaveLength(1);
  });
});

describe('SessionRecorder', () => {
  const mk = () => {
    let now = Date.UTC(2026, 10, 20, 18, 0, 0);
    const r = new SessionRecorder(['fingertip'], () => now, 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 HeadlessChrome/140 Safari/537.36');
    return { r, tick: (s: number) => (now += s * 1000) };
  };
  it('records estimates and events and exports schema-valid JSON without device details', () => {
    const { r, tick } = mk();
    tick(12);
    r.addEstimate({ metric: 'hr_fingertip', value: 71.96, quality: 0.8, reported: true, reason: 'ok', span_s: 10 });
    tick(1);
    r.addEstimate({ metric: 'hr_fingertip', value: null, quality: 0, reported: false, reason: 'settling' });
    tick(20);
    r.addEvent({ kind: 'possible_fall', agoS: 21.2, trigger: 'torso_angle' });
    r.addEvent({ kind: 'cough', agoS: 0.5, durationS: 0.5, peak_score: 0.8 });
    const j = r.toJSON();
    expect(validateSession(j)).toEqual([]);
    expect(j.device).toEqual({ user_agent_family: 'Chrome' });
    expect(j.estimates[0]).toMatchObject({ t: 12, metric: 'hr_fingertip', value: 71.96, span_s: 10 });
    expect(j.estimates[1]!.value).toBeNull();
    expect(j.events[0]).toMatchObject({ kind: 'possible_fall', t: 33, t_onset: 11.8 });
    expect(JSON.stringify(j)).not.toMatch(/deviceId|label|userAgent|Mozilla/);
    expect(j.estimates[0]!.wall).toBe(isoWithOffset(r.startedMs + 12000, r.tzOffsetMin));
  });
  it('summary: median of reported values per metric, duration, counts', () => {
    const { r, tick } = mk();
    for (const v of [70, 72, 80]) { tick(1); r.addEstimate({ metric: 'hr_fingertip', value: v, quality: 1, reported: true, reason: 'ok' }); }
    tick(1); r.addEstimate({ metric: 'hr_fingertip', value: 200, quality: 0.1, reported: false, reason: 'low_signal' });
    r.addEstimate({ metric: 'br_sound', value: 15, quality: 0.1, reported: false, reason: 'low_signal' });
    r.finish();
    const s = r.summary();
    expect(s.medians).toEqual({ hr_fingertip: 72, br_sound: null });
    expect(s.duration_s).toBe(4);
    expect(summaryLine(s)).toContain('HR fingertip: 72 bpm (median)');
    expect(summaryLine(s)).toContain('no reported reading');
  });
  it('CSV shape: header, one line per estimate, 8 columns, empty value for null', () => {
    const { r, tick } = mk();
    tick(5); r.addEstimate({ metric: 'br_camera', value: 15.2, quality: 0.5, reported: true, reason: 'ok', span_s: 30 });
    tick(1); r.addEstimate({ metric: 'br_camera', value: null, quality: 0, reported: false, reason: 'settling' });
    const lines = r.toCsv().trimEnd().split('\n');
    expect(lines[0]).toBe(CSV_HEADER);
    expect(lines).toHaveLength(3);
    for (const l of lines) expect(l.split(',')).toHaveLength(8);
    expect(lines[1]!.split(',').slice(2)).toEqual(['br_camera', '15.2', '0.5', 'true', 'ok', '30']);
    expect(lines[2]!.split(',').slice(2)).toEqual(['br_camera', '', '0', 'false', 'settling', '']);
    expect(estimatesCsv([])).toBe(CSV_HEADER + '\n');
  });
  it('caps rows per metric, sets truncated and says so', () => {
    const { r } = mk();
    for (let i = 0; i < MAX_PER_METRIC + 5; i++) r.addEstimate({ metric: 'hr_face', value: 70, quality: 1, reported: true, reason: 'ok' });
    expect(r.estimates).toHaveLength(MAX_PER_METRIC);
    expect(r.truncated).toBe(true);
    r.addEstimate({ metric: 'br_sound', value: 15, quality: 1, reported: true, reason: 'ok' }); // another metric still has room
    expect(r.estimates).toHaveLength(MAX_PER_METRIC + 1);
    const j = r.toJSON();
    expect(j.truncated).toBe(true);
    expect(validateSession(j)).toEqual([]);
    expect(MAX_PER_METRIC).toBeGreaterThanOrEqual(24 * 3600);
    expect(mk().r.toJSON().truncated).toBe(false);
  });
  it('camera breathing: a row per estimator step, withheld ones with the core status as reason', () => {
    let now = 0;
    const r = new SessionRecorder(['camera'], () => now);
    const log = new BreathCameraLogger(r, 2);
    const view = (status: PoseView['breathing']['status'], est: PoseView['breathing']['estimate']): PoseView => ({ t: 0, state: {} as never, held: false, breathing: { status, estimate: est } });
    for (const [t, st] of [[0, 'absent'], [0.5, 'absent'], [2, 'moving'], [4, 'ok']] as const) { now = t * 1000; log.onView(view(st, null)); }
    now = 6000; log.onView(view('ok', { t: 6, value: 15.2, quality: 0.7, reported: true, span_s: 28.5 }));
    now = 6125; log.onView(view('ok', { t: 6, value: 15.2, quality: 0.7, reported: true, span_s: 28.5 })); // same estimate: no new row
    now = 8000; log.onView(view('ok', { t: 8, value: 40, quality: 0.1, reported: false, span_s: 29 }));
    expect(r.estimates.map((e) => [e.t, e.value, e.reported, e.reason, e.span_s])).toEqual([
      [0, null, false, 'absent', undefined], [2, null, false, 'moving', undefined], [4, null, false, 'settling', undefined],
      [6, 15.2, true, 'ok', 28.5], [8, 40, false, 'low_signal', 29],
    ]);
    expect(validateSession(r.toJSON())).toEqual([]);
  });
  it('user-agent families and iso offsets', () => {
    expect(uaFamily('Mozilla/5.0 Edg/120.0')).toBe('Edge');
    expect(uaFamily('Mozilla/5.0 Firefox/130.0')).toBe('Firefox');
    expect(uaFamily('Mozilla/5.0 (iPhone) Version/17 Safari/605.1')).toBe('Safari');
    expect(uaFamily('curl/8')).toBe('Other');
    expect(isoWithOffset(Date.UTC(2026, 10, 20, 18, 0, 0, 250), -480)).toBe('2026-11-20T10:00:00.250-08:00');
    expect(isoWithOffset(Date.UTC(2026, 10, 20, 18, 0, 0), 330)).toBe('2026-11-20T23:30:00.000+05:30');
  });
});

describe('cough events follow the tracker', () => {
  const frames: [number, number][] = [];
  const hop = 0.48;
  const put = (t0: number, scores: number[]) => scores.forEach((sc, i) => frames.push([t0 + i * hop, sc]));
  put(0.96, [0.02, 0.02]);
  put(1.92, [0.6, 0.9]);               // first cough, still going on at the first view
  put(2.88, [0.02]);
  put(3.36, [0.95, 0.7]);              // starts < refractory after the first: merges into the SAME event (end and peak grow)
  put(4.32, new Array(30).fill(0.02));
  put(20.0, [0.8, 0.8, 0.8]);          // a separate cough
  put(21.5, new Array(8).fill(0.02));
  it('stored rows are updated until they end and equal the tracker/batch result at the end', () => {
    let now = 0;
    const rec = new SessionRecorder(['microphone'], () => now);
    const cr = new CoughRecorder(rec);
    const tr = new LabelEventTracker(params.audio.cough);
    const early: { t_end: number | undefined; peak: number | undefined }[] = [];
    for (const [t, sc] of frames) {
      now = t * 1000;
      tr.push(t, sc);
      cr.update({ ...tr.view(), t });
      if (t > 2.8 && t < 2.9) early.push({ t_end: rec.events[0]?.t_end, peak: rec.events[0]?.peak_score });
    }
    const batch = detectLabelEvents(frames, params.audio.cough).events;
    expect(rec.events.filter((e) => e.kind === 'cough')).toHaveLength(batch.length);
    expect(batch.length).toBe(2);
    batch.forEach((b, i) => {
      const row = rec.events[i]!;
      expect(row.t_end! - row.t_onset!).toBeCloseTo(b.t_end - b.t_start, 2);
      expect(row.peak_score).toBeCloseTo(b.peak_score, 2);
      expect(row.t_onset).toBeCloseTo(b.t_start, 2);
    });
    expect(early[0]!.peak).toBeLessThan(0.95); // it really was extended later (not exported prematurely)
    expect(rec.events[0]!.peak_score).toBeCloseTo(0.95, 2);
    rec.finish();
    now += 5000;
    cr.update({ count: 5, last: { t_start: 50, t_end: 51, peak_score: 1 }, t: 51 }); // after the session ended nothing changes
    expect(rec.events).toHaveLength(2);
  });
});

describe('delete all', () => {
  it('clearLocalPrefs removes only carelens.* keys', () => {
    const m = new Map([['carelens.theme', 'dark'], ['carelens.videoDeviceId', 'x'], ['other', '1']]);
    const st = { get length() { return m.size; }, key: (i: number) => [...m.keys()][i] ?? null, removeItem: (k: string) => void m.delete(k) };
    clearLocalPrefs(st);
    expect([...m.keys()]).toEqual(['other']);
  });
});

describe('production build contains no e2e provider', () => {
  const dir = 'dist/assets';
  it('has no scripted pose provider code or marker (run `npm run build` first; required in CI)', () => {
    if (!existsSync(dir)) { if (process.env.CI) expect(existsSync(dir)).toBe(true); return; }
    const files = readdirSync(dir).filter((f) => /\.(js|css|html)$/.test(f));
    expect(files.length).toBeGreaterThan(0);
    for (const f of files) {
      const text = readFileSync(join(dir, f), 'utf8');
      expect(text, f).not.toContain('CARELENS_E2E_SCRIPTED_POSE');
      expect(text, f).not.toContain('__E2E_POSE__');
      expect(text, f).not.toContain('scriptedPose');
      expect(text, f).not.toContain('CARELENS_E2E_SCRIPTED_HAND');
      expect(text, f).not.toContain('__E2E_HAND__');
      expect(text, f).not.toContain('scriptedHand');
      expect(text, f).not.toContain('CARELENS_E2E_SCRIPTED_FACE');
      expect(text, f).not.toContain('__E2E_FACE__');
      expect(text, f).not.toContain('scriptedFace');
    }
  });
});
