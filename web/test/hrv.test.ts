// W6 HRV TS port: unit rules, fixture-based accuracy against the known RR truth stored in the fixtures, and the card wording.
// Bit-level parity with the Python reference is covered by test/parity (hrv_* fixtures). Not a medical device.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { HrvRecorder } from '../src/export/hrv';
import { SessionRecorder } from '../src/export/recorder';
import { ContactPPG } from '../src/core/ppg';
import { analyzeHrv, assertHrvParams, hrvErrors, hrvMetrics, keepMask, runHrv, type HrvParams } from '../src/core/hrv';
import type { ContactPpgParams } from '../src/core/ppg';
import { HRV_NOTE, HRV_OFF_TEXT, hrvCardText, hrvWithheldText } from '../src/ui/hrvCard';

const SHARED = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'shared');
const params = JSON.parse(readFileSync(join(SHARED, 'params.json'), 'utf8'));
const HP: HrvParams = params.hrv;
const PP: ContactPpgParams = params.contact_ppg;
const fx = (n: string): any => JSON.parse(readFileSync(join(SHARED, 'fixtures', `${n}.json`), 'utf8'));

describe('hrv rules', () => {
  it('keepMask drops out-of-range and locally deviating IBIs; RMSSD does not bridge a rejected beat', () => {
    const ib = Array(12).fill(800) as number[];
    ib[5] = 3000; ib[8] = 1100;
    expect(keepMask(ib, HP).map((k, i) => (k ? -1 : i)).filter((i) => i >= 0)).toEqual([5, 8]);
    const m = hrvMetrics([800, 810, 1500, 820, 830], [true, true, false, true, true]);
    expect(m.rmssd).toBeCloseTo(Math.sqrt((100 + 100) / 2), 9);
    expect(m.meanHr).toBeCloseTo(60000 / 815, 9);
  });
  it('validates params', () => {
    expect(hrvErrors(HP)).toEqual([]);
    expect(() => assertHrvParams({ ...HP, ibi_min_ms: 3000 })).toThrow();
    expect(() => assertHrvParams({ ...HP, min_kept_frac: 0 })).toThrow();
  });
  it('withholds short, slow and unsteady segments', () => {
    const f = fx('hrv_clean_30ms').input;
    expect(analyzeHrv([0], [1], [1], 0, 0, PP, HP).reason).toBe('too_short');
    expect(analyzeHrv(f.t, f.r, f.g, 62, 50, PP, HP).reported).toBe(true);
    expect(analyzeHrv(f.t, f.r, f.g, 62, 49, PP, HP).reason).toBe('unsteady');
    const slow = { ...f, t: f.t.filter((_: number, i: number) => i % 2 === 0) }; // 15 fps
    expect(analyzeHrv(slow.t, slow.r.filter((_: number, i: number) => i % 2 === 0), slow.g.filter((_: number, i: number) => i % 2 === 0), 62, 62, PP, HP).reason).toBe('low_fps');
  });
});

describe('hrv accuracy on the fixtures (truth stored in the input)', () => {
  it.each(['hrv_clean_30ms', 'hrv_clean_80ms', 'hrv_ectopics'])('%s within max(6 ms, 20 %) of truth', (n) => {
    const inp = fx(n).input;
    const { hrv } = runHrv(inp, PP, HP);
    expect(hrv.reported).toBe(true);
    expect(Math.abs(hrv.rmssd_ms! - inp.truth_rmssd_ms)).toBeLessThanOrEqual(Math.max(6, 0.2 * inp.truth_rmssd_ms) + (n === 'hrv_ectopics' ? 1 : 0)); // ectopics: truth is of the undamaged series
    expect(Math.abs(hrv.sdnn_ms! - inp.truth_sdnn_ms)).toBeLessThanOrEqual(Math.max(6, 0.15 * inp.truth_sdnn_ms));
  });
  it('withholds with the right reasons', () => {
    expect(runHrv(fx('hrv_too_many_artifacts').input, PP, HP).hrv.reason).toBe('too_many_artifacts');
    expect(runHrv(fx('hrv_too_short').input, PP, HP).hrv.reason).toBe('too_short');
  });
});

describe('HRV card text', () => {
  const res = { rmssd_ms: 34.3, sdnn_ms: 28.6, mean_hr_bpm: 70, n_beats: 83, n_kept: 82, kept_frac: 0.988, duration_s: 70, reported: true, reason: 'ok' as const };
  it('is inactive before min duration, then shows RMSSD/SDNN with beats, duration and kept %', () => {
    expect(hrvCardText(null, 20, 60)).toEqual({ value: 'Hold steady for 60 s after the signal settles to see this (20 / 60 s).', muted: true, detail: null, quality: null });
    const t = hrvCardText(res, 70, 60);
    expect(t.value).toBe('RMSSD 34 ms, SDNN 29 ms');
    expect(t.detail).toBe('82 beats over 70 s. A wellbeing trend only.');
    expect(t.quality).toContain('99 %');
    const w = hrvCardText({ ...res, reported: false, rmssd_ms: null, sdnn_ms: null, reason: 'too_many_artifacts', kept_frac: 0.7 }, 70, 60);
    expect(w.value).not.toContain('RMSSD');
    expect(w.muted).toBe(true);
  });
  it('keeps the agreed note and never uses stress or health-score wording', () => {
    expect(HRV_NOTE).toBe('Experimental. Short recordings at camera frame rates give rough values; trends over many days mean more than a single number. Camera timing adds noise, so RMSSD tends to read higher than a chest strap or ECG — compare trends, not single numbers. Not a stress or health score.');
    expect(HRV_OFF_TEXT).toBe('Available in fingertip mode only.');
    const all = [HRV_OFF_TEXT, hrvCardText(null, 0, 60).value, hrvCardText(res, 70, 60).value + hrvCardText(res, 70, 60).detail,
      ...(['too_many_artifacts', 'unsteady', 'low_fps'] as const).map((r) => hrvWithheldText(r))].join(' ');
    expect(all).not.toMatch(/stress|recovery|readiness|score|diagnos|disease/i);
  });
});

describe('HRV card keeps the last reported result; collecting; recorder; segment', () => {
  const rep = { rmssd_ms: 34.3, sdnn_ms: 28.6, mean_hr_bpm: 70, n_beats: 83, n_kept: 82, kept_frac: 0.988, duration_s: 70, reported: true, reason: 'ok' as const };
  it('shows the last reported result while a new measurement is not ready', () => {
    for (const cur of [null, { ...rep, reported: false, rmssd_ms: null, sdnn_ms: null, reason: 'unsteady' as const }]) {
      const t = hrvCardText(cur, 5, 60, rep);
      expect(t.value).toBe('RMSSD 34 ms, SDNN 29 ms');
      expect(t.detail).toContain('Last completed measurement.');
    }
    expect(hrvCardText(rep, 70, 60, rep).detail).not.toContain('Last completed');
  });
  it('collecting is neutral', () => {
    const t = hrvCardText({ ...rep, reported: false, rmssd_ms: null, sdnn_ms: null, reason: 'collecting', n_kept: 40, duration_s: 61 }, 61, 60);
    expect(t.value).toBe('Collecting more beats…');
    expect(t.quality).toBeNull();
    expect(hrvWithheldText('too_many_artifacts')).not.toBe(t.value);
  });
  it('slow heart rate (48 bpm) is collecting, not artifacts, until the segment is full', () => {
    const f = fx('hrv_clean_30ms').input; // 62 s segment at ~71 bpm: reuse with a stricter beat floor
    const strict = { ...HP, min_kept_beats: 90 };
    expect(analyzeHrv(f.t, f.r, f.g, 62, 62, PP, strict).reason).toBe('collecting');
    expect(analyzeHrv(f.t, f.r, f.g, 62, 62, PP, { ...strict, max_duration_s: 62 }).reason).toBe('too_many_artifacts');
  });
  it('segment starts after settling', () => {
    const cp = new ContactPPG(PP, HP);
    const f = fx('hrv_clean_30ms').input;
    f.t.forEach((t: number, i: number) => cp.update(t, f.r[i], f.g[i], 0, 0));
    expect(cp.segmentSeconds()).toBeCloseTo(f.t[f.t.length - 1] - f.t[0] - PP.settle_s, 1);
  });
  it('HrvRecorder keeps the longest reported result and never replaces it by a withheld one', () => {
    const rec = new SessionRecorder(['fingertip'], () => 0, 'x');
    const hr = new HrvRecorder(rec, 60);
    const bad = { ...rep, reported: false, rmssd_ms: null, sdnn_ms: null, reason: 'too_many_artifacts' as const, duration_s: 80 };
    hr.update(1, { ...rep, reported: false, rmssd_ms: null, sdnn_ms: null, reason: 'collecting', duration_s: 61 });
    expect(rec.summaries[0]!.reason).toBe('collecting');
    hr.update(1, { ...rep, duration_s: 62 });
    expect(rec.summaries[0]!.reported).toBe(true);
    hr.update(1, bad);
    hr.update(1, { ...rep, duration_s: 61, rmssd_ms: 99 }); // shorter reported: ignored
    expect(rec.summaries[0]!.rmssd_ms).toBe(34.3);
    hr.update(1, { ...rep, duration_s: 90, rmssd_ms: 41 });
    expect(rec.summaries[0]!.rmssd_ms).toBe(41);
    hr.update(2, bad); // a new epoch gets its own row
    expect(rec.summaries.length).toBe(2);
    expect(rec.toJSON().summaries!.length).toBe(2);
  });
});
