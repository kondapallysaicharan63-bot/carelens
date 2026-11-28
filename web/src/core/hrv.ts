// Heart-rate variability (RMSSD, SDNN) from a fingertip contact segment (W6). DOM-free TypeScript port of
// carelens/hrv.py: read its docstring for the design. Not a medical device: a wellbeing trend, not a diagnosis, never a
// stress or health score. Driven by shared/params.json -> hrv (plus contact_ppg for the grid and band-pass).

import { bandpass, detrendRelative, fftSize, gridTimes, interp, mean, powerSpectrum, spectralPeak, spectralSnr } from './dsp';
import { ContactPPG, findBeats, type ContactPpgParams } from './ppg';

/** a frozen segment is within a frame or two of max_duration_s (mirrors COLLECT_MARGIN_S in hrv.py) */
export const COLLECT_MARGIN_S = 0.5;

export interface HrvParams {
  min_duration_s: number; max_duration_s: number; min_rate_hz: number; ibi_min_ms: number; ibi_max_ms: number;
  neighbours: number; max_dev_frac: number; min_kept_frac: number; min_kept_beats: number; min_ok_frac: number; min_amp_frac: number;
}

export type HrvReason = 'ok' | 'too_short' | 'low_fps' | 'unsteady' | 'too_many_artifacts' | 'collecting';

export interface HrvResult {
  rmssd_ms: number | null; sdnn_ms: number | null; mean_hr_bpm: number | null;
  n_beats: number; n_kept: number; kept_frac: number; duration_s: number; reported: boolean; reason: HrvReason;
}

/** Mirrors carelens.params.hrv_errors. */
export function hrvErrors(p: HrvParams): string[] {
  const e: string[] = [];
  if (!(0 < p.ibi_min_ms && p.ibi_min_ms < p.ibi_max_ms)) e.push('need 0 < ibi_min_ms < ibi_max_ms');
  if (!(0 < p.min_duration_s && p.min_duration_s <= p.max_duration_s)) e.push('need 0 < min_duration_s <= max_duration_s');
  for (const k of ['min_rate_hz', 'max_dev_frac'] as const) if (!(p[k] > 0)) e.push(`${k} must be > 0`);
  for (const k of ['min_kept_frac', 'min_ok_frac', 'min_amp_frac'] as const) if (!(p[k] > 0 && p[k] <= 1)) e.push(`${k} must be in (0, 1]`);
  if (p.neighbours < 1 || p.min_kept_beats < 2) e.push('neighbours must be >= 1 and min_kept_beats >= 2');
  return e;
}

export function assertHrvParams(p: HrvParams): HrvParams {
  const errs = hrvErrors(p);
  if (errs.length) throw new Error('invalid hrv params: ' + errs.join('; '));
  return p;
}

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m]! : 0.5 * (s[m - 1]! + s[m]!);
};

export function keepMask(ibis: number[], hp: HrvParams): boolean[] {
  const n = ibis.length, k = hp.neighbours;
  const rng = ibis.map((x) => hp.ibi_min_ms <= x && x <= hp.ibi_max_ms);
  const keep = [...rng];
  for (let i = 0; i < n; i++) {
    if (!rng[i]) continue;
    const nb: number[] = [];
    for (let j = Math.max(0, i - k); j < Math.min(n, i + k + 1); j++) if (j !== i && rng[j]) nb.push(ibis[j]!);
    if (nb.length === 0) { keep[i] = false; continue; }
    const m = median(nb);
    if (Math.abs(ibis[i]! - m) > hp.max_dev_frac * m) keep[i] = false;
  }
  return keep;
}

export function hrvMetrics(ibis: number[], keep: boolean[]): { rmssd: number | null; sdnn: number | null; meanHr: number | null } {
  const kept = ibis.filter((_, i) => keep[i]);
  let s2 = 0, nd = 0;
  for (let i = 0; i + 1 < ibis.length; i++) if (keep[i] && keep[i + 1]) { const d = ibis[i + 1]! - ibis[i]!; s2 += d * d; nd++; }
  const rmssd = nd ? Math.sqrt(s2 / nd) : null;
  let sdnn: number | null = null, meanHr: number | null = null;
  if (kept.length) {
    const m = kept.reduce((a, b) => a + b, 0) / kept.length;
    meanHr = 60000 / m;
    if (kept.length >= 2) sdnn = Math.sqrt(kept.reduce((a, b) => a + (b - m) * (b - m), 0) / (kept.length - 1));
  }
  return { rmssd, sdnn, meanHr };
}

/** Whole-segment beat series: IBIs (ms) and the beat count. */
export function segmentIbis(ts: number[], rs: number[], gs: number[], pp: ContactPpgParams, hp: HrvParams): { ibis: number[]; nBeats: number } {
  const tEnd = ts[ts.length - 1]!;
  const n = Math.floor((tEnd - ts[0]!) * pp.fs_hz + 1e-6);
  const grid = gridTimes(tEnd, n, pp.fs_hz);
  const nSpec = fftSize(n, pp.zero_pad_factor);
  let best: { snr: number; rel: number[] } | null = null;
  for (const raw of [rs, gs]) {
    const y = interp(grid, ts, raw);
    if (!(mean(y) > 1e-6)) continue;
    const { rel } = detrendRelative(y);
    const power = powerSpectrum(rel, nSpec);
    const snr = spectralSnr(power, nSpec, pp.fs_hz, spectralPeak(power, nSpec, pp.fs_hz, pp), pp);
    if (best === null || snr > best.snr) best = { snr, rel };
  }
  if (best === null) return { ibis: [], nBeats: 0 };
  const bp = bandpass(best.rel, fftSize(n, 1), pp.fs_hz, pp);
  const bm = mean(bp);
  let ss = 0;
  for (const v of bp) ss += (v - bm) * (v - bm);
  let beats = findBeats(bp, pp.fs_hz, 60 / pp.beat_max_bpm, pp.min_prom * Math.sqrt(ss / bp.length));
  if (beats.length) {
    const hts = beats.map((b) => bp[Math.floor(b + 0.5)]!);
    const mh = median(hts);
    beats = beats.filter((_, i) => hts[i]! >= hp.min_amp_frac * mh);
  }
  const bt = beats.map((b) => grid[0]! + b / pp.fs_hz);
  const ibis: number[] = [];
  for (let i = 0; i + 1 < bt.length; i++) ibis.push((bt[i + 1]! - bt[i]!) * 1000);
  return { ibis, nBeats: bt.length };
}

export function analyzeHrv(ts: number[], rs: number[], gs: number[], nUpd: number, nOk: number, pp: ContactPpgParams, hp: HrvParams): HrvResult {
  const out: HrvResult = { rmssd_ms: null, sdnn_ms: null, mean_hr_bpm: null, n_beats: 0, n_kept: 0, kept_frac: 0, duration_s: 0, reported: false, reason: 'too_short' };
  if (ts.length < 2) return out;
  const dur = ts[ts.length - 1]! - ts[0]!;
  out.duration_s = dur;
  if (dur < hp.min_duration_s) return out;
  if ((ts.length - 1) / dur < hp.min_rate_hz) { out.reason = 'low_fps'; return out; }
  if (nUpd <= 0 || nOk < hp.min_ok_frac * nUpd) { out.reason = 'unsteady'; return out; }
  const { ibis, nBeats } = segmentIbis(ts, rs, gs, pp, hp);
  const keep = keepMask(ibis, hp);
  const nKept = keep.filter(Boolean).length;
  out.n_beats = nBeats; out.n_kept = nKept; out.kept_frac = ibis.length ? nKept / ibis.length : 0;
  const { rmssd, sdnn, meanHr } = hrvMetrics(ibis, keep);
  if (out.kept_frac < hp.min_kept_frac || rmssd === null || sdnn === null) { out.reason = 'too_many_artifacts'; return out; }
  if (nKept < hp.min_kept_beats) { out.reason = dur < hp.max_duration_s - COLLECT_MARGIN_S ? 'collecting' : 'too_many_artifacts'; return out; } // slow heart rate: needs more time
  out.rmssd_ms = rmssd; out.sdnn_ms = sdnn; out.mean_hr_bpm = meanHr; out.reported = true; out.reason = 'ok';
  return out;
}

/** HRV of the contact segment buffered so far by `cp` (which must have been built with HRV params). */
export function hrvResult(cp: ContactPPG, pp: ContactPpgParams, hp: HrvParams): HrvResult {
  const s = cp.segment();
  return analyzeHrv(s.t, s.r, s.g, s.nUpd, s.nOk, pp, hp);
}

/** Fixture runner: mirrors carelens.hrv.run_hrv. */
export function runHrv(inp: { t: number[]; r: number[]; g: number[]; b: number[]; clip_frac: number[] }, pp: ContactPpgParams, hp: HrvParams): { hrv: HrvResult } {
  const cp = new ContactPPG(pp, hp);
  inp.t.forEach((t, i) => cp.update(t, inp.r[i]!, inp.g[i]!, inp.b[i]!, inp.clip_frac[i]!));
  return { hrv: hrvResult(cp, pp, hp) };
}
