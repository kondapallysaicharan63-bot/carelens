// Rhythmic hand movement (3-12 Hz) from MediaPipe HandLandmarker points (W8, Experimental): DOM-free TypeScript port of
// carelens/tremor.py (read its docstring for the design). Not a medical device and not a test for any condition: a
// wellbeing trend to compare with your own earlier results; "no clear rhythmic hand movement" is the expected result for
// most people. Driven by shared/params.json -> tremor.

import { bandpass, bandBins, detrendLinear, fftSize, gridTimes, interp, parabolicLog, powerSpectrum, type BandParams } from './dsp';

export interface TremorParams extends BandParams {
  fs_hz: number; test_s: number; min_window_s: number; min_rate_hz: number; max_gap_s: number; min_hand_score: number;
  zero_pad_factor: number; edge_trim_s: number; prom_halfwidth_hz: number; motion_band_hi_hz: number; moving_max_pct: number;
  min_prominence: number; noise_floor_pct: number;
}

export type TremorReason = 'ok' | 'no_hand' | 'too_short' | 'low_fps' | 'moving' | 'no_rhythm';

export interface TremorResult {
  reported: boolean; reason: TremorReason;
  freq_hz: number | null; amp_pct: number | null; p2p_pct: number | null;
  prominence: number | null; rms_pct: number | null; motion_pct: number | null;
  duration_s: number; rate_hz: number | null; n_frames: number;
}

/** One camera frame: 21 [x, y] points normalised by frame width / height (or null) and the hand score. */
export interface HandFrame { t: number; lm: number[][] | null; score: number }

export const WRIST = 0;
export const MIDDLE_MCP = 9;
/** wrist, thumb MCP + 4 finger MCPs, 5 fingertips */
export const CENTROID_INDICES = [0, 2, 5, 9, 13, 17, 4, 8, 12, 16, 20];

const EPS = 1e-30;

/** Mirrors carelens.params.tremor_errors. */
export function tremorErrors(p: TremorParams): string[] {
  const e: string[] = [];
  if (!(0 < p.taper_stop_lo_hz && p.taper_stop_lo_hz < p.band_lo_hz && p.band_lo_hz < p.band_hi_hz && p.band_hi_hz < p.taper_stop_hi_hz && p.taper_stop_hi_hz < p.fs_hz / 2)) e.push('need 0 < taper_stop_lo < band_lo < band_hi < taper_stop_hi < Nyquist');
  if (!(0 < p.motion_band_hi_hz && p.motion_band_hi_hz < p.taper_stop_lo_hz)) e.push('need 0 < motion_band_hi_hz < taper_stop_lo_hz');
  if (!(2 * p.band_hi_hz < p.min_rate_hz && p.min_rate_hz <= p.fs_hz)) e.push('need 2 * band_hi_hz < min_rate_hz <= fs_hz (never report an aliased frequency)');
  if (!(0 < p.min_window_s && p.min_window_s <= p.test_s)) e.push('need 0 < min_window_s <= test_s');
  if (!(0 <= p.edge_trim_s && p.edge_trim_s < p.min_window_s / 4)) e.push('edge_trim_s must be in [0, min_window_s / 4)');
  for (const k of ['max_gap_s', 'prom_halfwidth_hz', 'moving_max_pct', 'noise_floor_pct'] as const) if (!(p[k] > 0)) e.push(`${k} must be > 0`);
  for (const k of ['min_hand_score', 'min_prominence'] as const) if (!(p[k] > 0 && p[k] <= 1)) e.push(`${k} must be in (0, 1]`);
  if (p.zero_pad_factor < 1) e.push('zero_pad_factor must be >= 1');
  return e;
}

export function assertTremorParams(p: TremorParams): TremorParams {
  const errs = tremorErrors(p);
  if (errs.length) throw new Error('invalid tremor params: ' + errs.join('; '));
  return p;
}

function result(reason: TremorReason, kw: Partial<TremorResult> = {}): TremorResult {
  return {
    reported: reason === 'ok', reason, freq_hz: null, amp_pct: null, p2p_pct: null, prominence: null, rms_pct: null, motion_pct: null,
    duration_s: 0, rate_hz: null, n_frames: 0, ...kw,
  };
}

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m]! : 0.5 * (s[m - 1]! + s[m]!);
};

/** One test's frames -> result. `aspect` = frame width / height. */
export function analyseTremor(frames: HandFrame[], aspect: number, tp: TremorParams): TremorResult {
  const present: number[] = [];
  frames.forEach((f, i) => { if (f.lm !== null && f.score >= tp.min_hand_score) present.push(i); });
  if (present.length === 0) return result('no_hand');
  const runs: number[][] = [];
  let cur: number[] = [present[0]!];
  for (const i of present.slice(1)) {
    if (frames[i]!.t - frames[cur[cur.length - 1]!]!.t > tp.max_gap_s) { runs.push(cur); cur = []; }
    cur.push(i);
  }
  runs.push(cur);
  let run = runs[0]!;
  for (const r of runs) if (frames[r[r.length - 1]!]!.t - frames[r[0]!]!.t > frames[run[run.length - 1]!]!.t - frames[run[0]!]!.t) run = r; // longest; ties: the first
  const tEnd = frames[run[run.length - 1]!]!.t;
  const sel = run.filter((i) => frames[i]!.t >= tEnd - tp.test_s - 1e-9);
  const duration = tEnd - frames[sel[0]!]!.t;
  const rate = duration > 0 ? (sel.length - 1) / duration : 0;
  const base = { duration_s: duration, rate_hz: rate, n_frames: sel.length };
  if (duration < tp.min_window_s - 1e-9) return result('too_short', base);
  if (rate < tp.min_rate_hz) return result('low_fps', base);

  const pts = sel.map((i) => frames[i]!.lm!.map((p) => [p[0]! * aspect, p[1]!]));
  const size = median(pts.map((p) => Math.hypot(p[MIDDLE_MCP]![0]! - p[WRIST]![0]!, p[MIDDLE_MCP]![1]! - p[WRIST]![1]!)));
  if (!(size > 1e-6)) return result('no_hand', base);
  const cx: number[] = [], cy: number[] = [];
  for (const p of pts) {
    let sx = 0, sy = 0;
    for (const k of CENTROID_INDICES) { sx += p[k]![0]!; sy += p[k]![1]!; }
    cx.push(sx / CENTROID_INDICES.length / size);
    cy.push(sy / CENTROID_INDICES.length / size);
  }
  const tSel = sel.map((i) => frames[i]!.t);
  const fs = tp.fs_hz;
  const n = Math.floor(duration * fs + 1e-6);
  const grid = gridTimes(tEnd, n, fs);
  const x = detrendLinear(interp(grid, tSel, cx));
  const y = detrendLinear(interp(grid, tSel, cy));
  const nFft = fftSize(n, tp.zero_pad_factor);

  const low: BandParams = { taper_stop_lo_hz: 0, band_lo_hz: 0, band_hi_hz: tp.motion_band_hi_hz, taper_stop_hi_hz: tp.taper_stop_lo_hz };
  const lx = bandpass(x, nFft, fs, low), ly = bandpass(y, nFft, fs, low);
  let ms = 0;
  for (let i = 0; i < n; i++) ms += lx[i]! * lx[i]! + ly[i]! * ly[i]!;
  const motion = 100 * Math.sqrt(ms / n);
  if (motion > tp.moving_max_pct) return result('moving', { motion_pct: motion, ...base });

  const k = Math.round(tp.edge_trim_s * fs);
  const trim = (a: number[]): number[] => {
    const b = a.slice(k, n - k);
    let m = 0;
    for (const v of b) m += v;
    m /= b.length;
    return b.map((v) => v - m);
  };
  const bx = trim(bandpass(x, nFft, fs, tp)), by = trim(bandpass(y, nFft, fs, tp));
  const m = bx.length;
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < m; i++) { sxx += bx[i]! * bx[i]!; syy += by[i]! * by[i]!; sxy += bx[i]! * by[i]!; }
  sxx /= m; syy /= m; sxy /= m;
  const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const c = Math.cos(th), s = Math.sin(th);
  const u = new Array<number>(m);
  let uu = 0, umax = -Infinity, umin = Infinity;
  for (let i = 0; i < m; i++) {
    const v = c * bx[i]! + s * by[i]!;
    u[i] = v; uu += v * v;
    if (v > umax) umax = v;
    if (v < umin) umin = v;
  }
  const rms = 100 * Math.sqrt(uu / m);
  const p2p = 100 * (umax - umin);

  const n2 = fftSize(m, tp.zero_pad_factor);
  const power = powerSpectrum(u, n2);
  const [kLo, kHi] = bandBins(n2, fs, tp.band_lo_hz, tp.band_hi_hz);
  let k0 = kLo;
  for (let j = kLo + 1; j <= kHi; j++) if (power[j]! > power[k0]!) k0 = j;
  const freq = ((k0 + parabolicLog(power, k0)) * fs) / n2;
  let near = 0, total = 0;
  for (let j = kLo; j <= kHi; j++) {
    total += power[j]!;
    if (Math.abs((j * fs) / n2 - freq) <= tp.prom_halfwidth_hz) near += power[j]!;
  }
  const prom = near / (total + EPS);

  const common = { prominence: prom, rms_pct: rms, motion_pct: motion, ...base };
  if (prom < tp.min_prominence || rms < tp.noise_floor_pct) return result('no_rhythm', common);
  return result('ok', { freq_hz: freq, amp_pct: rms, p2p_pct: p2p, ...common });
}

/** The sample clock could not be verified (no per-frame signal) and the camera does not declare >= min_rate_hz: withhold as low_fps
 *  instead of risking an aliased value. Keeps the usable-window numbers of `r`; `no_hand` and `too_short` stay as they are. */
export function forceLowFps(r: TremorResult): TremorResult {
  if (r.reason === 'no_hand' || r.reason === 'too_short') return r;
  return result('low_fps', { duration_s: r.duration_s, rate_hz: r.rate_hz, n_frames: r.n_frames });
}

/** Replays `tremor` fixtures (carelens.tremor.run_tremor). */
export function runTremorFixture(input: any, params: { tremor: TremorParams }): TremorResult {
  const lms = input.landmarks as (number[][] | null)[];
  const frames: HandFrame[] = lms.map((lm, i) => ({ t: input.t ? input.t[i] : i / input.fps, lm, score: input.score[i] }));
  return analyseTremor(frames, input.w / input.h, params.tremor);
}
