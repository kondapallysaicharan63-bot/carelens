// scipy-compatible building blocks for the vitals window estimate (carelens.vitals.estimate_window):
// Butterworth band-pass as second-order sections, zero-phase sosfiltfilt (odd padding, steady-state initial
// conditions) and a Welch spectrum (periodic Hann, linear detrend per segment). DOM-free. Not a medical device.

import { fft } from './dsp';

export type Sos = number[][]; // rows [b0, b1, b2, 1, a1, a2]

/** Digital Butterworth band-pass, like scipy.signal.butter(order, [lo, hi], 'band', output='sos') (wn = f / nyquist).
 *  The section pairing may differ from scipy's; the overall transfer function (and so the filtfilt output) is identical. */
export function butterBandpassSos(order: number, wnLo: number, wnHi: number): Sos {
  const fs2 = 4; // bilinear with fs = 2 -> 2 * fs
  const uLo = 4 * Math.tan((Math.PI * wnLo) / 2);
  const uHi = 4 * Math.tan((Math.PI * wnHi) / 2);
  const bw = uHi - uLo;
  const wo = Math.sqrt(uLo * uHi);
  // analogue low-pass prototype poles (left half plane, unit circle)
  const proto: [number, number][] = [];
  for (let k = 1 - order; k < order; k += 2) {
    const th = (Math.PI * k) / (2 * order);
    proto.push([-Math.cos(th), -Math.sin(th)]);
  }
  // lowpass -> bandpass: p -> p*bw/2 +- sqrt((p*bw/2)^2 - wo^2)
  const poles: [number, number][] = [];
  for (const [pr, pi] of proto) {
    const ar = (pr * bw) / 2, ai = (pi * bw) / 2;
    // sqrt of complex (ar+ i ai)^2 - wo^2
    const dr = ar * ar - ai * ai - wo * wo, di = 2 * ar * ai;
    const m = Math.hypot(dr, di);
    let sr = Math.sqrt((m + dr) / 2);
    let si = Math.sqrt(Math.max(0, (m - dr) / 2));
    if (di < 0) si = -si;
    poles.push([ar + sr, ai + si], [ar - sr, ai - si]);
  }
  // bilinear transform of the poles; gain k = bw^order * real(prod(fs2 - z) / prod(fs2 - p)) with z = 0 -> fs2^order
  let kr = 1, ki = 0; // prod(fs2 - p) as complex
  const zp: [number, number][] = [];
  for (const [pr, pi] of poles) {
    // (fs2 - p)
    const dr = fs2 - pr, di = -pi;
    const nr = kr * dr - ki * di, ni = kr * di + ki * dr;
    kr = nr; ki = ni;
    // z = (fs2 + p) / (fs2 - p)
    const ar = fs2 + pr, ai = pi, den = dr * dr + di * di;
    zp.push([(ar * dr + ai * di) / den, (ai * dr - ar * di) / den]);
  }
  const num = Math.pow(fs2, order); // prod(fs2 - 0) over the `order` analogue zeros at 0
  const gain = (Math.pow(bw, order) * (num * kr)) / (kr * kr + ki * ki); // real part of num / prod
  // pair conjugate poles; digital zeros: `order` at +1 and `order` at -1 -> one (z-1)(z+1) per section
  const used = new Array<boolean>(zp.length).fill(false);
  const sos: Sos = [];
  for (let i = 0; i < zp.length; i++) {
    if (used[i]) continue;
    used[i] = true;
    let best = -1, bd = Infinity;
    for (let j = i + 1; j < zp.length; j++) {
      if (used[j]) continue;
      const d = Math.hypot(zp[j]![0] - zp[i]![0], zp[j]![1] + zp[i]![1]);
      if (d < bd) { bd = d; best = j; }
    }
    used[best] = true;
    const [r1, i1] = zp[i]!, [r2, i2] = zp[best]!;
    // (z - p1)(z - p2) with conjugate-ish pair: a1 = -(p1+p2), a2 = p1*p2 (both real up to rounding)
    sos.push([1, 0, -1, 1, -(r1 + r2), r1 * r2 - i1 * i2]);
  }
  const first = sos[0]!;
  first[0] = first[0]! * gain; first[1] = first[1]! * gain; first[2] = first[2]! * gain;
  return sos;
}

/** Steady-state section states for a unit step (scipy.signal.sosfilt_zi). */
export function sosfiltZi(sos: Sos): number[][] {
  const zi: number[][] = [];
  let scale = 1;
  for (const s of sos) {
    const b0 = s[0]!, b1 = s[1]!, b2 = s[2]!, a1 = s[4]!, a2 = s[5]!;
    const g = (b0 + b1 + b2) / (1 + a1 + a2);
    zi.push([scale * (g - b0), scale * (b2 - a2 * g)]);
    scale *= g;
  }
  return zi;
}

/** Cascade of direct-form-II-transposed biquads with initial states `zi` (scipy.signal.sosfilt). */
export function sosfilt(sos: Sos, x: ArrayLike<number>, zi: number[][]): Float64Array {
  const y = Float64Array.from(x);
  for (let k = 0; k < sos.length; k++) {
    const [b0, b1, b2, , a1, a2] = sos[k]! as [number, number, number, number, number, number];
    let z0 = zi[k]![0]!, z1 = zi[k]![1]!;
    for (let i = 0; i < y.length; i++) {
      const xi = y[i]!;
      const yi = b0 * xi + z0;
      z0 = b1 * xi - a1 * yi + z1;
      z1 = b2 * xi - a2 * yi;
      y[i] = yi;
    }
  }
  return y;
}

/** Zero-phase filtering like scipy.signal.sosfiltfilt (padtype 'odd', default padlen). */
export function sosfiltfilt(sos: Sos, x: ArrayLike<number>): Float64Array {
  const n = x.length;
  let ntaps = 2 * sos.length + 1;
  const z2 = sos.filter((s) => s[2] === 0).length, z5 = sos.filter((s) => s[5] === 0).length;
  ntaps -= Math.min(z2, z5);
  const edge = ntaps * 3;
  if (n <= edge) throw new Error('sosfiltfilt: input too short');
  const ext = new Float64Array(n + 2 * edge);
  for (let i = 0; i < edge; i++) {
    ext[i] = 2 * x[0]! - x[edge - i]!;
    ext[edge + n + i] = 2 * x[n - 1]! - x[n - 2 - i]!;
  }
  for (let i = 0; i < n; i++) ext[edge + i] = x[i]!;
  const zi0 = sosfiltZi(sos);
  const scaled = (v: number) => zi0.map((z) => [z[0]! * v, z[1]! * v]);
  const y1 = sosfilt(sos, ext, scaled(ext[0]!));
  y1.reverse();
  const y2 = sosfilt(sos, y1, scaled(y1[0]!)); // scipy: zi * y_0 with y_0 = last sample of the forward pass (= first of the reversed)
  y2.reverse();
  return y2.slice(edge, edge + n);
}

/** Welch power spectral density (scipy.signal.welch: periodic Hann, linear detrend, mean average, one-sided). */
export function welch(x: ArrayLike<number>, fs: number, nperseg: number, noverlap: number, nfft: number): { f: Float64Array; pxx: Float64Array } {
  const n = x.length;
  const step = nperseg - noverlap;
  const nseg = Math.floor((n - noverlap) / step);
  const win = new Float64Array(nperseg);
  let wss = 0;
  for (let k = 0; k < nperseg; k++) { win[k] = 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / nperseg); wss += win[k]! * win[k]!; }
  const scale = 1 / (fs * wss);
  const half = nfft / 2 + 1;
  const pxx = new Float64Array(half);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  const tmean = (nperseg - 1) / 2;
  let sxx = 0;
  for (let k = 0; k < nperseg; k++) sxx += (k - tmean) * (k - tmean);
  for (let s = 0; s < nseg; s++) {
    const o = s * step;
    let ym = 0;
    for (let k = 0; k < nperseg; k++) ym += x[o + k]!;
    ym /= nperseg;
    let sxy = 0;
    for (let k = 0; k < nperseg; k++) sxy += (k - tmean) * (x[o + k]! - ym);
    const slope = sxy / sxx;
    re.fill(0); im.fill(0);
    for (let k = 0; k < nperseg; k++) re[k] = (x[o + k]! - (ym + slope * (k - tmean))) * win[k]!;
    fft(re, im);
    for (let k = 0; k < half; k++) {
      let p = (re[k]! * re[k]! + im[k]! * im[k]!) * scale;
      if (k !== 0 && k !== nfft / 2) p *= 2;
      pxx[k] = pxx[k]! + p;
    }
  }
  const f = new Float64Array(half);
  for (let k = 0; k < half; k++) { pxx[k] = pxx[k]! / nseg; f[k] = (k * fs) / nfft; }
  return { f, pxx };
}
