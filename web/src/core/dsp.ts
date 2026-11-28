// Shared DSP building blocks for the PPG pipelines (contact PPG W1, face rPPG W2): DOM-free mirror of
// carelens/dsp.py. Hand-written radix-2 FFT, linear interpolation, Hann, raised-cosine band-pass,
// harmonic-sum fundamental selection with parabolic refinement, band SNR. Functions take any params
// object with the SpectralParams fields (contact_ppg and rppg in shared/params.json both qualify).
// Not a medical device.

/** Fields used by the peak picker and the SNR (audio.breathing provides just these). */
export interface PeakParams {
  band_lo_hz: number; band_hi_hz: number;
  snr_halfwidth_hz: number; harm_halfwidth_hz: number; harm_w2: number; harm_w3: number;
}

export interface SpectralParams extends PeakParams {
  taper_stop_lo_hz: number; taper_stop_hi_hz: number;
}

/** What the band-pass gain needs (the peak-picker fields are not used by taper/bandpass). */
export type BandParams = Pick<SpectralParams, 'band_lo_hz' | 'band_hi_hz' | 'taper_stop_lo_hz' | 'taper_stop_hi_hz'>;

export const EPS = 1e-30;

// ------------------------------------------------------------------ FFT (iterative radix-2, no deps)

/** In-place forward complex FFT; length must be a power of two. */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n & (n - 1)) throw new Error('fft length must be a power of two');
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]!; re[i] = re[j]!; re[j] = t;
      t = im[i]!; im[i] = im[j]!; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const wr = new Float64Array(half);
    const wi = new Float64Array(half);
    for (let k = 0; k < half; k++) {
      const a = (-2 * Math.PI * k) / len;
      wr[k] = Math.cos(a);
      wi[k] = Math.sin(a);
    }
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const xr = re[b]! * wr[k]! - im[b]! * wi[k]!;
        const xi = re[b]! * wi[k]! + im[b]! * wr[k]!;
        re[b] = re[a]! - xr; im[b] = im[a]! - xi;
        re[a] = re[a]! + xr; im[a] = im[a]! + xi;
      }
    }
  }
}

/** In-place inverse FFT (conjugate trick, 1/N scaling). */
export function ifft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i]!;
  fft(re, im);
  for (let i = 0; i < n; i++) { re[i] = re[i]! / n; im[i] = -im[i]! / n; }
}

// ------------------------------------------------------------------ pure building blocks

export function gridTimes(tEnd: number, n: number, fs: number): number[] {
  const g = new Array<number>(n);
  for (let k = 0; k < n; k++) g[k] = tEnd - (n - 1 - k) / fs;
  return g;
}

/** Linear interpolation, holding the end values outside [xp[0], xp[last]] (like numpy.interp). */
export function interp(x: number[], xp: number[], fp: number[]): number[] {
  const out = new Array<number>(x.length);
  let j = 0;
  const last = xp.length - 1;
  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    if (v <= xp[0]!) { out[i] = fp[0]!; continue; }
    if (v >= xp[last]!) { out[i] = fp[last]!; continue; }
    while (xp[j + 1]! <= v) j++;
    const slope = (fp[j + 1]! - fp[j]!) / (xp[j + 1]! - xp[j]!);
    out[i] = slope * (v - xp[j]!) + fp[j]!;
  }
  return out;
}

export const mean = (a: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]!;
  return s / a.length;
};

/** Subtract the closed-form least-squares line (against the sample index); divide by the raw mean. */
export function detrendRelative(y: number[]): { rel: number[]; mean: number } {
  const n = y.length;
  const ybar = mean(y);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const kk = i - (n - 1) / 2;
    num += kk * (y[i]! - ybar);
    den += kk * kk;
  }
  const slope = num / den;
  const rel = new Array<number>(n);
  for (let i = 0; i < n; i++) rel[i] = (y[i]! - (ybar + slope * (i - (n - 1) / 2))) / ybar;
  return { rel, mean: ybar };
}

/** Linear-interpolation percentile (numpy default), q in [0, 100]. */
export function percentile(x: number[], q: number): number {
  const s = Float64Array.from(x).sort();
  const h = ((s.length - 1) * q) / 100;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo]! + (s[hi]! - s[lo]!) * (h - lo);
}

export function hann(n: number): number[] {
  const w = new Array<number>(n);
  for (let k = 0; k < n; k++) w[k] = 0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (n - 1));
  return w;
}

export function fftSize(n: number, factor: number): number {
  const need = factor * n;
  let m = 1;
  while (m < need) m *= 2;
  return m;
}

/** |fft(hann * rel, nFft)|^2 for bins 0..nFft/2. */
export function powerSpectrum(rel: number[], nFft: number): Float64Array {
  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  const w = hann(rel.length);
  for (let i = 0; i < rel.length; i++) re[i] = rel[i]! * w[i]!;
  fft(re, im);
  const p = new Float64Array(nFft / 2 + 1);
  for (let k = 0; k < p.length; k++) p[k] = re[k]! * re[k]! + im[k]! * im[k]!;
  return p;
}

export function bandBins(nFft: number, fs: number, lo: number, hi: number): [number, number] {
  return [Math.ceil((lo * nFft) / fs - 1e-9), Math.floor((hi * nFft) / fs + 1e-9)];
}

function argmax(a: ArrayLike<number>, from: number, to: number): number {
  let best = from;
  for (let k = from + 1; k <= to; k++) if (a[k]! > a[best]!) best = k;
  return best;
}

export const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

/** Peak offset in [-0.5, 0.5] bins from three linear samples. */
export function parabolicLin(a: number, b: number, c: number): number {
  const den = a - 2 * b + c;
  return den === 0 ? 0 : clamp((0.5 * (a - c)) / den, -0.5, 0.5);
}

/** Peak offset in [-0.5, 0.5] bins at k, parabola through log power. */
export function parabolicLog(power: ArrayLike<number>, k: number): number {
  return parabolicLin(Math.log(power[k - 1]! + EPS), Math.log(power[k]! + EPS), Math.log(power[k + 1]! + EPS));
}

export function peakBin(power: ArrayLike<number>, nFft: number, fs: number, f: number, hw: number, kLo: number, kHi: number): number {
  const lo = Math.max(kLo, Math.ceil(((f - hw) * nFft) / fs - 1e-9));
  const hi = Math.min(kHi, Math.floor(((f + hw) * nFft) / fs + 1e-9));
  return hi < lo ? -1 : argmax(power, lo, hi);
}

/** S(f) = P(f) + w2 P(2f) + w3 P(3f); P = peak power within +-harm_halfwidth; terms at/above Nyquist are 0. */
function harmonicScore(power: ArrayLike<number>, nFft: number, fs: number, f: number, p: PeakParams): number {
  let sc = 0;
  for (const [mult, w] of [[1, 1], [2, p.harm_w2], [3, p.harm_w3]] as const) {
    if (mult * f < fs / 2) {
      const k = peakBin(power, nFft, fs, mult * f, p.harm_halfwidth_hz, 0, nFft / 2);
      if (k >= 0) sc += w * power[k]!;
    }
  }
  return sc;
}

/** Refined in-band fundamental (Hz): candidates f0, f0/2 and 2 f0 scored by the harmonic sum; ties go to the higher f. */
export function spectralPeak(power: Float64Array, nFft: number, fs: number, p: PeakParams): number {
  const [kLo, kHi] = bandBins(nFft, fs, p.band_lo_hz, p.band_hi_hz);
  const k0 = argmax(power, kLo, kHi);
  const f0 = (k0 * fs) / nFft;
  let bestK = k0, bestS = harmonicScore(power, nFft, fs, f0, p);
  for (const fc of [f0 / 2, f0 * 2]) {
    if (!(p.band_lo_hz <= fc && fc <= p.band_hi_hz)) continue;
    const kc = peakBin(power, nFft, fs, fc, p.harm_halfwidth_hz, kLo, kHi);
    if (kc < 0) continue;
    const sc = harmonicScore(power, nFft, fs, (kc * fs) / nFft, p);
    if (sc > bestS || (sc === bestS && kc > bestK)) { bestK = kc; bestS = sc; }
  }
  return ((bestK + parabolicLog(power, bestK)) * fs) / nFft;
}

export function spectralSnr(power: Float64Array, nFft: number, fs: number, fStar: number, p: PeakParams): number {
  const [kLo, kHi] = bandBins(nFft, fs, p.band_lo_hz, p.band_hi_hz);
  let sig = 0, noise = 0;
  const two = 2 * fStar < fs / 2;
  for (let k = kLo; k <= kHi; k++) {
    const f = (k * fs) / nFft;
    const inSig = Math.abs(f - fStar) <= p.snr_halfwidth_hz || (two && Math.abs(f - 2 * fStar) <= p.snr_halfwidth_hz);
    if (inSig) sig += power[k]!; else noise += power[k]!;
  }
  return sig / (noise + EPS);
}

/** Raised-cosine band-pass gain at frequency f (Hz). */
export function taper(f: number, p: BandParams): number {
  if (f >= p.taper_stop_lo_hz && f < p.band_lo_hz) {
    return 0.5 - 0.5 * Math.cos((Math.PI * (f - p.taper_stop_lo_hz)) / (p.band_lo_hz - p.taper_stop_lo_hz));
  }
  if (f >= p.band_lo_hz && f <= p.band_hi_hz) return 1;
  if (f > p.band_hi_hz && f < p.taper_stop_hi_hz) {
    return 0.5 + 0.5 * Math.cos((Math.PI * (f - p.band_hi_hz)) / (p.taper_stop_hi_hz - p.band_hi_hz));
  }
  return 0;
}

/** FFT-domain band-pass of the zero-padded signal, truncated back to the window length. */
export function bandpass(rel: number[], nFft: number, fs: number, p: BandParams): number[] {
  const re = new Float64Array(nFft), im = new Float64Array(nFft);
  for (let i = 0; i < rel.length; i++) re[i] = rel[i]!;
  fft(re, im);
  for (let k = 0; k < nFft; k++) {
    const g = taper(((k <= nFft / 2 ? k : nFft - k) * fs) / nFft, p);
    re[k] = re[k]! * g; im[k] = im[k]! * g;
  }
  ifft(re, im);
  return Array.from(re.subarray(0, rel.length));
}

/** Subtract the closed-form least-squares line (against the sample index); no scaling. */
export function detrendLinear(y: ArrayLike<number>): number[] {
  const n = y.length;
  const ybar = mean(y);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const kk = i - (n - 1) / 2;
    num += kk * (y[i]! - ybar);
    den += kk * kk;
  }
  const slope = num / den;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = y[i]! - (ybar + slope * (i - (n - 1) / 2));
  return out;
}
