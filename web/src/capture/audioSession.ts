// Microphone -> AudioWorklet (audio thread) -> 16 kHz -> 50 Hz envelope (breathing from sound) and YAMNet windows
// (cough count, speech gate). Only created when the mic consent is on. Audio is processed in memory and never stored or sent.
// The worklet keeps capture off the main thread; envelope DSP is ~50 small FFTs/s and YAMNet runs once per 0.48 s on the main thread.
// Not a medical device.
import params from '../../../shared/params.json';
import {
  BreathSoundRate, EnvelopeStream, LabelEventTracker, Resampler, assertAudioParams, frameSeconds, type AudioEventsParams, type AudioParams,
  type BreathSoundUpdate, type LabelEvent, type LabelView,
} from '../core/audio';
import { PauseTracker, assertSleepParams, type PauseView, type SleepSoundsParams } from '../core/sleepSounds';
import type { YamnetHandle } from './yamnet';

export const AUDIO_PARAMS = assertAudioParams(params.audio as AudioParams);
export const EVENT_PARAMS = params.audio_events as AudioEventsParams;
export const SLEEP_PARAMS = params.sleep_sounds as SleepSoundsParams;
assertSleepParams(EVENT_PARAMS, SLEEP_PARAMS);
/** the four W10 distress-sound labels, in display order (Screaming, Crying/sobbing, Glass, Thump/thud) */
export const DISTRESS_SECTIONS = [EVENT_PARAMS.screaming, EVENT_PARAMS.crying, EVENT_PARAMS.glass, EVENT_PARAMS.thump] as const;
const BASE = import.meta.env.BASE_URL;

/** W10 sound results after every YAMNet frame (all counts only grow within a session) */
export interface SoundsView {
  /** stream time of the frame (s) */
  t: number;
  snoring: LabelView;
  talking: LabelView;
  distress: { label: string; view: LabelView }[];
  pauses: PauseView;
}

export interface AudioHooks {
  /** per block: RMS and peak of the (16 kHz) signal, both in [0, 1] */
  onLevel(rms: number, peak: number): void;
  onBreath(u: BreathSoundUpdate): void;
  /** every new YAMNet frame; `count` only grows. `ticks` = onset times (stream s) of the most recent events. */
  onCough(r: { count: number; last: LabelEvent | null; ticks: number[]; t: number }): void;
  /** the microphone stopped delivering audio (true) / is delivering again (false); every session emits its initial state */
  onNoMic(dead: boolean): void;
  onYamnet(state: 'loading' | 'ready' | 'failed', msg?: string): void;
  /** W10: snoring, talking time, distress sounds and possible pauses in breathing sounds (every new YAMNet frame) */
  onSounds?(v: SoundsView): void;
}

/** Browser objects the session needs; replaceable in tests. */
export interface AudioDeps {
  createContext(sampleRate?: number): AudioContext;
  createWorkletNode(ctx: AudioContext): AudioWorkletNode;
  nowMs(): number;
}

export const REAL_DEPS: AudioDeps = {
  createContext: (sampleRate) => (sampleRate ? new AudioContext({ sampleRate, latencyHint: 'interactive' }) : new AudioContext()),
  createWorkletNode: (ctx) => new AudioWorkletNode(ctx, 'pcm-capture', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 }),
  nowMs: () => performance.now(),
};

export class AudioSession {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private resampler: Resampler | null = null;
  private readonly env = new EnvelopeStream(AUDIO_PARAMS);
  private readonly breath = new BreathSoundRate(AUDIO_PARAMS);
  private readonly cough = new LabelEventTracker(AUDIO_PARAMS.cough);
  private readonly snoring = new LabelEventTracker(EVENT_PARAMS.snoring, 500, frameSeconds(AUDIO_PARAMS));
  private readonly talking = new LabelEventTracker(EVENT_PARAMS.speech, 500, frameSeconds(AUDIO_PARAMS));
  private readonly distress = DISTRESS_SECTIONS.map((p) => ({ p, tracker: new LabelEventTracker(p, 100, frameSeconds(AUDIO_PARAMS)) }));
  private readonly pause = new PauseTracker(SLEEP_PARAMS, AUDIO_PARAMS.fs_hz / AUDIO_PARAMS.envelope.hop);
  private yamnet: YamnetHandle | null = null;
  private ended = false;
  private readonly ring = new Float32Array(AUDIO_PARAMS.yamnet.window_samples);
  private readonly win = new Float32Array(AUDIO_PARAMS.yamnet.window_samples);
  private ringPos = 0;
  private ringFill = 0;
  private sinceClassify = 0;
  private total = 0; // 16 kHz samples seen
  private lastBlockMs = 0;
  private watchdog: ReturnType<typeof setInterval> | null = null;
  private dead = false;

  /** `offsetS`: session time at which this audio session starts. The W10 sound trackers (snoring, talking, distress, pauses) run on session
   *  time, so their per-hour buckets stay right across audio restarts; the breathing and cough results keep stream time. */
  constructor(private readonly hooks: AudioHooks, private readonly deps: AudioDeps = REAL_DEPS, private readonly offsetS = 0) {}

  get isEnded(): boolean { return this.ended; }

  /**
   * Opens the worklet on the stream's audio track. Resolves true when capture is running, false when the session was ended
   * meanwhile (nothing is left open then); throws if the track or worklet is unavailable. Firefox cannot connect a 16 kHz
   * context to a stream at another rate: that case retries with a default-rate context and the Resampler.
   */
  async start(stream: MediaStream): Promise<boolean> {
    this.hooks.onNoMic(false);
    const track = stream.getAudioTracks()[0];
    if (!track) throw new Error('no audio track');
    if (this.ended) return false;
    try {
      let ok: boolean;
      try {
        ok = await this.attach(this.deps.createContext(AUDIO_PARAMS.fs_hz), track);
      } catch (err) {
        if (this.ended) return false;
        if ((err as { name?: string })?.name !== 'NotSupportedError') throw err;
        this.teardownGraph(); // the 16 kHz context cannot take this stream: retry at the default rate
        ok = await this.attach(this.deps.createContext(), track);
      }
      if (!ok || this.ended) { this.teardownGraph(); return false; }
    } catch (err) {
      this.teardownGraph();
      throw err;
    }
    this.lastBlockMs = this.deps.nowMs();
    this.watchdog = setInterval(() => {
      // no blocks for 1.5 s, or the track ended: report "no microphone signal" and restart the estimator when it returns
      const dead = this.deps.nowMs() - this.lastBlockMs > 1500 || track.readyState === 'ended';
      if (dead !== this.dead) {
        this.dead = dead;
        if (dead) { this.breath.reset(); this.pause.breakStream(); }
        this.hooks.onNoMic(dead);
      }
    }, 500);
    return true;
  }

  /** Builds the graph on `ctx`; false if the session ended during an await. Throws on failure. */
  private async attach(ctx: AudioContext, track: MediaStreamTrack): Promise<boolean> {
    this.ctx = ctx;
    await ctx.audioWorklet.addModule(`${BASE}worklets/pcm-capture.js`);
    if (this.ended) return false;
    if (ctx.state === 'suspended') await ctx.resume().catch(() => undefined);
    if (this.ended) return false;
    this.resampler = ctx.sampleRate !== AUDIO_PARAMS.fs_hz ? new Resampler(ctx.sampleRate, AUDIO_PARAMS.fs_hz) : null;
    const node = this.deps.createWorkletNode(ctx);
    this.node = node;
    node.port.onmessage = (e: MessageEvent<Float32Array>) => this.onBlock(e.data);
    const mute = ctx.createGain();
    mute.gain.value = 0; // keep the graph pulling without any audible output
    this.source = ctx.createMediaStreamSource(new MediaStream([track])); // throws NotSupportedError on a rate mismatch (Firefox)
    this.source.connect(node);
    node.connect(mute).connect(ctx.destination);
    return true;
  }

  private teardownGraph(): void {
    try { this.node?.port.close(); } catch { /* already closed */ }
    try { this.source?.disconnect(); } catch { /* not connected */ }
    try { this.node?.disconnect(); } catch { /* not connected */ }
    void this.ctx?.close().catch(() => undefined);
    this.ctx = this.node = this.source = null;
    this.resampler = null;
  }

  async loadYamnet(loader: () => Promise<YamnetHandle>): Promise<void> {
    if (this.ended) return;
    this.hooks.onYamnet('loading');
    try {
      const h = await loader();
      if (this.ended) { h.close(); return; }
      this.yamnet = h;
      this.hooks.onYamnet('ready');
    } catch (err) {
      if (!this.ended) this.hooks.onYamnet('failed', err instanceof Error ? err.message : String(err));
    }
  }

  private onBlock(raw: Float32Array): void {
    if (this.ended) return;
    this.lastBlockMs = this.deps.nowMs();
    const x = this.resampler ? this.resampler.push(raw) : raw;
    if (x.length === 0) return;
    let s = 0, pk = 0;
    for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]!); s += a * a; if (a > pk) pk = a; }
    this.hooks.onLevel(Math.sqrt(s / x.length), pk);
    for (const f of this.env.push(x)) {
      const u = this.breath.push(f.t, f.env, f.clip);
      if (u) this.hooks.onBreath(u);
      if (this.yamnet) this.pause.pushFrame(this.offsetS + f.t, f.env); // without the sound model speech cannot be told apart from breathing: no pause counting
    }
    this.feedYamnet(x);
  }

  private feedYamnet(x: Float32Array): void {
    const y = AUDIO_PARAMS.yamnet, fs = AUDIO_PARAMS.fs_hz, n = y.window_samples;
    for (let i = 0; i < x.length; i++) {
      this.ring[this.ringPos] = x[i]!; // circular buffer of the last window_samples samples
      this.ringPos = (this.ringPos + 1) % n;
      if (this.ringFill < n) this.ringFill++;
      this.total++;
      if (this.ringFill === n && ++this.sinceClassify >= y.hop_samples) {
        this.sinceClassify = 0;
        this.win.set(this.ring.subarray(this.ringPos), 0);
        this.win.set(this.ring.subarray(0, this.ringPos), n - this.ringPos);
        this.classifyNow(this.total / fs);
      }
    }
  }

  private classifyNow(tEnd: number): void {
    const h = this.yamnet;
    if (!h) return;
    try {
      const labels = [AUDIO_PARAMS.cough.label, AUDIO_PARAMS.yamnet.speech_label, EVENT_PARAMS.snoring.label, ...DISTRESS_SECTIONS.map((p) => p.label)];
      const sc = h.classify(this.win, labels, tEnd);
      const speech = sc[AUDIO_PARAMS.yamnet.speech_label]!;
      this.breath.pushSpeech(tEnd, speech);
      const ts = this.offsetS + tEnd;
      this.pause.pushSpeech(ts, speech);
      this.cough.push(tEnd, sc[AUDIO_PARAMS.cough.label]!);
      this.snoring.push(ts, sc[EVENT_PARAMS.snoring.label] ?? 0);
      this.talking.push(ts, speech);
      for (const d of this.distress) d.tracker.push(ts, sc[d.p.label] ?? 0);
      this.hooks.onCough({ ...this.cough.view(), t: tEnd });
      this.hooks.onSounds?.({
        t: ts, snoring: this.snoring.view(), talking: this.talking.view(),
        distress: this.distress.map((d) => ({ label: d.p.label, view: d.tracker.view() })), pauses: this.pause.view(),
      });
    } catch (err) {
      this.yamnet = null;
      this.pause.breakStream();
      try { h.close(); } catch { /* already closed */ }
      this.hooks.onYamnet('failed', err instanceof Error ? err.message : String(err));
    }
  }

  end(): void {
    this.ended = true;
    if (this.watchdog) clearInterval(this.watchdog);
    this.watchdog = null;
    this.teardownGraph();
    this.yamnet?.close();
    this.yamnet = null;
  }
}
