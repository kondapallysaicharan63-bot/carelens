// AudioWorklet: runs on the audio thread, mixes the input to mono and posts ~2048-sample blocks to the page.
// Served from our own origin (CSP script-src 'self'). Audio never leaves the page. Not a medical device.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(2048);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0];
    if (!ch || ch.length === 0) return true;
    const frames = ch[0].length;
    for (let i = 0; i < frames; i++) {
      let s = 0;
      for (let c = 0; c < ch.length; c++) s += ch[c][i];
      this.buf[this.n++] = s / ch.length;
      if (this.n === this.buf.length) {
        const out = this.buf;
        this.port.postMessage(out, [out.buffer]);
        this.buf = new Float32Array(2048);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
