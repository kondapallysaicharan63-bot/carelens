export const DISCLAIMER =
  'Not a medical device. For demonstration and wellbeing trends only. Do not use for diagnosis or emergencies; call emergency services if you feel unwell.';

export interface ConsentState {
  camera: boolean;
  mic: boolean;
  started: boolean;
}

/** Consent state machine. No capture may open until `start()` succeeds (user gesture). */
export class Consent {
  state: ConsentState = { camera: true, mic: false, started: false };

  setCamera(on: boolean): void {
    if (!this.state.started) this.state.camera = on;
  }
  setMic(on: boolean): void {
    if (!this.state.started) this.state.mic = on;
  }
  canStart(): boolean {
    return !this.state.started && (this.state.camera || this.state.mic);
  }
  start(): boolean {
    if (!this.canStart()) return false;
    this.state.started = true;
    return true;
  }
  /** Stop: returns to the consent screen; toggles can be changed again. */
  stop(): void {
    this.state.started = false;
  }
}
