// Holds the other per-frame sessions (pose, face) while the hand test runs, INCLUDING sessions that are created after the test
// started (the test card is usable before the camera has finished opening). The hand test needs > 25 distinct frames/s, so nothing
// else may run inference or canvas readbacks per frame meanwhile.
export interface Pausable { setPaused(paused: boolean): void }

export class PauseGroup {
  private held = false;
  private readonly members = new Set<Pausable>();

  get isHeld(): boolean { return this.held; }

  /** Pause (true) or resume (false) every member, and every member added later while held. */
  hold(held: boolean): void {
    this.held = held;
    for (const m of this.members) m.setPaused(held);
  }

  add(m: Pausable): void {
    this.members.add(m);
    if (this.held) m.setPaused(true);
  }

  remove(m: Pausable): void { this.members.delete(m); }
}
