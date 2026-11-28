// Night mode keeps the screen on with the Screen Wake Lock API where the browser has it, and says so honestly where it does not.
// Nothing is sent anywhere. Not a medical device.
export type WakeStatus = 'idle' | 'active' | 'unsupported' | 'denied';

export const WAKE_TEXT: Record<WakeStatus, string> = {
  idle: 'Screen wake lock not started.',
  active: 'Keeping the screen awake while this page is open.',
  unsupported: 'This browser can’t keep the screen awake — switch off auto-lock in your phone settings so the screen stays on.',
  denied: 'The browser would not keep the screen awake — switch off auto-lock in your phone settings so the screen stays on.',
};
export const PLUG_TEXT = 'Keep the phone plugged in';

interface Sentinel { release(): Promise<void>; addEventListener?(type: 'release', cb: () => void): void }
export interface WakeNav { wakeLock?: { request(type: 'screen'): Promise<Sentinel> } }
export interface WakeDoc { visibilityState?: string; addEventListener(type: 'visibilitychange', cb: () => void): void; removeEventListener(type: 'visibilitychange', cb: () => void): void }

export class WakeLockKeeper {
  private sentinel: Sentinel | null = null;
  private wanted = false;
  status: WakeStatus = 'idle';
  private readonly onVis = (): void => { if (this.wanted && this.doc.visibilityState === 'visible' && !this.sentinel) void this.acquire(); };

  constructor(private readonly nav: WakeNav, private readonly doc: WakeDoc, private readonly onStatus: (s: WakeStatus) => void = () => undefined) {}

  private set(s: WakeStatus): void { this.status = s; this.onStatus(s); }

  private inflight: Promise<void> | null = null;

  /** One request at a time: a visibility change or a second start() while one is pending joins it instead of asking again. */
  private acquire(): Promise<void> {
    this.inflight ??= this.request().finally(() => { this.inflight = null; });
    return this.inflight;
  }

  private async request(): Promise<void> {
    if (!this.nav.wakeLock) { this.set('unsupported'); return; }
    try {
      const s = await this.nav.wakeLock.request('screen');
      if (!this.wanted) { void s.release().catch(() => undefined); return; } // stop() came while the request was pending
      const old = this.sentinel;
      this.sentinel = s;
      if (old) void old.release().catch(() => undefined); // never leave an earlier lock unreleased
      s.addEventListener?.('release', () => { if (this.sentinel === s) { this.sentinel = null; if (this.wanted) this.set('idle'); } });
      this.set('active');
    } catch {
      this.sentinel = null;
      this.set('denied');
    }
  }

  /** Ask for the lock (needs a visible page); the lock is asked for again when the page becomes visible. Returns the resulting status. */
  async start(): Promise<WakeStatus> {
    this.wanted = true;
    this.doc.addEventListener('visibilitychange', this.onVis);
    await this.acquire();
    return this.status;
  }

  stop(): void {
    this.wanted = false;
    this.doc.removeEventListener('visibilitychange', this.onVis);
    const s = this.sentinel;
    this.sentinel = null;
    if (s) void s.release().catch(() => undefined);
    this.status = 'idle';
  }
}
