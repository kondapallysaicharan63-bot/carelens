// The only persistence in the app: remembered theme and device choice. Nothing else is stored.
export function safeGet(key: string): string | null {
  try {
    return localStorage.getItem('carelens.' + key);
  } catch {
    return null;
  }
}
export function safeSet(key: string, value: string): void {
  try {
    localStorage.setItem('carelens.' + key, value);
  } catch {
    /* storage unavailable (private mode / blocked): ignore */
  }
}
export function safeRemove(key: string): void {
  try {
    localStorage.removeItem('carelens.' + key);
  } catch {
    /* ignore */
  }
}
