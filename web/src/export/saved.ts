// Opt-in "save summary on this device": IndexedDB, one small summary per session (no raw estimates, no media).
// Nothing is sent anywhere. "Delete all saved data" removes the database and every carelens.* localStorage preference.
import type { SessionSummary } from './recorder';

const DB_NAME = 'carelens';
const STORE = 'summaries';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB unavailable'));
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => { req.result.createObjectStore(STORE, { keyPath: 'id' }); };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB error'));
  });
}

async function run<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const req = fn(db.transaction(STORE, mode).objectStore(STORE));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error('IndexedDB error'));
    });
  } finally {
    db.close();
  }
}

export const saveSummary = (s: SessionSummary): Promise<IDBValidKey> => run('readwrite', (st) => st.put(s));

/** newest first */
export async function listSummaries(): Promise<SessionSummary[]> {
  const all = await run<SessionSummary[]>('readonly', (st) => st.getAll());
  return all.sort((a, b) => (a.id < b.id ? 1 : -1));
}

export function clearLocalPrefs(storage: Pick<Storage, 'length' | 'key' | 'removeItem'> | null = typeof localStorage === 'undefined' ? null : localStorage): void {
  try {
    if (!storage) return;
    const keys: string[] = [];
    for (let i = 0; i < storage.length; i++) { const k = storage.key(i); if (k && k.startsWith('carelens.')) keys.push(k); }
    keys.forEach((k) => storage.removeItem(k));
  } catch { /* storage blocked: nothing to clear */ }
}

/** Deletes the IndexedDB database and the carelens.* localStorage keys. */
export async function deleteAllSavedData(): Promise<void> {
  clearLocalPrefs();
  if (typeof indexedDB === 'undefined') return;
  await new Promise<void>((resolve, reject) => {
    const req = indexedDB.deleteDatabase(DB_NAME);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error ?? new Error('could not delete'));
    req.onblocked = () => resolve(); // closes when the other connection does
  });
}
