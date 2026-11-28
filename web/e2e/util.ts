import { existsSync } from 'node:fs';
import { expect, test as base, type LaunchOptions } from '@playwright/test';

/** Chromium: $PW_CHROMIUM, else the pre-installed /opt/pw-browsers/chromium when present, else Playwright's own download. */
export const executablePath = process.env.PW_CHROMIUM ?? (existsSync('/opt/pw-browsers/chromium') ? '/opt/pw-browsers/chromium' : undefined);

export function launch(...extra: string[]): LaunchOptions {
  return {
    ...(executablePath ? { executablePath } : {}),
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required', ...extra],
  };
}

/**
 * Every test records all network requests and, at the end, asserts the privacy promise: every request is same-origin
 * (data: and blob: are local), and none carries a body (no upload of media or anything else).
 */
export const test = base.extend<{ network: { url: string; method: string; hasBody: boolean }[] }>({
  network: [
    async ({ context, baseURL }, use) => {
      const seen: { url: string; method: string; hasBody: boolean }[] = [];
      context.on('request', (r) => seen.push({ url: r.url(), method: r.method(), hasBody: r.postDataBuffer() !== null }));
      await use(seen);
      const origin = new URL(baseURL!).origin;
      expect(seen.length, 'requests were recorded').toBeGreaterThan(0);
      const foreign = seen.filter((r) => !/^(data|blob):/.test(r.url) && new URL(r.url).origin !== origin);
      expect(foreign, 'third-party requests').toEqual([]);
      expect(seen.filter((r) => r.method !== 'GET' && r.method !== 'HEAD'), 'non-GET requests').toEqual([]);
      expect(seen.filter((r) => r.hasBody), 'requests with a body').toEqual([]);
    },
    { auto: true },
  ],
});
export { expect };
