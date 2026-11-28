import { defineConfig } from '@playwright/test';

// End-to-end tests run against `npm run build:e2e` output (dist-e2e) served by `vite preview`.
// Chromium comes from $PW_CHROMIUM, the pre-installed /opt/pw-browsers/chromium, or `npx playwright install chromium` (CI).
export default defineConfig({
  testDir: 'e2e',
  testMatch: '**/*.spec.ts',
  timeout: 240_000,
  workers: 1,
  fullyParallel: false,
  reporter: 'list',
  use: { baseURL: 'http://localhost:4173', trace: 'off', screenshot: 'off', video: 'off' },
  webServer: {
    command: 'npx vite preview --mode e2e --outDir dist-e2e --host localhost --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
