import { readFileSync } from 'node:fs';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { handScript, writeGreyY4m } from './media';

// Rhythmic hand movement (W8): a scripted hand model (e2e build only) replays synthetic landmarks with a known frequency.
const video = writeGreyY4m('grey30.y4m', 4, 30);
test.use({ launchOptions: launch(`--use-file-for-fake-video-capture=${video}`) });

async function runHandTest(page: import('@playwright/test').Page, script: ReturnType<typeof handScript>): Promise<void> {
  await page.addInitScript((s) => { (window as unknown as { __E2E_HAND__: unknown }).__E2E_HAND__ = s; }, script);
  await page.goto('/');
  await page.locator('#start').click();
  await page.locator('#tm-pick').click();
  await expect(page.locator('#tm-instructions')).toContainText('Hold your hand out in front of the camera, palm facing the camera, fingers spread, about 40–60 cm away, for 15 seconds. Rest your elbow if you need to.');
  await expect(page.locator('#tm-comfort')).toContainText('Rest your elbow');
  await page.locator('#tm-ready').click();
  await expect(page.locator('#tm-chk-hand')).toContainText('in view', { timeout: 60_000 });
  await expect(page.locator('#tm-start')).toBeEnabled();
  await page.locator('#tm-start').click();
  await expect(page.locator('#tm-live')).toBeVisible({ timeout: 20_000 }); // Go
  await page.evaluate(() => { (window as unknown as { __E2E_PLAY__: boolean }).__E2E_PLAY__ = true; }); // the movement starts at Go
  await expect(page.locator('#tm-result')).toBeVisible({ timeout: 60_000 });
}

async function exportedRows(page: import('@playwright/test').Page): Promise<SessionExport['summaries']> {
  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  const session = JSON.parse(readFileSync((await dl.path())!, 'utf8')) as SessionExport;
  expect(validateSession(session)).toEqual([]);
  return (session.summaries ?? []).filter((r) => r.kind === 'hand_tremor');
}

test('hand test: a scripted 6 Hz movement is reported within 0.3 Hz and exported', async ({ page }) => {
  await runHandTest(page, handScript(6, 2.5));
  const value = page.locator('#tm-result-value');
  await expect(value).toHaveAttribute('data-reason', 'ok');
  const hz = Number(await value.getAttribute('data-freq'));
  console.log(`tremor: scripted 6 Hz, card says "${(await value.textContent())?.trim()}"`);
  expect(Math.abs(hz - 6)).toBeLessThanOrEqual(0.3);
  await expect(value).toHaveText(/^Rhythmic hand movement: \d+\.\d Hz, amplitude \d+\.\d % of hand size$/);
  await expect(page.locator('#tm-note')).toHaveText('Experimental. Many things affect this — caffeine, tiredness, cold, holding position. Compare with your own earlier results.');
  const rows = await exportedRows(page);
  expect(rows!.length).toBe(1);
  expect(rows![0]!.reported).toBe(true);
  expect(Math.abs(rows![0]!.freq_hz! - 6)).toBeLessThanOrEqual(0.3);
  expect(rows![0]!.amp_pct!).toBeGreaterThan(1);
});

test('hand test: a steady hand gets the neutral no-rhythm text', async ({ page }) => {
  await runHandTest(page, handScript(0));
  await expect(page.locator('#tm-result-value')).toHaveText('No clear rhythmic hand movement detected');
  await expect(page.locator('#tm-result-value')).toHaveAttribute('data-reason', 'no_rhythm');
  const rows = await exportedRows(page);
  expect(rows!.length).toBe(1);
  expect(rows![0]).toMatchObject({ reported: false, reason: 'no_rhythm' });
});
