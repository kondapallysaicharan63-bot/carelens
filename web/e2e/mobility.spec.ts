import { readFileSync } from 'node:fs';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { chairStandScript, writeGreyY4m } from './media';

const video = writeGreyY4m();
test.use({ launchOptions: launch(`--use-file-for-fake-video-capture=${video}`) });

test('movement tests: scripted 30 s chair stand counts the scripted reps and exports the summary', async ({ page }) => {
  const script = chairStandScript();
  await page.addInitScript((s) => { (window as unknown as { __E2E_POSE__: unknown }).__E2E_POSE__ = s; }, script);
  await page.goto('/');
  await page.locator('#start').click();

  // pick -> safety screen first, explicit "I'm ready"
  await page.locator('#mv-pick-chair_stand_30s').click();
  await expect(page.locator('#mv-safety-text')).toContainText('Only do this if you can do it safely. Use a sturdy chair without wheels against a wall, have someone nearby, stop if you feel unwell or unsteady.');
  await page.locator('#mv-ready').click();

  // placement: live visibility check, then the 3-2-1 countdown
  await expect(page.locator('#mv-placement-text')).toContainText('2–3 m');
  await expect(page.locator('#mv-chk-ankles')).toContainText('in view', { timeout: 60_000 });
  await expect(page.locator('#mv-start')).toBeEnabled();
  await page.locator('#mv-start').click();
  await expect(page.locator('#mv-countdown')).toBeVisible();
  await expect(page.locator('#mv-live')).toBeVisible({ timeout: 20_000 }); // Go
  await page.evaluate(() => { (window as unknown as { __E2E_PLAY__: boolean }).__E2E_PLAY__ = true; }); // the figure starts moving at Go

  // live counter/timer, then the result card
  await expect(page.locator('#mv-timer')).toContainText('s');
  await expect(page.locator('#mv-result')).toBeVisible({ timeout: 90_000 });
  const value = page.locator('#mv-result-value');
  await expect(value).toHaveAttribute('data-reps', String(script.reps));
  await expect(value).toContainText(`${script.reps} full stands in 30 s`);
  await expect(page.locator('#mv-trend')).toHaveText('A wellbeing trend. Compare with your own earlier results, not with other people.');
  console.log(`mobility: scripted ${script.reps} reps, card says "${(await value.textContent())?.trim()}"`);

  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  const session = JSON.parse(readFileSync((await dl.path())!, 'utf8')) as SessionExport;
  expect(validateSession(session)).toEqual([]);
  const rows = (session.summaries ?? []).filter((r) => r.kind === 'chair_stand_30s');
  expect(rows.length).toBe(1);
  expect(rows[0]!.reported).toBe(true);
  expect(rows[0]!.reps).toBe(script.reps);
  expect(rows[0]!.rep_times_s?.length).toBe(script.reps);
});
