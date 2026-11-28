import { readFileSync } from 'node:fs';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { fallLandmarks, writeGreyY4m } from './media';

const video = writeGreyY4m();
test.use({ launchOptions: launch(`--use-file-for-fake-video-capture=${video}`) });

test('pose chain: scripted sudden fall raises the check-in banner and exports the event', async ({ page }) => {
  const script = fallLandmarks();
  await page.addInitScript((s) => { (window as unknown as { __E2E_POSE__: unknown }).__E2E_POSE__ = s; }, script);
  await page.goto('/');
  await page.locator('#start').click();

  const posture = page.locator('#posture-value');
  const banner = page.locator('#checkin-banner');
  const seen = new Set<string>();
  const t0 = Date.now();
  while (Date.now() - t0 < 90_000 && !(await banner.isVisible())) {
    seen.add(((await posture.textContent()) ?? '').trim());
    await page.waitForTimeout(400);
  }
  await expect(banner).toBeVisible();
  await expect(page.locator('#banner-text')).toContainText('Possible fall — check-in suggested');
  seen.add(((await posture.textContent()) ?? '').trim());
  console.log(`pose: posture labels seen ${JSON.stringify([...seen])}; banner after ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  expect([...seen].some((s) => /standing/.test(s))).toBe(true);
  expect([...seen].some((s) => /lying/.test(s))).toBe(true);

  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  const session = JSON.parse(readFileSync((await dl.path())!, 'utf8')) as SessionExport;
  expect(validateSession(session)).toEqual([]);
  expect(session.modes).toContain('camera');
  const falls = session.events.filter((e) => e.kind === 'possible_fall');
  expect(falls.length).toBe(1);
  expect(falls[0]!.t_onset).toBeLessThan(falls[0]!.t);
});
