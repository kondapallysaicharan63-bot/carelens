import { readFileSync } from 'node:fs';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { blinkScript, writeGreyY4m } from './media';

// Blinks & eyes-closed time (W9). A scripted face model (e2e build only) replays a synthetic eye-aspect-ratio trace with a known blink
// every 4 s (15 per minute). Needs about 20 distinct camera frames per second (the fake camera sends 30): do not run under heavy CPU load.
// The card calibrates for 20 s, then reports over the last minute; the test waits until a full 60 s window has been counted.
const video = writeGreyY4m('grey30.y4m', 4, 30);
test.use({ launchOptions: launch(`--use-file-for-fake-video-capture=${video}`) });

test('blink card: a scripted 15 per minute pattern is reported within 1 per minute and exported', async ({ page }) => {
  const script = blinkScript(15);
  await page.addInitScript((s) => { (window as unknown as { __E2E_FACE__: unknown }).__E2E_FACE__ = s; }, script);
  await page.goto('/');
  await page.locator('#start').click();

  const card = page.locator('#blink-card');
  await expect(card).toContainText('Blinks & eyes-closed time');
  await expect(card.locator('.tag')).toHaveText('Experimental');
  await expect(page.locator('#blink-note')).toHaveText('Experimental. Screens, reading, dry air and light all change blink rate. Compare with your own earlier results.');
  await expect(page.locator('#blink-guidance')).toContainText('Blink measurement is off');
  await expect(page.locator('#blink-value')).toHaveAttribute('data-reason', 'idle');

  // the face model is shared: it runs for the blink card even with face heart rate switched off
  await page.locator('#face-active-toggle').uncheck();
  await page.locator('#blink-active-toggle').check();
  const value = page.locator('#blink-value');
  await expect(page.locator('#blink-guidance')).toContainText(/Learning your open-eye level|Measuring over the last minute/, { timeout: 60_000 });
  const t0 = Date.now();
  let window_ = 0;
  while (Date.now() - t0 < 150_000 && window_ < 59.9) {
    window_ = Number((await value.getAttribute('data-window')) ?? 0);
    await page.waitForTimeout(1000);
  }
  await expect(value).toHaveAttribute('data-reason', 'ok');
  const bpm = Number(await value.getAttribute('data-bpm'));
  console.log(`blink: scripted 15/min, card says "${(await value.textContent())?.trim()}" (${bpm.toFixed(2)}) over ${window_.toFixed(1)} s; eyes closed ${await value.getAttribute('data-closed-pct')} %`);
  expect(Math.abs(bpm - 15)).toBeLessThanOrEqual(1);
  await expect(value).toHaveText(/^\d+ blinks per minute$/);
  await expect(page.locator('#blink-closed')).toContainText('Eyes closed:');
  expect(Number(await value.getAttribute('data-long'))).toBe(0);
  expect(Number(await value.getAttribute('data-closed-pct'))).toBeGreaterThan(1);

  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  const session = JSON.parse(readFileSync((await dl.path())!, 'utf8')) as SessionExport;
  expect(validateSession(session)).toEqual([]);
  const rows = (session.summaries ?? []).filter((r) => r.kind === 'blink');
  expect(rows.length).toBe(1);
  expect(rows[0]).toMatchObject({ reported: true, reason: 'ok', long_closures: 0 });
  expect(Math.abs(rows[0]!.blinks_per_min! - 15)).toBeLessThanOrEqual(1.5);
  expect(rows[0]!.eyes_closed_pct!).toBeGreaterThan(1);
});
