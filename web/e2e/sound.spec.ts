import { readFileSync } from 'node:fs';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { writeBreathWav } from './media';

const BPM = 15;
const wav = writeBreathWav('breath-15bpm.wav', BPM);
test.use({ launchOptions: launch(`--use-file-for-fake-audio-capture=${wav}`) });

test('breathing from sound: synthetic 15/min breath noise, no coughs', async ({ page }) => {
  await page.goto('/');
  await page.locator('#consent-camera').uncheck();
  await page.locator('#consent-mic').check();
  await page.locator('#start').click();

  const val = page.locator('#brs-value');
  await expect(val).toHaveText(/^\d+ breaths\/min$/, { timeout: 150_000 });
  const reads: number[] = [];
  for (let i = 0; i < 6; i++) {
    const m = /^(\d+) breaths\/min$/.exec((await val.textContent()) ?? '');
    if (m) reads.push(Number(m[1]));
    await page.waitForTimeout(2000);
  }
  console.log(`sound: Breathing (sound) readings ${JSON.stringify(reads)} (truth ${BPM}); cough card: ${await page.locator('#cough-count').textContent()}; ${await page.locator('#cough-status').textContent()}`);
  expect(reads.length).toBeGreaterThanOrEqual(3);
  for (const v of reads) expect(Math.abs(v - BPM)).toBeLessThanOrEqual(2);
  await expect(page.locator('#cough-count')).toHaveText('Cough sounds this session: 0');
  await expect(page.locator('#cough-status')).toContainText('Listening on this device'); // the sound model really ran

  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  const session = JSON.parse(readFileSync((await dl.path())!, 'utf8')) as SessionExport;
  expect(validateSession(session)).toEqual([]);
  expect(session.modes).toEqual(['microphone']);
  expect(session.estimates.some((e) => e.metric === 'br_sound' && e.reported)).toBe(true);
  expect(session.events.filter((e) => e.kind === 'cough')).toEqual([]);
});
