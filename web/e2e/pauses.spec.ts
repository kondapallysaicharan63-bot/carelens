import { readFileSync } from 'node:fs';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { writeBreathPauseWav } from './media';

// Synthetic breath noise at 14/min, 66 s of it, then a 15 s gap, then breathing again; the sound model is scripted (no speech).
const wav = writeBreathPauseWav('breath-14bpm-gap15.wav', 14, 66, 15, 30);
test.use({ launchOptions: launch(`--use-file-for-fake-audio-capture=${wav}`) });
test.setTimeout(300_000);

test('possible pauses in breathing sounds: synthetic breath WAV with one 15 s gap counts 1, per-hour figure withheld', async ({ page }) => {
  await page.addInitScript(() => { (window as unknown as { __E2E_YAMNET__: unknown }).__E2E_YAMNET__ = { active: {} }; });
  await page.goto('/');
  await page.locator('#consent-camera').uncheck();
  await page.locator('#consent-mic').check();
  await page.locator('#start').click();

  const count = page.locator('#pause-count');
  await expect(page.locator('#cough-status')).toContainText('Listening on this device', { timeout: 60_000 });
  await expect(count).toHaveText('Possible pauses this session: 1', { timeout: 170_000 });
  console.log(`pauses: "${await count.textContent()}"; "${(await page.locator('#pause-rate').textContent())?.trim()}"; ${await page.locator('#pause-lengths').textContent()}`);
  await expect(page.locator('#pause-rate')).toContainText('withheld until 30 min');
  await expect(page.locator('#pause-note')).toHaveText('Counts gaps in breathing sounds after regular breathing. Phone placement, bedding and movement all affect it. Not a sleep test — talk to a clinician if you’re worried about your sleep.');
  await expect(page.locator('#checkin-banner')).toBeHidden(); // a pause is a count, not an alert

  await page.waitForTimeout(6000); // the summary row refreshes every 5 s
  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  const session = JSON.parse(readFileSync((await dl.path())!, 'utf8')) as SessionExport;
  expect(validateSession(session)).toEqual([]);
  const pauses = session.events.filter((e) => e.kind === 'possible_pause');
  expect(pauses.length).toBeGreaterThanOrEqual(1);
  const dur = pauses[0]!.t_end! - pauses[0]!.t_onset!;
  console.log(`pauses: exported first pause ${dur.toFixed(1)} s`);
  expect(dur).toBeGreaterThan(13);
  expect(dur).toBeLessThan(21);
  const sum = (session.summaries ?? []).find((s) => s.kind === 'pauses');
  expect(sum).toMatchObject({ reported: true, reason: 'ok' });
  expect(sum!.count).toBeGreaterThanOrEqual(1);
  expect(sum!.rate_per_h).toBeUndefined();
});
