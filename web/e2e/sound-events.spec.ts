import { readFileSync } from 'node:fs';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { fallLandmarks, writeGreyY4m } from './media';

// The sound model is scripted: the test raises a YAMNet label for a while (window.__E2E_YAMNET__.active). Camera: grey clip + scripted pose.
const video = writeGreyY4m();
test.use({ launchOptions: launch(`--use-file-for-fake-video-capture=${video}`) });
test.setTimeout(240_000);

const sound = (page: import('@playwright/test').Page, scores: Record<string, number>) =>
  page.evaluate((s) => { (window as unknown as { __E2E_YAMNET__: { active: Record<string, number> } }).__E2E_YAMNET__.active = s; }, scores);
const script = (page: import('@playwright/test').Page) =>
  page.addInitScript(() => { (window as unknown as { __E2E_YAMNET__: unknown }).__E2E_YAMNET__ = { active: {} }; });
const download = async (page: import('@playwright/test').Page): Promise<SessionExport> => {
  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  return JSON.parse(readFileSync((await dl.path())!, 'utf8')) as SessionExport;
};

test('night mode, microphone only: scripted scream raises the distress banner; scripted speech adds talking time; snoring counts', async ({ page }) => {
  await script(page);
  await page.goto('/');
  await page.locator('#consent-mic').uncheck();
  await page.locator('#consent-camera').check();
  await page.locator('#consent-night').check();
  await expect(page.locator('#consent-camera')).not.toBeChecked();
  await expect(page.locator('#consent-mic')).toBeChecked();
  await expect(page.locator('#consent-mic')).toBeDisabled(); // night mode needs the microphone
  await page.locator('#consent-night').uncheck(); // the earlier choices come back
  await expect(page.locator('#consent-camera')).toBeChecked();
  await expect(page.locator('#consent-mic')).not.toBeChecked();
  await expect(page.locator('#consent-mic')).toBeEnabled();
  await page.locator('#consent-night').check();
  await page.locator('#consent-camera').uncheck(); // microphone only
  await page.locator('#start').click();

  await expect(page.locator('html')).toHaveAttribute('data-night', 'on');
  await expect(page.locator('#night-plug')).toHaveText('Keep the phone plugged in');
  await expect(page.locator('#night-wake')).toHaveText(/keeping the screen awake|can’t keep the screen awake|would not keep the screen awake/i);
  console.log(`night: wake status "${await page.locator('#night-wake').textContent()}"`);
  await expect(page.locator('#cough-status')).toContainText('Listening on this device', { timeout: 60_000 });
  await expect(page.locator('#pause-note')).toBeHidden(); // minimal view: the cards are hidden until "Show all cards"

  await sound(page, { Screaming: 0.9 });
  await page.waitForTimeout(2500);
  await sound(page, {});
  const banner = page.locator('#checkin-banner');
  await expect(banner).toBeVisible();
  await expect(page.locator('#banner-text')).toHaveText('Possible distress sound — check-in suggested');
  await page.locator('#banner-ack').click();
  await expect(banner).toBeHidden();

  await sound(page, { Speech: 0.9 });
  await page.waitForTimeout(8000);
  await sound(page, { Snoring: 0.8 });
  await page.waitForTimeout(6000);
  await sound(page, {});
  await expect(page.locator('#night-summary')).toContainText('Snoring sounds: 1');
  const talkS = Number(await page.locator('#talk-minutes').getAttribute('data-seconds'));
  console.log(`night: night-summary "${await page.locator('#night-summary').textContent()}"; talking ${talkS} s; snoring ${await page.locator('#snore-count').textContent()}`);
  expect(talkS).toBeGreaterThan(0);
  await expect(page.locator('#talk-minutes')).not.toHaveText('Talking time: 0.0 min');
  await page.locator('#night-details').click(); // all cards back
  await expect(page.locator('#snore-count')).toBeVisible();
  await expect(page.locator('#snore-count')).toHaveAttribute('data-count', '1');

  await page.waitForTimeout(6000);
  const session = await download(page);
  expect(validateSession(session)).toEqual([]);
  expect(session.modes).toEqual(['microphone']);
  const d = session.events.filter((e) => e.kind === 'distress_sound');
  expect(d).toHaveLength(1);
  expect(d[0]).toMatchObject({ trigger: 'Screaming' });
  const talking = (session.summaries ?? []).find((s) => s.kind === 'talking');
  const snoring = (session.summaries ?? []).find((s) => s.kind === 'snoring');
  expect(talking!.minutes!).toBeGreaterThan(0);
  expect(snoring).toMatchObject({ reported: true, count: 1 });
  await page.locator('button', { hasText: 'Stop' }).click();
  await expect(page.locator('#last-sound-summary')).toContainText('Snoring sounds: 1');
  await expect(page.locator('html')).not.toHaveAttribute('data-night', 'on');
});

const startFall = async (page: import('@playwright/test').Page) => {
  await script(page);
  await page.addInitScript((s) => { (window as unknown as { __E2E_POSE__: unknown }).__E2E_POSE__ = s; }, fallLandmarks());
  await page.goto('/');
  await page.locator('#consent-mic').check();
  await page.locator('#start').click();
};
const CUE = 'Possible fall — with a sound cue — check-in suggested';

test('scripted thump BEFORE the fall confirms: its banner turns into the single fall check-in "with a sound cue"; no second banner', async ({ page }) => {
  await startFall(page);
  await expect(page.locator('#cough-status')).toContainText('Listening on this device', { timeout: 60_000 });
  await sound(page, { 'Thump, thud': 0.9 });
  await page.waitForTimeout(2500);
  await sound(page, {});
  await expect(page.locator('#banner-text')).toHaveText('Possible distress sound — check-in suggested');
  await expect(page.locator('#banner-text')).toHaveText(CUE, { timeout: 120_000 }); // replaced in place when the fall confirms
  await expect(page.locator('#event-list li')).toHaveCount(1);
  console.log(`cue: thump first, banner "${(await page.locator('#banner-text').textContent())?.trim()}"`);
  await page.locator('#banner-ack').click();
  await expect(page.locator('#checkin-banner')).toBeHidden(); // acknowledging it clears everything

  const session = await download(page);
  expect(validateSession(session)).toEqual([]);
  const falls = session.events.filter((e) => e.kind === 'possible_fall');
  expect(falls).toHaveLength(1);
  expect(falls[0]!.sound_cue).toBe(true);
  const thumps = session.events.filter((e) => e.kind === 'distress_sound' && e.trigger === 'Thump, thud');
  expect(thumps).toHaveLength(1);
  expect(thumps[0]!.merged_into_fall).toBe(true);
});

test('possible fall, then a scripted thump 4 s after the check-in: still ONE check-in, upgraded "with a sound cue"', async ({ page }) => {
  await startFall(page);
  const banner = page.locator('#checkin-banner');
  await expect(banner).toBeVisible({ timeout: 120_000 });
  await expect(page.locator('#banner-text')).toHaveText('Possible fall — check-in suggested');
  await expect(page.locator('#cough-status')).toContainText('Listening on this device', { timeout: 60_000 });
  await page.waitForTimeout(4000);
  await sound(page, { 'Thump, thud': 0.9 });
  await page.waitForTimeout(2500);
  await sound(page, {});
  await expect(page.locator('#banner-text')).toHaveText(CUE);
  await expect(page.locator('#event-list li')).toHaveCount(1);
  console.log(`cue: fall first, banner "${(await page.locator('#banner-text').textContent())?.trim()}"`);

  const session = await download(page);
  expect(validateSession(session)).toEqual([]);
  expect(session.events.filter((e) => e.kind === 'possible_fall')[0]!.sound_cue).toBe(true);
  expect(session.events.filter((e) => e.kind === 'distress_sound' && e.trigger === 'Thump, thud')[0]!.merged_into_fall).toBe(true);
});
