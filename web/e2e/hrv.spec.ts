import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { TMP, writeFingertipRrY4m } from './media';

// HRV (W6): a synthetic fingertip clip whose pulse follows a known RR series (true RMSSD 40 ms), held for > 60 s.
const RMSSD = 40;
const clip = writeFingertipRrY4m('fingertip-rr-40ms.y4m', RMSSD);
test.use({ launchOptions: launch(`--use-file-for-fake-video-capture=${clip.path}`) });

test('HRV card reports after 60 s of contact, close to the true RMSSD, and the export holds the summary', async ({ page }) => {
  await page.goto('/');
  await page.locator('#mode-fingertip').check();
  await page.locator('#start').click();
  const hrv = page.locator('#fp-hrv');
  await expect(hrv).toContainText('Hold steady for 60 s');
  await expect(hrv).toHaveText(/^RMSSD \d+ ms, SDNN \d+ ms$/, { timeout: 150_000 });
  const m = /^RMSSD (\d+) ms, SDNN (\d+) ms$/.exec((await hrv.textContent()) ?? '')!;
  const rmssd = Number(m[1]), sdnn = Number(m[2]);
  console.log(`hrv: card RMSSD ${rmssd} ms SDNN ${sdnn} ms (truth RMSSD ${clip.rmssd} ms, SDNN ${clip.sdnn.toFixed(1)} ms)`);
  expect(Math.abs(rmssd - clip.rmssd)).toBeLessThanOrEqual(10);
  expect(Math.abs(sdnn - clip.sdnn)).toBeLessThanOrEqual(10);
  await expect(page.locator('#fp-hrv-detail')).toContainText(/\d+ beats over (6\d|7\d) s/);
  await expect(page.locator('#fp-hrv-quality')).toContainText(/\d+ % of beats kept/);
  await expect(page.locator('#fp-hrv-card')).toContainText('Experimental. Short recordings at camera frame rates give rough values');
  await page.waitForTimeout(3000); // hold on: the summary row is updated in place, never duplicated

  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  const jsonPath = join(TMP, 'fingertip-hrv-session.json');
  await dl.saveAs(jsonPath);
  const session = JSON.parse(readFileSync(jsonPath, 'utf8')) as SessionExport;
  expect(validateSession(session)).toEqual([]);
  expect(session.summaries?.length, 'one HRV summary per contact epoch').toBe(1);
  const s = session.summaries![0]!;
  console.log(`hrv: export summary ${JSON.stringify(s)}`);
  expect(s.kind).toBe('hrv');
  expect(s.reported && s.reason === 'ok').toBe(true);
  expect(Math.abs(s.rmssd_ms! - clip.rmssd)).toBeLessThanOrEqual(10);
  expect(s.duration_s!).toBeGreaterThanOrEqual(60);
  expect(s.n_kept!).toBeGreaterThanOrEqual(50);
  expect(s.kept_frac!).toBeGreaterThanOrEqual(0.9);
  const py = process.env.PYTHON ?? 'python3';
  execFileSync(py, ['-c', 'import sys; from carelens.websession import load_session; load_session(sys.argv[1])', jsonPath], { cwd: join(import.meta.dirname, '..', '..'), stdio: 'pipe' }); // the Python reader accepts it
});
