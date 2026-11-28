import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { validateSession, type SessionExport } from '../src/export/schema';
import { expect, launch, test } from './util';
import { TMP, writeFingertipY4m } from './media';

const BPM = 72;
const video = writeFingertipY4m('fingertip-72bpm.y4m', BPM);
test.use({ launchOptions: launch(`--use-file-for-fake-video-capture=${video}`) });

test('fingertip heart rate from a synthetic 72 bpm clip, export and accuracy round trip', async ({ page }) => {
  await page.goto('/');
  await page.locator('#mode-fingertip').check();
  await page.locator('#start').click();

  // the HRV card is present but inactive until 60 s of steady contact
  await expect(page.locator('#fp-hrv')).toContainText('Hold steady for 60 s');
  await expect(page.locator('#fp-hrv-card')).toContainText('Not a stress or health score');
  // wait for the first reported reading, then watch it for a few seconds
  const bpmEl = page.locator('#fp-bpm');
  await expect(bpmEl).toHaveText(/^\d+ bpm$/, { timeout: 90_000 });
  const reads: number[] = [];
  for (let i = 0; i < 8; i++) {
    const m = /^(\d+) bpm$/.exec((await bpmEl.textContent()) ?? '');
    if (m) reads.push(Number(m[1]));
    await page.waitForTimeout(1000);
  }
  console.log(`fingertip: HR card readings ${JSON.stringify(reads)} (truth ${BPM})`);
  expect(reads.length).toBeGreaterThanOrEqual(5);
  for (const v of reads) expect(Math.abs(v - BPM)).toBeLessThanOrEqual(3);

  // export (Blob download) -> schema-valid, reported hr_fingertip estimates
  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-json').click()]);
  const jsonPath = join(TMP, 'fingertip-session.json');
  await dl.saveAs(jsonPath);
  const session = JSON.parse(readFileSync(jsonPath, 'utf8')) as SessionExport;
  expect(validateSession(session)).toEqual([]);
  expect(session.modes).toEqual(['fingertip']);
  const rep = session.estimates.filter((e) => e.metric === 'hr_fingertip' && e.reported && e.value !== null);
  expect(rep.length).toBeGreaterThanOrEqual(5);
  expect(rep.every((e) => e.span_s !== undefined && e.span_s > 0 && e.span_s <= 10.001), 'span_s is the analysed window').toBe(true);
  expect(session.estimates.some((e) => e.value === null && !e.reported), 'withheld steps are recorded too').toBe(true);
  const med = rep.map((e) => e.value as number).sort((a, b) => a - b)[Math.floor(rep.length / 2)]!;
  console.log(`fingertip: export has ${rep.length} reported estimates, median ${med} bpm`);
  expect(Math.abs(med - BPM)).toBeLessThanOrEqual(3);

  // CSV download has the estimates table
  const [dlc] = await Promise.all([page.waitForEvent('download'), page.locator('#dl-csv').click()]);
  const csv = readFileSync((await dlc.path())!, 'utf8').trim().split('\n');
  expect(csv[0]).toBe('t,wall,metric,value,quality,reported,reason,span_s');
  expect(csv.length - 1).toBe(session.estimates.length);

  // round trip: python -m carelens.accuracy --web-session against a synthetic 72 bpm reference
  const startMs = Date.parse(session.started_at);
  const lines = ['timestamp,bpm'];
  for (let s = -60; s <= 180; s += 2) lines.push(`${new Date(startMs + s * 1000).toISOString()},${BPM}`);
  const refPath = join(TMP, 'ref-72.csv');
  writeFileSync(refPath, lines.join('\n') + '\n');
  const out = join(TMP, 'accuracy-out');
  mkdirSync(out, { recursive: true });
  const py = process.env.PYTHON ?? 'python3';
  execFileSync(py, ['-m', 'carelens.accuracy', '--web-session', jsonPath, '--metric', 'hr_fingertip', '--hr-ref', refPath, '--reference-name', 'synthetic 72 bpm', '--out', out], { cwd: join(import.meta.dirname, '..', '..'), stdio: 'pipe' });
  const m = JSON.parse(readFileSync(join(out, 'metrics.json'), 'utf8'));
  const s = m.heart_rate.summary;
  console.log(`accuracy round trip: n=${s.n} MAE=${s.mae?.toFixed(2)} bpm, coverage ${s.n_reported}/${s.n_windows}`);
  expect(s.n).toBeGreaterThanOrEqual(5);
  expect(s.mae).toBeLessThan(3);
  expect(s.n_windows, 'withheld (settling) rows count towards coverage').toBeGreaterThan(s.n_reported);
});
