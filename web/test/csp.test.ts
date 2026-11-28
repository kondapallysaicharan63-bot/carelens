import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const candidates: [string, string][] = [
  ['source', 'index.html'],
  ['built', 'dist/index.html'],
];
const files = candidates.filter(([, f]) => existsSync(f));

describe.each(files)('%s index.html privacy', (_n, file) => {
  const html = readFileSync(file, 'utf8');
  const csp = /http-equiv="Content-Security-Policy"\s+content="([^"]+)"/.exec(html)?.[1] ?? '';
  it('has a CSP meta tag with the required directives', () => {
    expect(csp).toContain("default-src 'self'");
    expect(csp).toContain("connect-src 'self'");
    expect(csp).toContain("script-src 'self' 'wasm-unsafe-eval'");
    expect(csp).toContain("worker-src 'self' blob:");
    expect(csp).not.toMatch(/https?:/);
    expect(csp).not.toContain("'unsafe-eval'");
  });
  it('references no external http(s) URLs', () => {
    expect(html.match(/(?:src|href)="(?:https?:)?\/\//g)).toBeNull();
    expect(html.replace(/xmlns="[^"]*"/g, '')).not.toMatch(/https?:\/\//);
  });
});

describe('index.html presence', () => {
  it('built dist/index.html exists in CI (run `npm run build` before `npm test`)', () => {
    if (process.env.CI) expect(existsSync('dist/index.html')).toBe(true);
  });
  it('source index.html exists', () => expect(files.length).toBeGreaterThan(0));
});
