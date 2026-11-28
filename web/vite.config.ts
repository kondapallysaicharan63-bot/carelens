import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

/** Dev only: Vite's HMR client injects <style> tags, so allow inline styles there. */
function devInlineStyles(): Plugin {
  return {
    name: 'carelens-dev-inline-styles',
    apply: 'serve',
    transformIndexHtml: (html) => html.replace("style-src 'self'", "style-src 'self' 'unsafe-inline'"),
  };
}

// Pages serves under /carelens/ -> CARELENS_BASE=/carelens/ npm run build
export default defineConfig(({ mode }) => ({
  base: process.env.CARELENS_BASE ?? '/',
  // `vite build --mode e2e` (npm run build:e2e) adds the scripted test pose provider; the e2e server is plain http on localhost
  // (a secure context), so the self-signed certificate is left out there. Production builds always get VITE_E2E = '0'.
  define: { 'import.meta.env.VITE_E2E': JSON.stringify(mode === 'e2e' ? '1' : '0') },
  plugins: [...(mode === 'e2e' ? [] : [basicSsl()]), devInlineStyles()],
  // fs.allow: this directory plus shared/ only (params.json is imported from there); nothing else in the repo is served
  server: { host: true, fs: { allow: [fileURLToPath(new URL('.', import.meta.url)), fileURLToPath(new URL('../shared', import.meta.url))] } },
  preview: { host: true },
  build: { target: 'es2022', sourcemap: false },
}));
