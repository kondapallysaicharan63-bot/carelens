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
export default defineConfig({
  base: process.env.CARELENS_BASE ?? '/',
  plugins: [basicSsl(), devInlineStyles()],
  // fs.allow: this directory plus shared/ only (params.json is imported from there); nothing else in the repo is served
  server: { host: true, fs: { allow: [fileURLToPath(new URL('.', import.meta.url)), fileURLToPath(new URL('../shared', import.meta.url))] } },
  preview: { host: true },
  build: { target: 'es2022', sourcemap: false },
});
