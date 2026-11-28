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
  server: { host: true },
  preview: { host: true },
  build: { target: 'es2022', sourcemap: false },
});
