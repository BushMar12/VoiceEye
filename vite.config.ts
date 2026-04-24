import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// PWA/service-worker is intentionally disabled right now. The generated SW was
// caching the old 24 MB WASM on phones and intercepting fetches, which kept
// OOM-crashing iOS Safari on load. Re-enable once the app is stable on mobile.
// https://vite.dev/config/
export default defineConfig({
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  plugins: [
    react(),
    basicSsl(),
  ],
  server: {
    watch: {
      ignored: ['**/data/**', '**/runs/**', '**/venv/**', '**/venv_win/**']
    },
    proxy: {
      '/api/ollama': {
        target: 'http://127.0.0.1:11434',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api\/ollama/, ''),
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.removeHeader('Origin');
            proxyReq.removeHeader('Referer');
          });
        }
      },
    },
  },
})
