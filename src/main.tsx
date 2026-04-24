import React, { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import ErrorBoundary from './components/ErrorBoundary.tsx'

// Unconditionally unregister any previously-installed service worker and clear
// all browser caches. We used to ship a PWA service worker, but it aggressively
// cached the old 24 MB WASM binary and kept intercepting fetches on iOS, which
// contributed to the tab's repeated OOM crashes. Running this on every load
// guarantees phones that previously visited the site recover after the next
// hard refresh.
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(regs => {
    regs.forEach(r => r.unregister());
  }).catch(() => {});
}
if ('caches' in window) {
  caches.keys().then(keys => keys.forEach(k => caches.delete(k))).catch(() => {});
}

// StrictMode double-mounts every effect which creates two camera streams, two
// YOLO model loads, and two speech-recognition sessions — fatal on iOS Safari.
// Keep it in dev only; production (including npm run phone) skips it.
const Root = import.meta.env.DEV
  ? ({ children }: { children: React.ReactNode }) => <StrictMode>{children}</StrictMode>
  : ({ children }: { children: React.ReactNode }) => <>{children}</>;

createRoot(document.getElementById('root')!).render(
  <Root>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </Root>,
)
