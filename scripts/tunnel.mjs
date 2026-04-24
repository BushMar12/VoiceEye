#!/usr/bin/env node
// Start a Cloudflare Quick Tunnel that exposes the local Ollama server and
// write the public URL into .env.local as VITE_OLLAMA_URL. The tunnel stays
// attached to this process — Ctrl-C to stop it.
//
// Usage:
//   npm run tunnel               # foreground; streams cloudflared logs
//   npm run tunnel -- --deploy   # same, plus runs `npm run deploy` once URL is ready

import { spawn } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync, chmodSync, unlinkSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { arch, platform } from 'node:os';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..');
const BIN_DIR = resolve(REPO_ROOT, '.bin');
const CF_BIN = resolve(BIN_DIR, 'cloudflared');
const ENV_FILE = resolve(REPO_ROOT, '.env.local');
const OLLAMA_URL = 'http://127.0.0.1:11434';

const args = new Set(process.argv.slice(2));
const shouldDeploy = args.has('--deploy');

async function ensureCloudflared() {
  if (existsSync(CF_BIN)) return;
  mkdirSync(BIN_DIR, { recursive: true });
  const plat = platform();
  const a = arch();
  if (plat !== 'darwin') {
    console.error(`tunnel: this helper auto-installs cloudflared only on macOS. Install it manually and re-run: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/`);
    process.exit(1);
  }
  const asset = a === 'arm64' ? 'cloudflared-darwin-arm64.tgz' : 'cloudflared-darwin-amd64.tgz';
  const url = `https://github.com/cloudflare/cloudflared/releases/latest/download/${asset}`;
  // Follow redirect to resolve actual tag (the `latest/download/...` pattern
  // returns a 302 that curl/fetch both follow transparently).
  console.log(`tunnel: downloading cloudflared (${asset}) ...`);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) {
    console.error(`tunnel: download failed: HTTP ${res.status}`);
    process.exit(1);
  }
  // The asset is a .tgz containing a single `cloudflared` binary.
  const tgzPath = resolve(BIN_DIR, 'cloudflared.tgz');
  writeFileSync(tgzPath, Buffer.from(await res.arrayBuffer()));
  await new Promise((done, fail) => {
    const p = spawn('tar', ['-xzf', tgzPath, '-C', BIN_DIR], { stdio: 'inherit' });
    p.on('exit', (code) => code === 0 ? done() : fail(new Error(`tar exit ${code}`)));
  });
  chmodSync(CF_BIN, 0o755);
  try { unlinkSync(tgzPath); } catch {
    // ignore — cleanup is best-effort
  }
  console.log(`tunnel: cloudflared installed at ${CF_BIN}`);
}

function updateEnvFile(tunnelUrl) {
  const line = `VITE_OLLAMA_URL=${tunnelUrl}`;
  let body = '';
  if (existsSync(ENV_FILE)) {
    body = readFileSync(ENV_FILE, 'utf8');
    if (/^VITE_OLLAMA_URL=.*/m.test(body)) {
      body = body.replace(/^VITE_OLLAMA_URL=.*$/m, line);
    } else {
      if (!body.endsWith('\n')) body += '\n';
      body += `${line}\n`;
    }
  } else {
    body = `${line}\n`;
  }
  writeFileSync(ENV_FILE, body);
  console.log(`tunnel: wrote ${line} → ${ENV_FILE}`);
}

function runDeploy() {
  console.log('tunnel: running `npm run deploy`…');
  const p = spawn('npm', ['run', 'deploy'], { stdio: 'inherit', cwd: REPO_ROOT });
  p.on('exit', (code) => {
    if (code !== 0) console.error(`tunnel: deploy exited ${code}`);
  });
}

async function main() {
  await ensureCloudflared();

  console.log(`tunnel: starting quick tunnel → ${OLLAMA_URL}`);
  const child = spawn(
    CF_BIN,
    [
      'tunnel',
      '--no-autoupdate',
      '--url', OLLAMA_URL,
      '--http-host-header', 'localhost:11434',
    ],
    { stdio: ['ignore', 'pipe', 'pipe'] },
  );

  let deployed = false;
  const onData = (chunk) => {
    const text = chunk.toString();
    process.stdout.write(text);
    if (!deployed) {
      const m = text.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/);
      if (m) {
        const url = m[0];
        updateEnvFile(url);
        console.log(`\ntunnel: ✅ ready at ${url}`);
        console.log('tunnel: paste this into `curl -H "Origin: https://voiceeye.pages.dev" <url>/api/tags` to verify CORS.');
        if (shouldDeploy) runDeploy();
        deployed = true;
      }
    }
  };
  child.stdout.on('data', onData);
  child.stderr.on('data', onData);

  const shutdown = (signal) => {
    console.log(`\ntunnel: received ${signal}, stopping cloudflared…`);
    child.kill(signal);
  };
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));

  child.on('exit', (code) => {
    console.log(`tunnel: cloudflared exited with code ${code}`);
    process.exit(code ?? 0);
  });
}

main().catch((err) => {
  console.error('tunnel: fatal error:', err);
  process.exit(1);
});
