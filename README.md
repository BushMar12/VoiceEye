# VoiceEye: AI Vision Assistant

VoiceEye is an accessibility-focused **Progressive Web App (PWA)** that gives visually impaired users real-time spatial awareness through their phone camera. It uses a **Dual-Lane Processing Architecture** — combining instant object detection with deep scene understanding. All AI processing runs locally on-device — no data leaves your environment.

---

## Core Architecture: The Two-Lane Approach

### Fast Lane (Real-Time Object Detection + Tracking)
- **Engine**: YOLO26n via ONNX Runtime Web (WebAssembly, runs entirely in-browser)
- **Function**: Continuous object detection and multi-object tracking, throttled to ~10fps for battery efficiency
- **Tracking**: Each detected object gets a persistent **Track ID** via IoU-based matching
- **Attention Pipeline**: Filters which tracks become speech. Top-K budget per 2s window (Quiet=1, Normal=3, Detailed=6), reason-gated cooldowns, same-class spatial clustering, and a tier-1 hazard override that bypasses the budget for safe→danger step-jumps
- **Velocity & Proximity**: EMA-smoothed velocity estimation detects approaching objects. Proximity zones (safe/near/danger) drive priority and gating
- **Bounding Boxes**: Labelled as `{class} #{trackId} {distance}` (e.g. `person #3 ~1.2m`), color-coded green/amber/red per current proximity zone
- **Distance**: Estimated using a pinhole camera model with known average object heights
- **Feedback**: Batched TTS per frame plus haptic vibration patterns; 440Hz de-escalation tone when an object exits the danger zone

### Slow Lane (Deep Context VLM)
- **Engine**: Qwen3-VL via local Ollama API
- **Function**: Scene description, text extraction (OCR), and targeted object search
- **Trigger**: Tap the screen or use a voice command
- **Timeout**: 15-second AbortController timeout with spoken error feedback

---

## Key Features

- **Hands-Free Voice Commands** — Continuous background listening for the "Voice Eye" wake word with instant 880Hz beep confirmation
- **Cognitive-Load Budgeting** — Verbosity setting (Quiet/Normal/Detailed) caps announcements per 2s window so the user is never drowned in speech in crowded scenes
- **Tier-1 Hazard Override** — Cars, buses, motorcycles etc. bypass the budget when they step from safe directly into the danger zone
- **Approaching-Object Alerts** — Warns when hazardous objects are moving toward the user, with estimated distance
- **Spatial Clustering** — Same-class objects within 15% of frame diagonal are announced as a group ("3 people, ~4m") instead of individually
- **De-escalation Cue** — A 440Hz tone fires when an object exits the danger zone, distinct from the wake-word beep
- **Distance Estimation** — Monocular distance on every bounding box (e.g. `~1.2m`, `>5m`)
- **Haptic Proximity Engine** — Device vibration patterns:
  - *Single pulse*: Object occupies >50% of screen
  - *Triple pulse*: Collision warning (>60%)
  - *Rapid burst*: Hazardous object approaching
- **Configurable Settings** — TTS speed, detection sensitivity, haptic toggle — persisted to localStorage
- **Camera Error Recovery** — Automatic retry (up to 3 attempts) with spoken error messages
- **Error Boundary** — Crash recovery with TTS feedback ("VoiceEye encountered an error") and tap-to-reload
- **VLM Error Handling** — Spoken feedback for timeout, Ollama unavailability, and generic errors
- **Native PWA** — Installable on Android and iOS with full-screen standalone mode
- **MLOps Pipeline** — ClearML-tracked training with automated pipeline, HPO (Optuna), and remote agent execution
- **Inference Monitoring** — In-browser latency/FPS/confidence tracking with periodic console logging
- **Local-First Privacy** — All detection (ONNX Runtime Web) and VLM (Ollama) runs on your hardware

---

## Technology Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | React 19, TypeScript 5.6, Vite 6 |
| **Object Detection** | YOLO26n via ONNX Runtime Web (WASM) |
| **Object Tracking** | IoU-based multi-object tracker with velocity estimation |
| **Distance Estimation** | Pinhole camera model with known object heights |
| **Vision-Language Model** | Qwen3-VL (local via Ollama) |
| **Styling** | Vanilla CSS — glassmorphism design system |
| **Testing** | Vitest + Testing Library |
| **MLOps** | ClearML + Ultralytics YOLO + Optuna HPO |
| **CI/CD** | GitHub Actions |

---

## Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/) v18+
- [Ollama](https://ollama.com/) running locally (for Slow Lane)
- A YOLO26n model exported to ONNX format
- [ClearML](https://clear.ml/) (optional — for custom model training)

### Quick Start

**Windows** — double-click `start.bat`

**macOS / Linux** — run `./start.sh`

Both scripts auto-install dependencies and open the browser.

### Manual Setup

#### 1. Prepare the Ollama VLM
```bash
ollama serve                # Start the API server
ollama run qwen3-vl:2b        # Pull + verify the model
```

#### 2. Prepare the YOLO Model
Export your trained YOLO model to ONNX and place it at `public/models/best.onnx`:
```bash
pip install ultralytics
yolo export model=best.pt format=onnx imgsz=640
# Copy best.onnx → public/models/best.onnx
```

#### 3. Install Dependencies
```bash
npm install
```
ONNX Runtime Web ships its WASM binary via the `onnxruntime-web/wasm` sub-import,
and Vite bundles only the non-threaded, non-JSEP variant (~12 MB) into `dist/assets/`.
No manual WASM copy step is required.

#### 4. Run the App

**Desktop (development):**
```bash
npm run dev
```
Opens `https://localhost:5173` with hot-module reload.

**Phone (production preview — use this for phone testing):**
```bash
npm run phone
```
Builds a production bundle and serves it on your local network IP at port 5173.
Open `https://<your-local-ip>:5173` on your phone and accept the self-signed certificate warning.

> **If a previous build crashed the page on your phone** (you see
> "A problem repeatedly occurred"), the old service worker is still cached.
> In Safari iOS: **Settings → Safari → Advanced → Website Data → search
> `192.168.x.x` → Delete**, then reopen the URL. The new build unregisters
> the service worker automatically on first load.

> **Important:** Do not use `npm run dev -- --host` to test on your phone.
> Safari iOS will crash the tab (OOM kill, ~5 s refresh loop) because the dev server
> ships unminified bundles, source maps, and StrictMode double-mounts — roughly 3–5×
> the memory footprint of a production build.

#### 5. Run Tests
```bash
npm test                       # Run all tests once (70+)
npm run test:watch             # Watch mode
```

---

## Voice Commands

| Command | Action |
| :--- | :--- |
| **"Voice Eye, describe"** | Generates a scene description via Qwen VLM |
| **"Voice Eye, read"** | OCR mode — reads visible text, documents, or labels |
| **"Voice Eye, find [object]"** | Searches the scene for a specific object (e.g. "find my keys") |

All commands trigger an instant beep + "Got it" confirmation before processing.

---

## Settings

Accessible via the gear icon in the top-right corner. Persisted to localStorage.

| Setting | Range | Default |
| :--- | :--- | :--- |
| **Voice Speed** | 0.5x – 2.0x | 1.0x |
| **Detection Sensitivity** | 30% – 80% | 50% |
| **Verbosity** | Quiet / Normal / Detailed | Normal |
| **Haptic Feedback** | On / Off | On |

---

## Project Structure

```
VoiceEye/
├── src/
│   ├── App.tsx                      # Slim orchestrator (~165 lines)
│   ├── main.tsx                     # Entry point with ErrorBoundary
│   ├── config.ts                    # All tunable constants (single source of truth)
│   ├── index.css                    # Glassmorphism design system
│   ├── hooks/
│   │   ├── useDetectionLoop.ts      # Fast Lane: YOLO inference + tracking + alerts
│   │   ├── useVLMEngine.ts          # Slow Lane: Ollama VLM with timeout/dedup
│   │   ├── useVoiceRecognition.ts   # Wake word + command parsing
│   │   └── useSpatialAudio.ts       # TTS, beep, haptic primitives
│   ├── components/
│   │   ├── CameraView.tsx           # Camera access + error recovery + retry
│   │   ├── SettingsPanel.tsx         # Settings UI + localStorage persistence
│   │   └── ErrorBoundary.tsx        # Crash recovery with TTS feedback
│   ├── utils/
│   │   ├── yolo.ts                  # ONNX session + YOLO pre/post-processing + NMS
│   │   ├── tracker.ts              # IoU tracker + velocity + proximity zones
│   │   ├── attention.ts            # Priority + budget + cluster + cooldown filter
│   │   ├── distance.ts             # Monocular distance estimation
│   │   ├── inferenceMetrics.ts     # Latency/FPS/confidence + announcement counters
│   │   ├── tracker.test.ts         # 17 tests
│   │   ├── attention.test.ts       # 30 tests
│   │   ├── distance.test.ts        # 12 tests
│   │   └── inferenceMetrics.test.ts # 11 tests
│   └── test/
│       └── setup.ts                 # Vitest setup
├── public/
│   ├── models/                      # YOLO ONNX model (not in git — add manually)
│   ├── icons/                       # PWA icons
│   └── ort-wasm/                    # ONNX Runtime WASM (auto-copied by postinstall)
├── training/
│   ├── config.yaml                  # Training hyperparameters (single source of truth)
│   ├── train.py                     # ClearML-tracked training script
│   ├── pipeline.py                  # 4-step ClearML pipeline (local or remote)
│   ├── hpo.py                       # Hyperparameter optimization (Optuna + ClearML)
│   ├── validate_config.py           # Config validation for CI
│   ├── requirements.txt             # Python dependencies
│   ├── agent_setup.md               # ClearML Agent setup guide
│   └── model_card.md                # Model card template
├── .github/workflows/
│   ├── frontend-ci.yml              # Lint + build on PR
│   ├── training-config-validate.yml # Config validation on PR
│   └── model-download.yml           # Download model from ClearML
├── mlops_clearml_yolo.ipynb         # Interactive training notebook
├── start.bat                        # One-click start (Windows)
├── start.sh                         # One-click start (macOS/Linux)
├── IMPLEMENTATION_PLAN.md           # Roadmap: Phases 1-5
└── vite.config.ts                   # Ollama proxy + PWA config
```

---

## MLOps Workflow

VoiceEye includes a full MLOps pipeline for training, optimising, and deploying YOLO26n models.

### Quick Start

```bash
pip install -r training/requirements.txt

# Option 1: Interactive notebook
jupyter notebook mlops_clearml_yolo.ipynb

# Option 2: Standalone script
python training/train.py

# Option 3: Automated pipeline (local)
python training/pipeline.py

# Option 4: Remote execution via clearml-agent
clearml-agent daemon --queue gpu cpu --gpus 0
python training/pipeline.py --remote
```

### Hyperparameter Optimization

```bash
# 1. Create template task (short run)
python training/train.py --epochs 5

# 2. Run HPO with Optuna (requires clearml-agent)
python training/hpo.py --template-task-id <TASK_ID> --max-trials 20

# 3. Best params auto-saved to config.yaml
# 4. Full training with optimized config
python training/pipeline.py --remote
```

### Pipeline Steps

| Step | Queue | Function |
| :--- | :--- | :--- |
| 1. Data Prep | cpu | Fetch dataset from ClearML, validate `data.yaml` |
| 2. Train | gpu | YOLO26n training (150 epochs, cosine LR, early stopping) |
| 3. Evaluate | cpu | Validation metrics + quality gate (mAP@50 >= 0.40) |
| 4. Export | cpu | ONNX export + SHA-256 checksum + ClearML artifact upload |

### CI/CD

| Workflow | Trigger | Purpose |
| :--- | :--- | :--- |
| Frontend CI | PR (src/, public/) | `npm run lint` + `npm run build` |
| Config Validation | PR (training/config.yaml) | Validate training parameters |
| Model Download | Manual dispatch | Download ONNX from ClearML, open PR |

See [training/agent_setup.md](training/agent_setup.md) for ClearML Agent configuration.

---

## Dev Commands

```bash
npm run dev              # Start dev server (https://localhost:5173)
npm run dev -- --host    # Expose on local network for phone testing
npm run build            # TypeScript check + production bundle
npm run lint             # ESLint
npm run preview          # Preview production build
npm test                 # Run all tests (Vitest)
npm run test:watch       # Watch mode
```

---

## Deploying to Cloudflare Pages

VoiceEye is a static SPA — Cloudflare Pages gives it global HTTPS, edge caching,
and free custom domains with zero server maintenance. The full 12 MB ONNX
Runtime WASM bundle and the 9 MB YOLO model are both well under the 25 MB
per-file limit.

### 1. One-time setup

```bash
npm install                            # installs wrangler as a devDependency
npm run cf:login                       # opens a browser to authorise Wrangler
```

Create the Pages project (only needed once — the first `npm run deploy`
will also prompt to create it interactively):

```bash
npx wrangler pages project create voiceeye --production-branch=main
```

### 2. Ship a production build

```bash
npm run deploy                         # → https://voiceeye.pages.dev
npm run deploy:preview                 # preview channel (separate URL)
```

Both scripts run `vite build` first, so the deployed bundle always matches the
current working tree. `public/_headers` and `public/_redirects` are copied
into `dist/` automatically and tell Cloudflare to:

- serve `.wasm` with `Content-Type: application/wasm`
- cache hashed `/assets/*`, `/models/*` and `*.wasm` for a year (`immutable`)
- SPA-fallback every unknown path to `index.html`
- lock the `camera`, `microphone`, `accelerometer`, and `gyroscope` permissions
  to the site's own origin via `Permissions-Policy`

### 3. Slow Lane (VLM) backends

The Slow Lane picks an endpoint at build time, in this priority order:

1. **`VITE_OLLAMA_URL` set** → point at any Ollama-compatible server (e.g. a
   self-hosted Ollama reachable over a Cloudflare Tunnel). Used by
   `npm run tunnel` during development when you want to exercise the exact
   same Qwen3-VL model the app was tuned against. See `.env.example`.
2. **`vite dev`** → `/api/ollama/api/generate`, proxied to `127.0.0.1:11434`
   by `vite.config.ts`. Zero-setup dev loop with a local Ollama.
3. **Production build (default)** → `/api/vlm`, a [Cloudflare Pages Function](https://developers.cloudflare.com/pages/functions/) that
   runs [Workers AI](https://developers.cloudflare.com/workers-ai/) at the edge
   (see `functions/api/vlm.ts`). No tunnel, no Mac, no server to babysit.

#### Option 3a — Workers AI (recommended, default)

Already wired up — `npm run deploy` ships `functions/api/vlm.ts` alongside
the static bundle. The function uses the `[ai]` binding in `wrangler.toml`
and calls `@cf/meta/llama-3.2-11b-vision-instruct` by default. No secrets
needed; the binding is provisioned automatically for your Pages project.

First-time gotcha: Meta's Llama Vision models require a one-time license
acceptance on your Cloudflare account. After the first deploy, send the
literal prompt `agree` once and you're unlocked forever:

```bash
curl -X POST https://voiceeye.pages.dev/api/vlm \
  -H "Content-Type: application/json" \
  -d '{"prompt":"agree","images":["<any-base64-jpeg>"]}'
```

You'll get back *"Thank you for agreeing to this model's terms."* The next
real request will work. Free tier covers personal use comfortably — usage is
visible in the Cloudflare dashboard under **Workers AI → Analytics**.

To switch models, override `VLM_MODEL` in the Pages project's
**Settings → Environment variables** tab (or edit `wrangler.toml`). Options
include `@cf/llava-hf/llava-1.5-7b-hf` (no license prompt) or any other
vision model from the [Workers AI catalogue](https://developers.cloudflare.com/workers-ai/models/).

#### Option 3b — tunnel a local Ollama (Qwen3-VL or anything else)

The repo ships a helper that auto-downloads `cloudflared`, starts a free quick
tunnel to your local Ollama, rewrites the `Host` header so Ollama accepts the
request, and updates `.env.local` with the public URL:

```bash
# Terminal 1 — keep this running; tunnel lives as long as the process does
npm run tunnel                      # start tunnel, update .env.local
# or:
npm run tunnel:deploy               # same, then trigger `npm run deploy` once the URL is ready
```

Then tell your Ollama to trust the Pages origin **once**. On macOS (menu-bar app):

```bash
launchctl setenv OLLAMA_ORIGINS "*"
# → quit the Ollama menu-bar icon and relaunch Ollama.app
```

On macOS (`ollama serve` in a terminal) or Linux:

```bash
OLLAMA_ORIGINS="*" ollama serve
```

For a stricter allow-list replace `*` with
`https://voiceeye.pages.dev,https://*.voiceeye.pages.dev`.

Verify from any terminal — a `200` with an `access-control-allow-origin` header
means the Slow Lane will work end-to-end:

```bash
curl -i -H "Origin: https://voiceeye.pages.dev" \
  "$(grep VITE_OLLAMA_URL .env.local | cut -d= -f2)/api/tags"
```

> **Quick tunnels are ephemeral.** The `*.trycloudflare.com` URL is lost when
> the `cloudflared` process exits (Ctrl-C, sleep, reboot). Re-run
> `npm run tunnel:deploy` to mint a new URL and ship it in a fresh build.
> For a stable URL backed by your own domain, use a [named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/configure-tunnels/remote-management/).

If you wire a persistent named tunnel, put the hostname in your Pages project's
**Environment variables** tab instead of `.env.local` so every deploy inherits it.

### 4. Custom domain

In the Cloudflare Pages dashboard → **Custom domains → Set up a custom domain**.
Cloudflare issues the TLS cert automatically; no DNS changes are needed if the
domain is already on Cloudflare.

---

## License

MIT License — Copyright (c) 2025 VoiceEye Project
