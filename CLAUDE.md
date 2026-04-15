# VoiceEye — Claude Code Guide

## Project Summary

VoiceEye is an accessibility-focused **Progressive Web App (PWA)** that gives visually impaired users real-time spatial awareness through their phone camera. All processing is local-first — no data leaves the device.

---

## Architecture — Dual-Lane Processing

| Lane | Technology | Purpose |
|------|-----------|---------|
| **Fast Lane** | YOLO26n via ONNX Runtime Web | Real-time object detection + tracking, throttled to ~10fps |
| **Slow Lane** | Qwen2.5-VL via Ollama (local) | On-demand deep scene description, OCR, and object search |

The two lanes run independently. Fast Lane feeds a continuous `requestAnimationFrame` loop (inference throttled to ~10fps via `performance.now()`). Slow Lane is triggered by voice command or screen tap.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | React 19 + TypeScript 6 |
| Build tool | Vite 8 |
| Object detection | ONNX Runtime Web + YOLO26n |
| Vision-language model | Qwen2.5-VL (via Ollama REST API) |
| Icons | Lucide React |
| Styling | Vanilla CSS — glassmorphism design system |
| PWA | vite-plugin-pwa |
| ML training pipeline | ClearML + Ultralytics YOLO (Python, `training/`) |
| HPO | ClearML HyperParameterOptimizer + Optuna |
| CI/CD | GitHub Actions |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/App.tsx` | Core logic: detection loop, tracker, haptics, TTS, voice commands, VLM calls, settings |
| `src/components/CameraView.tsx` | Camera hardware access (`getUserMedia`), back-facing preference, error recovery + retry |
| `src/components/SettingsPanel.tsx` | Settings UI: TTS speed, detection sensitivity, haptic toggle. Persists to localStorage |
| `src/utils/yolo.ts` | ONNX session loader + YOLO pre/post-processing + NMS. Singleton canvas/buffer reuse, tensor disposal |
| `src/utils/tracker.ts` | IoU-based multi-object tracker with velocity estimation, proximity zones, re-announcement logic |
| `src/utils/distance.ts` | Monocular distance estimation via pinhole camera model |
| `src/utils/inferenceMetrics.ts` | In-memory inference latency/FPS/confidence tracking (singleton) |
| `vite.config.ts` | Ollama proxy (`/api/ollama` → port 11434), PWA config with icons |
| `public/models/yolo26n.onnx` | YOLO26n ONNX model (must be placed here manually — not in git) |
| `training/config.yaml` | All training hyperparameters — single source of truth |
| `training/train.py` | ClearML-tracked training script with quality gate + ONNX export |
| `training/pipeline.py` | 4-step ClearML pipeline (data_prep → train → evaluate → export) |
| `training/hpo.py` | Hyperparameter optimization via ClearML + Optuna TPE |
| `training/requirements.txt` | Python dependencies for training |
| `training/agent_setup.md` | ClearML Agent setup guide |
| `training/model_card.md` | Model card template |
| `training/validate_config.py` | Config validation script (used by CI) |
| `mlops_clearml_yolo.ipynb` | Interactive notebook (uses config.yaml, exports ONNX) |

---

## Dev Commands

```bash
npm run dev                # Start dev server (https://localhost:5173)
npm run dev -- --host      # Expose on local network IP — for testing on a physical phone
npm run build              # TypeScript check + production bundle
npm run lint               # ESLint
npm run preview            # Preview production build locally
```

---

## External Dependencies (must be running locally)

### Ollama (for Slow Lane VLM)
```bash
ollama serve               # Starts REST API on http://127.0.0.1:11434
ollama run qwen2.5vl       # Pull + verify the Qwen model
```
Vite proxies `/api/ollama/*` to `http://127.0.0.1:11434` — removing CORS headers automatically.

### YOLO ONNX Model
Place the model at `public/models/yolo26n.onnx`. To export from Ultralytics:
```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=640
```
For a custom ClearML-trained model, export the best checkpoint the same way and drop it in `public/models/`.

### ONNX Runtime WASM Files
After `npm install`, a `postinstall` script automatically copies `*.wasm` files from
`node_modules/onnxruntime-web/dist/` to `public/ort-wasm/`.
The runtime path is set in `src/utils/yolo.ts` as `ort.env.wasm.wasmPaths = '/ort-wasm/'`.

---

## Tracking Behaviour

- Each detected object gets a unique **Track ID** that persists across frames via IoU matching.
- An object is **announced via TTS exactly once** per track lifetime (`track.announced` flag).
- When an object **leaves frame** and re-enters, it gets a **new Track ID** and is announced again.
- Tracks are dropped after **10 consecutive frames** without a matching detection.

### Velocity & Proximity

- **EMA-smoothed velocity** (alpha=0.3) tracks centroid movement and area growth rate per frame.
- **Proximity zones**: `safe` (<30% screen area), `near` (30–50%), `danger` (>50%).
- **Approaching alert**: If `areaGrowthRate > 0.05` and class is hazardous → "Warning: car approaching, ~2.1m" + rapid triple-burst haptic.
- **Zone-transition re-announcement**: When an object enters the `danger` zone → "Close: car, ~1.2m" (even if already announced).
- **Sustained proximity**: If in `danger` zone >3 seconds → "Still close: car". Capped at 3 re-announcements per track.
- **Color-coded bounding boxes**: green (safe), amber (near), red (danger).

---

## Bounding Box Label Format

```
{class} #{trackId} {distance}
```
Example: `person #3 ~1.2m`

Distance is estimated using known average object heights + a 70° vertical FOV approximation.

---

## Memory & Performance

- **Singleton canvas/buffer**: `yolo.ts` reuses a single `OffscreenCanvas` and `Float32Array` across frames (lazy-init). No per-frame allocation.
- **Tensor disposal**: ONNX input/output tensors are disposed in `try/finally` blocks to prevent WASM heap leaks.
- **Inference throttle**: rAF loop runs at display refresh, but `runYolo()` fires only every 100ms (~10fps).

---

## Settings

Persisted to `localStorage` under `voiceeye_settings`. Accessible via the gear icon in the header.

| Setting | Range | Default | Wired to |
|---------|-------|---------|----------|
| Voice Speed (TTS rate) | 0.5–2.0 | 1.0 | All `speak()` calls |
| Detection Sensitivity | 0.3–0.8 | 0.5 | `confThreshold` in `runYolo()` |
| Haptic Feedback | on/off | on | All `navigator.vibrate()` calls |

---

## Voice UX

- **Instant confirmation**: On wake word detection, an 880Hz beep fires immediately via `AudioContext`, followed by a queued "Got it" via `speakQuick()` (non-cancelling TTS).
- **Message brevity**: First launch shows full intro message; subsequent launches show just "Ready".

---

## Camera Error Handling

`CameraView.tsx` maps `DOMException` types to user-facing messages:
- `NotAllowedError` → "Camera access denied…"
- `NotFoundError` → "No camera found."
- `NotReadableError` / `OverconstrainedError` → retryable with up to 3 attempts (2s delay)

Errors are spoken via TTS and displayed in the message box.

---

## MLOps Workflow

### Training Pipeline

All training config lives in `training/config.yaml`. Training is convergence-tuned: 150 epochs, AdamW optimizer, cosine annealing LR, early stopping at patience 30.

```bash
pip install -r training/requirements.txt

# Interactive notebook
jupyter notebook mlops_clearml_yolo.ipynb

# Standalone script
python training/train.py                    # Local, config defaults
python training/train.py --epochs 200       # CLI override
```

### ClearML Pipeline + Remote Agent

```bash
# Local execution (all steps in-process)
python training/pipeline.py

# Remote execution (steps enqueued to clearml-agent)
clearml-agent daemon --queue gpu cpu --gpus 0    # Start agent first
python training/pipeline.py --remote
```

Pipeline steps: `data_prep` (cpu) → `train` (gpu) → `evaluate` (cpu) → `export` (cpu).

### Hyperparameter Optimization

```bash
# 1. Create template task (short run)
python training/train.py --epochs 5

# 2. Run HPO (requires clearml-agent running)
python training/hpo.py --template-task-id <TASK_ID> --max-trials 20

# 3. Best params auto-saved to config.yaml
# 4. Full training with optimized config
python training/pipeline.py --remote
```

HPO search space: lr0, lrf, optimizer, batch, weight_decay, warmup_epochs, mosaic, scale, mixup, cos_lr. Uses Optuna TPE (Bayesian) sampler via `OptimizerOptuna`.

### CI/CD (GitHub Actions)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `frontend-ci.yml` | PR to main (src/, public/) | Lint + build |
| `training-config-validate.yml` | PR to main (training/config.yaml) | Validate config correctness |
| `model-download.yml` | Manual dispatch | Download ONNX from ClearML, open PR |

### Inference Monitoring

`src/utils/inferenceMetrics.ts` tracks latency, FPS, and confidence distribution in a circular buffer (last 100 frames). Logs to console every 300 frames:
```
[VoiceEye] fps=8.2 avgLatency=118ms p95=145ms detections/frame=3.1
```

---

## Conventions

- **TypeScript strict mode** — no unused variables or parameters; no implicit `any`.
- **CSS** — glassmorphism design: `backdrop-filter: blur`, semi-transparent panels, indigo primary colour. Don't switch to a CSS framework without discussion.
- **No TF.js** — the project migrated from COCO-SSD to ONNX Runtime Web. Do not re-add `@tensorflow/tfjs`.
- **State vs. refs** — the live track list is held in `tracksRef` (mutable, avoids stale closures inside the rAF loop). `renderedTracks` state is set from it to drive React rendering. Settings are mirrored in `settingsRef` for the same reason.
- **Haptic patterns** — single pulse (`[100]`) for close objects; triple pulse (`[50,50,50]`) for collision warning; rapid burst (`[50,30,50,30,50]`) for approaching hazard. Keep these distinct.

---

## Testing

1. `npm install` — installs deps including `onnxruntime-web`; postinstall copies WASM files
2. Place `yolo26n.onnx` in `public/models/`
3. `npm run dev -- --host` — open the HTTPS URL on a physical phone (camera requires HTTPS)
4. Grant camera + microphone permissions
5. Verify bounding boxes appear with `{class} #{id} ~Xm` labels, color-coded by proximity
6. Verify each object is announced only once until it leaves and re-enters frame
7. Walk toward camera — verify "Warning: person approaching" fires when area is growing
8. Say "Voice Eye, describe" — confirm beep + "Got it" + Slow Lane fires
9. Open Settings — change TTS speed, verify speech rate changes
10. Disable haptics — verify no vibration on close objects
11. `npm run build` — confirm TypeScript compiles with zero errors
