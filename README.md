# VoiceEye: AI Vision Assistant

VoiceEye is an advanced assistive technology application designed to provide real-time spatial awareness and deep contextual understanding for visually impaired users. It utilises a **Dual-Lane Processing Architecture** to balance immediate feedback with complex scene analysis. All AI processing runs locally on-device — no data leaves your environment.

---

## Core Architecture: The Two-Lane Approach

VoiceEye operates using two distinct computational lanes:

### The Fast Lane (Real-Time Object Detection + Tracking)
- **Engine**: YOLO26n via ONNX Runtime Web (WebAssembly, runs entirely in-browser)
- **Function**: Continuous object detection and multi-object tracking, throttled to ~10fps for battery efficiency.
- **Tracking**: Each detected object is assigned a persistent **Track ID** across frames using IoU-based matching. Once an object is announced, it is not repeated until it leaves and re-enters the frame.
- **Velocity & Proximity**: EMA-smoothed velocity estimation detects approaching objects. Proximity zones (safe/near/danger) trigger escalating alerts.
- **Bounding boxes**: Labelled as `{class} #{trackId} {distance}` — e.g. `person #3 ~1.2m`. Color-coded green (safe), amber (near), or red (danger).
- **Distance**: Estimated using a pinhole camera model with known average object heights.
- **Feedback**: Instant audio announcements and haptic vibration for close or hazardous objects.

### The Slow Lane (Deep Context VLM)
- **Engine**: Qwen2.5-VL via local Ollama API
- **Function**: High-fidelity scene description, text extraction (OCR), and targeted object search.
- **Trigger**: Tap the screen or use a voice command.

---

## Key Features

- **Hands-Free Voice Commands**: Continuous background listening for the "Voice Eye" wake word. Instant audio beep confirmation on detection.
- **Per-Object Announcement**: Each tracked object is announced exactly once per appearance — no repetitive alerts.
- **Approaching-Object Alerts**: Warns when hazardous objects (cars, buses, trucks, etc.) are moving toward the user, with estimated distance.
- **Zone-Based Re-announcement**: Objects entering the danger zone trigger re-alerts. Sustained proximity (>3s) triggers "Still close" reminders, capped at 3 per track.
- **Distance Estimation**: Monocular distance shown on every bounding box (e.g. `~1.2m`, `>5m`).
- **Haptic Proximity Engine**: Device vibration alerts for nearby objects.
  - *Single pulse*: Object occupies >50% of screen area.
  - *Triple pulse*: Collision warning — object occupies >60% of screen area.
  - *Rapid burst*: Hazardous object approaching.
- **Configurable Settings**: Adjustable TTS speed, detection sensitivity, and haptic toggle — persisted across sessions.
- **Camera Error Recovery**: Automatic retry on transient camera errors with user-facing spoken error messages.
- **Native PWA**: Installable on Android and iOS with full-screen standalone mode.
- **MLOps Pipeline**: ClearML-tracked training with automated pipeline, HPO (Optuna), and remote agent execution.
- **Inference Monitoring**: In-browser latency/FPS/confidence tracking with console logging.
- **Local-First Privacy**: All detection (ONNX Runtime Web) and VLM (Ollama) runs on your hardware.

---

## Technology Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | React 19, TypeScript 6, Vite 8 |
| **Object Detection** | YOLO26n via ONNX Runtime Web |
| **Object Tracking** | IoU-based multi-object tracker with velocity estimation (`src/utils/tracker.ts`) |
| **Distance Estimation** | Pinhole camera model (`src/utils/distance.ts`) |
| **Vision-Language Model** | Qwen2.5-VL (local via Ollama) |
| **Styling** | Vanilla CSS — glassmorphism design system |
| **PWA** | vite-plugin-pwa |
| **MLOps** | ClearML + Ultralytics YOLO + Optuna HPO |
| **CI/CD** | GitHub Actions |

---

## Getting Started

### 1. Prerequisites
- [Node.js](https://nodejs.org/) v18+
- [Ollama](https://ollama.com/) running locally
- A YOLO26n model exported to ONNX format
- [ClearML](https://clear.ml/) (optional — for custom model training)

### 2. Prepare the Ollama VLM
```bash
ollama run qwen2.5vl
```

### 3. Prepare the YOLO Model
Export YOLO26n to ONNX and place it at `public/models/yolo26n.onnx`:
```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=640
# then copy yolo26n.onnx → public/models/yolo26n.onnx
```

### 4. Install Dependencies
```bash
npm install
```
> The `postinstall` script automatically copies ONNX Runtime WASM files to `public/ort-wasm/`.

### 5. Run the Dev Server

On desktop:
```bash
npm run dev
```

On a **physical mobile device** (required for camera, haptics, and PWA):
```bash
npm run dev -- --host
```
> Open the `https://` version of your local IP on your phone. Accept the self-signed certificate warning to grant camera and microphone permissions.

---

## Command Guide

| Voice Command | Action |
| :--- | :--- |
| **"Voice Eye, describe"** | Generates a general scene description via Qwen VLM. |
| **"Voice Eye, read text"** | Switches to OCR mode to read visible documents or labels. |
| **"Voice Eye, find [object]"** | Searches the scene for a specific target (e.g. "find my keys"). |

---

## Settings

Accessible via the gear icon in the top-right corner. Persisted to localStorage.

| Setting | Range | Default |
| :--- | :--- | :--- |
| **Voice Speed** | 0.5x – 2.0x | 1.0x |
| **Detection Sensitivity** | 30% – 80% | 50% |
| **Haptic Feedback** | On / Off | On |

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
# Create template task
python training/train.py --epochs 5

# Run HPO with Optuna (requires clearml-agent)
python training/hpo.py --template-task-id <TASK_ID> --max-trials 20

# Best params saved to config.yaml — run full training
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

## Project Structure

```text
VoiceEye/
├── public/
│   ├── models/
│   │   └── yolo26n.onnx          # YOLO model (not in git — add manually)
│   ├── icons/                    # PWA icons (192x192, 512x512, apple-touch-icon)
│   └── ort-wasm/                 # ONNX Runtime WASM (auto-copied by postinstall)
├── src/
│   ├── components/
│   │   ├── CameraView.tsx        # Camera hardware access, error recovery & retry
│   │   └── SettingsPanel.tsx     # Settings UI (TTS speed, sensitivity, haptics)
│   ├── utils/
│   │   ├── yolo.ts               # ONNX session loader + YOLO pre/post-processing
│   │   ├── tracker.ts            # IoU tracker with velocity & proximity zones
│   │   ├── distance.ts           # Monocular distance estimation
│   │   └── inferenceMetrics.ts   # In-browser latency/FPS/confidence tracking
│   ├── App.tsx                   # Core logic: detection loop, TTS, haptics, VLM, settings
│   └── index.css                 # Glassmorphism design system
├── training/
│   ├── config.yaml               # All training hyperparameters (single source of truth)
│   ├── train.py                  # ClearML-tracked training script
│   ├── pipeline.py               # 4-step ClearML pipeline (local or remote)
│   ├── hpo.py                    # Hyperparameter optimization (Optuna + ClearML)
│   ├── validate_config.py        # Config validation for CI
│   ├── requirements.txt          # Python dependencies
│   ├── agent_setup.md            # ClearML Agent setup guide
│   └── model_card.md             # Model card template
├── .github/workflows/
│   ├── frontend-ci.yml           # Lint + build on PR
│   ├── training-config-validate.yml  # Config validation on PR
│   └── model-download.yml        # Download model from ClearML
├── mlops_clearml_yolo.ipynb      # Interactive training notebook
└── vite.config.ts                # Ollama proxy + PWA config
```

---

## License
MIT License — Copyright (c) 2026 VoiceEye Project
