# VoiceEye Implementation Plan

Last updated: 25 April 2026  
Current stage: End of Sprint 2 / Sprint 3 planning

VoiceEye is the mobile-first web implementation of the original Scene-to-Speak proof of concept. The current product uses a two-lane architecture: a browser Fast Lane for real-time spatial awareness and a Slow Lane for on-demand scene description, visible text reading, and object search.

## Current Release Snapshot

| Area | Current Status |
|---|---|
| Frontend app | React 19 + TypeScript + Vite app implemented. |
| Camera runtime | Browser camera view, retry/error handling, and spoken camera errors implemented. |
| Fast Lane model | YOLO Fast Lane training completed. Deployable ONNX model exists. |
| Fast Lane runtime | ONNX Runtime Web inference, tracking, distance estimation, attention filtering, overlays, speech, and haptics implemented. |
| Slow Lane runtime | Describe / Read / Search flow implemented through an Ollama-compatible endpoint. |
| Voice control | Wake-word recognition and command parsing implemented. |
| Deployment | Vite production build and Cloudflare Pages deployment path implemented. |
| Testing | `npm test` passed: 95 tests across 5 files. |
| Build | `npm run build` passed. |
| Python data-prep tests | Not currently runnable in the local `.venv`; `pytest` is missing. |
| PWA service worker | Intentionally disabled for now because previous service-worker caching caused mobile Safari memory/cache issues. |

## Sprint 2 Model Result

YOLO Fast Lane training result:

`runs/detect/VoiceEye_Runs/fastlane_train`

Key artifacts:

| Artifact | Purpose | Size |
|---|---|---:|
| `weights/best.pt` | Best PyTorch checkpoint | 5.2 MB |
| `weights/last.pt` | Last PyTorch checkpoint | 5.2 MB |
| `weights/best.onnx` | Browser deployment artifact | 9.4 MB |

Observed metrics from Ultralytics plots:

| Metric | Result |
|---|---:|
| Best all-class F1 | 0.52 at confidence 0.204 |
| Final precision | about 0.62 |
| Final recall | about 0.46 |
| Final mAP50 | about 0.50 |
| Final mAP50-95 | about 0.35 |

Note: `results.csv` is not present in the run folder, so scalar metric values are plot-derived from `results.png` and `BoxF1_curve.png`.

## Completed Work

### Phase 1: Foundation and Quality - Completed

| Task | Status |
|---|---|
| Add Vitest and testing infrastructure | Done |
| Add tests for tracker, distance, inference metrics, attention, and voice command parsing | Done |
| Decompose app behavior into custom hooks | Done |
| Add React error boundary with TTS feedback | Done |
| Centralize tunable constants in `src/config.ts` | Done |
| Add VLM timeout and spoken error handling | Done; current timeout is 60 seconds to allow cold starts |
| Enable TypeScript build validation | Done |
| Add frontend production build path | Done |

### Phase 2: Mobile Product Prototype - Completed

| Task | Status |
|---|---|
| Full-screen browser camera UI | Done |
| Real-time bounding-box overlay | Done |
| YOLO ONNX model loading through ONNX Runtime Web | Done |
| IoU tracker with persistent track IDs | Done |
| Velocity and proximity zones | Done |
| Pinhole-camera distance estimation | Done |
| Attention pipeline with hazard tiers, clustering, cooldowns, and verbosity budget | Done |
| Batched TTS announcements for Fast Lane alerts | Done |
| Haptic warning patterns | Done |
| De-escalation tone | Done |
| Settings panel for TTS speed, sensitivity, verbosity, and haptics | Done |
| Wake-word command parsing | Done |
| Slow Lane Describe / Read / Search trigger flow | Done |
| Local Ollama dev proxy | Done |
| Cloudflare Pages Function for production VLM endpoint | Done |
| Cloudflare Pages deployment scripts | Done |
| Full pipeline diagram and Sprint 3 Jira tickets | Done |
| Demo deck | Done |

## Current Architecture

```text
src/
├── App.tsx                      # Main orchestrator for camera, settings, lanes, overlays, and UI state
├── main.tsx                     # React entry point with ErrorBoundary
├── config.ts                    # Central constants for detection, VLM, audio, attention, and UI
├── components/
│   ├── CameraView.tsx           # Camera access, retry, and error recovery
│   ├── SettingsPanel.tsx         # Persisted settings UI
│   └── ErrorBoundary.tsx        # Crash recovery with spoken feedback
├── hooks/
│   ├── useDetectionLoop.ts      # Fast Lane: YOLO inference, tracking, attention, speech, haptics
│   ├── useVLMEngine.ts          # Slow Lane: frame capture, prompt mode, endpoint call, timeout handling
│   ├── useVoiceRecognition.ts   # Wake word and command parsing
│   └── useSpatialAudio.ts       # Speech synthesis, beeps, audio unlock, haptic primitives
├── utils/
│   ├── yolo.ts                  # ONNX Runtime Web model loading and YOLO pre/post-processing
│   ├── tracker.ts               # IoU tracking, velocity, and proximity zone logic
│   ├── attention.ts             # Alert priority, clustering, cooldown, and budget logic
│   ├── distance.ts              # Monocular distance estimate
│   └── inferenceMetrics.ts      # Runtime metrics tracking
└── test/
    └── setup.ts                 # Vitest setup
```

Supporting paths:

| Path | Purpose |
|---|---|
| `public/models/best.onnx` | Runtime YOLO model loaded by the browser app. |
| `functions/api/vlm.ts` | Cloudflare Pages Function for production Slow Lane VLM calls. |
| `training/` | YOLO training, ClearML pipeline, HPO, and model-card support. |
| `training/data_prep/` | COCO / Open Images / Mapillary data preparation pipeline. |
| `runs/detect/VoiceEye_Runs/fastlane_train` | Completed Fast Lane YOLO training result. |
| `docs/full-pipeline-diagram.md` | Editable full project pipeline diagram. |
| `docs/sprint-3-jira-tickets.md` | Sprint 3 Jira-ready ticket plan. |

## Phase 3: Validation, Benchmarking, and Demo Hardening - Current Priority

Sprint 3 should focus on proving the Sprint 2 implementation on real devices rather than adding large new features.

| # | Task | Priority | Details |
|---|---|---|---|
| 3.1 | Promote and document trained model artifact | Critical | Confirm `runs/.../best.onnx` and `public/models/best.onnx` are the intended same release artifact. Record checksum, run path, metrics, and limitations. |
| 3.2 | Desktop browser benchmark | High | Measure model load time, average inference latency, p95 latency, FPS, memory observations, and browser version using a production build. |
| 3.3 | Phone browser benchmark | Critical | Test at least one real iOS or Android phone with live camera input. Record FPS, crashes, thermal behavior, battery observations, speech, haptics, and bounding-box behavior. |
| 3.4 | Tune detection confidence threshold | High | Training F1 curve peaks at confidence 0.204, while the app default is higher. Tune based on live precision/recall tradeoff, not plot data alone. |
| 3.5 | Validate local Ollama Slow Lane | High | Test Describe, Read, and Search through the frontend with local Qwen/Ollama. Record latency, usefulness, hallucinations, and failure modes. |
| 3.6 | Validate Cloudflare Workers AI Slow Lane | High | Test deployed `/api/vlm`, CORS/origin checks, timeout behavior, and model response quality. |
| 3.7 | Build phone compatibility matrix | High | Track camera, mic, SpeechRecognition, SpeechSynthesis, vibration, audio unlock, and standalone behavior by device/browser. |
| 3.8 | Graceful fallback for unsupported voice recognition | Medium | If SpeechRecognition is unavailable, keep tap-based use working and provide a clear user message. |
| 3.9 | Calibrate hazard alert behavior | Critical | Use live scenes to tune proximity thresholds, cooldowns, clustering, verbosity budgets, and haptic behavior. |
| 3.10 | Accessibility audit | High | Check labels, aria-live behavior, contrast, tap targets, focus behavior, and screen-reader behavior. |
| 3.11 | Restore Python data-prep test environment | High | Install or document the required Python environment so `python -m pytest training/data_prep/tests/ -q` runs. |
| 3.12 | Prepare user testing protocol | High | Draft supervised testing tasks, safety constraints, consent/privacy wording, and observation template. |
| 3.13 | Reconcile documentation naming | Medium | Make docs consistently explain the transition from Scene-to-Speak desktop POC to VoiceEye mobile PWA. |
| 3.14 | Prepare Sprint 3 demo script | Medium | Keep a repeatable demo path with fallback screenshots/expected outputs if phone or VLM fails live. |

See `docs/sprint-3-jira-tickets.md` for the full Jira-ready Sprint 3 backlog.

## Deferred or Re-scoped Items

These items were previously listed as near-term work, but should not block the Sprint 3 demo unless validation shows they are necessary.

| Item | Current Decision |
|---|---|
| Offline fallback and service worker caching | Deferred. Service worker is disabled until mobile memory/cache stability is proven. |
| High-contrast theme | Valuable, but after core phone validation and accessibility audit. |
| Confidence wording in every announcement | Deferred until alert calibration; may increase cognitive load. |
| Camera front/back selector | Useful for document reading, but lower priority than current camera reliability and phone compatibility. |
| Canvas overlay replacement | Consider only if phone benchmarks show DOM overlay performance issues. |
| Adaptive inference throttling | Sprint 3 should first collect real latency data; adaptive logic can follow. |
| ONNX int8 quantization | Useful for size/speed, but must be evaluated after baseline ONNX runtime measurements. |
| Battery-aware mode | Depends on browser support and benchmark results. |
| i18n / multi-language support | Later accessibility phase after English flow is validated. |
| Social VLM mode | Later feature; avoid face/age-sensitive claims until safety/privacy review. |
| Learned depth estimation | Later research item; current distance model remains approximate. |
| WebGPU backend | Later optimization after WASM baseline is measured. |

## Phase 4: Accessibility and User Readiness

| # | Task | Priority | Details |
|---|---|---|---|
| 4.1 | VoiceOver / TalkBack compatibility testing | Critical | Validate the full app flow with mobile screen readers. |
| 4.2 | High-contrast and low-vision UI mode | High | Add a high-contrast option if the accessibility audit confirms need. |
| 4.3 | Per-device FOV calibration | Medium | Improve distance estimation by calibrating the camera field of view. |
| 4.4 | User testing with visually impaired participants | Critical | Run supervised task-based testing only after Sprint 3 safety checklist is complete. |
| 4.5 | Ground-level hazard dataset expansion | High | Improve detection for curbs, stairs, crosswalks, poles, bollards, and doors. |
| 4.6 | Customizable haptic patterns | Medium | Support vibration sensitivity preferences. |
| 4.7 | Speech urgency tuning | Medium | Explore faster or shorter speech for danger-zone alerts after user feedback. |

## Phase 5: Ecosystem and Scale

| # | Task | Priority | Details |
|---|---|---|---|
| 5.1 | Automated model retraining pipeline | Medium | Use ClearML and GitHub Actions to evaluate and promote new ONNX models through PRs. |
| 5.2 | Docker container for training | Medium | Pin Python, CUDA, PyTorch, Ultralytics, and data-prep dependencies. |
| 5.3 | Opt-in privacy-preserving telemetry | High | Only after privacy review; never collect images or location. |
| 5.4 | WebGPU backend investigation | Medium | Benchmark ONNX Runtime WebGPU where supported; keep WASM fallback. |
| 5.5 | Learned depth estimation | Medium | Evaluate MiDaS or DepthAnything ONNX if mobile runtime budget allows. |
| 5.6 | Smart cane or wearable integration research | Low | Explore BLE haptic routing after core phone app is validated. |
| 5.7 | Indoor navigation research | Low | Explore landmarks, signs, doors, elevators, and AR anchors as a later phase. |

## Success Criteria

| Metric | Current | Sprint 3 Target | Later Target |
|---|---:|---:|---:|
| Frontend tests | 95 passing | Maintained | Broaden hook/component coverage |
| Production build | Passing | Maintained | CI enforced |
| Fast Lane model artifact | Trained ONNX exists | Provenance + runtime checksum documented | Automated promotion pipeline |
| Desktop browser inference | Not measured in current report | Baseline measured | Regression tracked |
| Phone browser inference | Not measured in current report | At least one target phone measured | Device matrix expanded |
| Slow Lane local Ollama | Flow implemented | Live quality/latency evaluated | Prompt tuned |
| Slow Lane Cloudflare | Function implemented | Deployed flow verified | Rate limits / access protection reviewed |
| Voice command fallback | Basic support | Unsupported-browser fallback implemented | Full compatibility matrix |
| Accessibility audit | Pending | Completed | User testing completed |
| Python data-prep tests | Blocked by missing pytest | Restored and recorded | CI validated |
| Offline Fast Lane | Deferred | Do not re-enable until mobile cache stability is proven | Reconsider after benchmarks |

## Immediate Next Actions

1. Verify whether `public/models/best.onnx` matches `runs/detect/VoiceEye_Runs/fastlane_train/weights/best.onnx`.
2. Add model provenance documentation with checksum and plot-derived metrics.
3. Run production desktop benchmark for Fast Lane inference.
4. Run phone benchmark using `npm run phone` or Cloudflare Pages deployment.
5. Validate Slow Lane against local Ollama and Cloudflare Workers AI.
6. Restore the Python test environment for `training/data_prep/tests`.
7. Use the Sprint 3 Jira tickets in `docs/sprint-3-jira-tickets.md` to populate the next sprint board.
