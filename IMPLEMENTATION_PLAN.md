# VoiceEye Implementation Plan

## Phase 1: Foundation & Quality (Weeks 1-3) — COMPLETED

| Task | Status |
|------|--------|
| Add Vitest + testing infrastructure | Done |
| Write unit tests for tracker, distance, inferenceMetrics (35 tests) | Done |
| Decompose App.tsx into custom hooks (446 → 165 lines) | Done |
| Add React error boundary with TTS feedback | Done |
| Centralize magic numbers into `src/config.ts` (40+ constants) | Done |
| Add VLM request timeout (15s) + deduplication | Done |
| Enable TypeScript strict null checks | Done |
| Add spoken error for VLM unavailability (3 error paths) | Done |

---

## Phase 2: User Experience (Weeks 4-6)

**Goal:** Fill gaps that real users would hit immediately.

| # | Task | Priority | Effort | Details |
|---|------|----------|--------|---------|
| 2.1 | Add "thinking..." spoken cue during VLM processing | High | 0.5 day | After the initial "Analyzing scene" message, add a periodic spoken cue every ~3s while VLM is processing so users know the app hasn't frozen. |
| 2.2 | Add "help" voice command | High | 1 day | When user says "Voice Eye, help", speak a list of available commands: "You can say: Voice Eye describe, Voice Eye read, Voice Eye find [object]". Display the list in the message box. |
| 2.3 | Add "what's around me" summary command | High | 1 day | New voice command "Voice Eye, what's around me" that reads out all currently tracked objects with distances and positions (left/center/right based on bbox centroid). |
| 2.4 | Implement offline fallback + service worker caching | High | 2 days | Configure `vite-plugin-pwa` workbox to precache the ONNX model, WASM files, and app shell. Add offline fallback page. Fast Lane should work fully offline; show clear message when Slow Lane is unavailable. |
| 2.5 | Add high-contrast theme toggle | Medium | 2 days | Add a "High Contrast" option to SettingsPanel. Uses solid black background, white text, high-contrast bounding box colors. Persisted to localStorage. Important for users with partial vision. |
| 2.6 | Add confidence level to announcements | Medium | 1 day | Change TTS announcements to include confidence: "car, high confidence, ~2m" (>0.8), "person, moderate confidence, ~3m" (0.6-0.8). Helps users judge reliability. |
| 2.7 | Camera selection UI (front/back toggle) | Medium | 1 day | Add a camera flip button to the header. Useful for reading documents (front camera) vs navigation (back camera). Restart stream on toggle. |
| 2.8 | Persistent error indicators | Medium | 1 day | Add a status bar showing system health: model loaded, camera active, Ollama reachable. Red/green indicators. Errors persist visually (not just spoken once). |
| 2.9 | VLM processing progress feedback | Medium | 1 day | Replace silent wait with spoken progress: "Analyzing..." at 0s, "Still working..." at 5s, "Almost done..." at 10s. Gives users a sense of progress. |

---

## Phase 3: Performance & Robustness (Weeks 7-9)

**Goal:** Make it reliable for daily use.

| # | Task | Priority | Effort | Details |
|---|------|----------|--------|---------|
| 3.1 | Replace DOM bounding boxes with canvas overlay | Medium | 2 days | Current implementation creates a `<div>` per tracked object every frame. For >10 objects, this causes layout thrashing. Switch to a single `<canvas>` overlay and draw boxes via `CanvasRenderingContext2D`. |
| 3.2 | Adaptive inference throttling | High | 2 days | Measure actual inference latency per device. If avg latency > 150ms, reduce target FPS. If < 50ms, increase. This adapts to phone capability automatically. Store device profile in localStorage. |
| 3.3 | ONNX model quantization (float32 → int8) | High | 2 days | Quantize the YOLO model to int8 using ONNX quantization tools. ~4x size reduction (5.5MB → ~1.4MB), faster inference on mobile. Validate mAP doesn't drop below quality gate. |
| 3.4 | Battery-aware mode | Medium | 1 day | Use the Battery Status API (`navigator.getBattery()`). When battery < 20%, reduce inference FPS to 5fps and show "Battery saver mode" message. When < 10%, pause Fast Lane entirely. |
| 3.5 | Inference metrics export | Low | 1 day | Add a button in Settings to export inference metrics as JSON. Useful for debugging performance issues on specific devices. |
| 3.6 | WASM cold-start optimization | Medium | 1 day | Measure and log WASM + model load time. Speak "Model loaded, ready" once initialization completes. Add loading progress indicator. Currently there's no feedback during the 2-5s model load. |
| 3.7 | Optimize getImageData buffer reuse | Low | 1 day | Currently `getImageData()` allocates a new `Uint8ClampedArray` per inference frame. Investigate reusing the buffer via `getImageData(0, 0, w, h, { willReadFrequently: true })` and pre-allocated storage. |

---

## Phase 4: Accessibility Excellence (Weeks 10-12)

**Goal:** Meet WCAG AAA and go beyond.

| # | Task | Priority | Effort | Details |
|---|------|----------|--------|---------|
| 4.1 | i18n infrastructure + first 3 languages | High | 3 days | Set up `react-i18next` or a lightweight alternative. Extract all user-facing strings. Add Spanish, Mandarin, and Arabic as first additional languages. TTS voice selection per locale. |
| 4.2 | VoiceOver/TalkBack compatibility testing | Critical | 2 days | Test the full app flow with iOS VoiceOver and Android TalkBack. Fix focus order, announce dynamic content correctly, ensure bounding box labels don't spam screen readers. Document findings. |
| 4.3 | Per-device FOV calibration | Medium | 2 days | Add a guided first-launch calibration step: "Hold a standard credit card at arm's length and tap when the edges align with the screen." Calculate actual FOV from known card dimensions. Improves distance accuracy from ~30% to ~10% error. |
| 4.4 | Add "social" VLM mode | Medium | 2 days | New voice command: "Voice Eye, who's there?" Triggers VLM with prompt: "Describe the people in this image — how many, approximate ages, what they're doing, whether anyone is looking toward the camera." Critical for social interaction. |
| 4.5 | Ground-level hazard detection | High | 3 days | Train or fine-tune YOLO model to detect curbs, steps, uneven surfaces, puddles, and construction zones. These cause the majority of injuries for visually impaired users but are not in COCO-80. Requires custom dataset collection. |
| 4.6 | Customizable haptic patterns | Low | 1 day | Let users adjust vibration intensity (duration multiplier) and choose between vibration patterns in Settings. Some users are more sensitive to vibration than others. |
| 4.7 | Speech rate auto-adjustment by urgency | Medium | 1 day | Danger-zone announcements speak at 1.5x speed regardless of user setting. Approaching-hazard warnings speak at 1.8x. Scene descriptions use the user's chosen rate. Urgency should be conveyed through pace. |

---

## Phase 5: Ecosystem & Scale (Weeks 13+)

**Goal:** Sustainable growth and real-world impact.

| # | Task | Priority | Effort | Details |
|---|------|----------|--------|---------|
| 5.1 | Recruit 5-10 visually impaired beta testers | Critical | Ongoing | Partner with local vision impairment organizations or university accessibility labs. Structured feedback sessions with task-based testing (navigate a hallway, read a sign, find a chair). |
| 5.2 | Opt-in anonymized telemetry | High | 2 days | Track: inference FPS by device, VLM latency, most-detected classes, feature usage (voice vs tap), error rates. Use a privacy-first approach — no images or location data leave the device. |
| 5.3 | Automated model retraining pipeline | Medium | 3 days | Schedule periodic ClearML pipeline runs when new annotated data is available. Auto-evaluate against quality gate. If passes, create a PR with the new ONNX model via GitHub Actions. |
| 5.4 | Docker container for training | Medium | 2 days | Create a `Dockerfile` for the training environment. Pin CUDA, PyTorch, and Ultralytics versions. Ensures reproducibility across machines and CI. |
| 5.5 | Depth estimation (MiDaS or DepthAnything) | High | 3 days | Replace the pinhole camera model with a learned monocular depth estimator. MiDaS or DepthAnything v2 can run via ONNX. Provides per-pixel depth → much more accurate distance for all objects, not just known-height classes. |
| 5.6 | Explore WebGPU backend for ONNX Runtime | Medium | 2 days | ONNX Runtime Web supports WebGPU on Chrome 113+. Benchmark against WASM on target devices. Expect 2-3x speedup on phones with capable GPUs. Fall back to WASM if WebGPU unavailable. |
| 5.7 | Smart cane Bluetooth integration (research) | Low | Research | Investigate Bluetooth LE APIs for connecting to smart canes (WeWalk, etc). Potential: send directional haptic feedback to cane based on detection zones. Requires hardware partnership. |
| 5.8 | Indoor navigation with landmark detection | Low | Research | Explore ARCore/ARKit Instant Placement for spatial anchors. Combined with YOLO detection of signs, doors, elevators → indoor wayfinding. Major feature, likely Phase 6+. |

---

## Architecture After Phase 1

```
src/
├── config.ts                    # All tunable constants (single source of truth)
├── App.tsx                      # Slim orchestrator (~165 lines)
├── main.tsx                     # Entry point with ErrorBoundary
├── hooks/
│   ├── useDetectionLoop.ts      # Fast Lane: YOLO inference + tracking + alerts
│   ├── useVLMEngine.ts          # Slow Lane: Ollama VLM with timeout/dedup
│   ├── useVoiceRecognition.ts   # Wake word + command parsing
│   └── useSpatialAudio.ts       # TTS, beep, haptic primitives
├── components/
│   ├── CameraView.tsx           # Camera access + error recovery
│   ├── SettingsPanel.tsx         # Settings UI + localStorage
│   └── ErrorBoundary.tsx        # Crash recovery with TTS
├── utils/
│   ├── yolo.ts                  # ONNX session + pre/post-processing
│   ├── tracker.ts               # IoU tracker + velocity + proximity
│   ├── distance.ts              # Pinhole camera distance estimation
│   ├── inferenceMetrics.ts      # Latency/FPS/confidence tracking
│   ├── tracker.test.ts          # 14 tests
│   ├── distance.test.ts         # 12 tests
│   └── inferenceMetrics.test.ts # 9 tests
└── test/
    └── setup.ts                 # Vitest setup
```

---

## Success Criteria

| Metric | Current | Phase 2 Target | Phase 4 Target |
|--------|---------|----------------|----------------|
| Test coverage | ~80% on utils | 80% utils, 50% hooks | 80%+ overall |
| TypeScript strict | Enabled | Maintained | Maintained |
| Lighthouse PWA score | ~60 | 90+ | 95+ |
| Languages supported | 1 (English) | 1 | 4 |
| WCAG compliance | Partial AA | AA | AAA |
| User studies | 0 | 0 | 5-10 participants |
| Inference FPS (mid-range phone) | ~8-10 | ~8-10 | 12-15 (WebGPU) |
| VLM timeout handling | None → 15s | 15s | 15s |
| Offline Fast Lane | No | Yes | Yes |
