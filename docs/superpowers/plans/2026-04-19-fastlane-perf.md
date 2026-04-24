# Fast Lane Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise sustained Fast Lane inference rate on iPhone 15 from ~10 fps / ~120 ms p95 to ≥20 fps / ≤60 ms p95 without regressing mAP on the 48-class val set.

**Architecture:** Move ONNX inference to a dedicated Web Worker and enable the WebGPU execution provider (with WASM/SIMD fallback). Make input size a settings-tunable constant. Add an INT8-quantized model variant selected at session init. Everything is measured against the existing `inferenceMetrics` singleton so regressions are caught numerically, not by feel.

**Tech Stack:** ONNX Runtime Web 1.22+ (webgpu + wasm EPs), Vite 8 Web Worker import (`new Worker(new URL(..., import.meta.url))`), existing React 19 hook surface (`useDetectionLoop`), `onnxruntime.quantization` for model prep.

---

## Baseline and measurement

Every task ends with a recorded benchmark line so we can see the cumulative effect. Use the console log emitted by `src/utils/inferenceMetrics.ts` — it already prints `fps=… avgLatency=…ms p95=…ms detections/frame=…`. Record the line after 2 minutes of continuous use in a busy scene on iPhone 15.

### Task 0: Lock in the baseline

**Files:**
- Create: `docs/superpowers/benchmarks/fastlane-perf.md` (new, append-only log)

- [ ] **Step 1: Run current build on iPhone 15**

```bash
npm run dev -- --host
# Open HTTPS URL on phone, grant camera, let it run 2 minutes in a crowded scene.
```

- [ ] **Step 2: Capture the metrics line from the desktop console**

Copy the first `[VoiceEye] fps=... avgLatency=...ms p95=...ms detections/frame=...` line emitted after the 2-minute mark.

- [ ] **Step 3: Write benchmark log**

```markdown
# Fast Lane Perf Log

## Baseline (pre-plan)
- Date: YYYY-MM-DD
- Device: iPhone 15, iOS X.Y, Safari
- Scene: <describe>
- Metrics: fps=X.X avgLatency=XXXms p95=XXXms detections/frame=X.X
- Notes: <anything>
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/benchmarks/fastlane-perf.md
git commit -m "chore: record Fast Lane perf baseline"
```

---

## Task 1: Add WebGPU execution provider with WASM fallback

**Files:**
- Modify: `src/utils/yolo.ts` (the `getSession()` init block)
- Test: `src/utils/yolo.test.ts` (new)

**Why:** On A17 Pro / A18, WebGPU is typically 3–5× faster than WASM for a nano YOLO. `executionProviders: ['webgpu', 'wasm']` auto-falls-back on devices without WebGPU so older phones still work.

- [ ] **Step 1: Write failing test for EP selection**

```ts
// src/utils/yolo.test.ts
import { describe, it, expect, vi } from 'vitest';
import { _getRequestedExecutionProviders } from './yolo';

describe('yolo execution providers', () => {
  it('prefers webgpu, falls back to wasm', () => {
    expect(_getRequestedExecutionProviders()).toEqual(['webgpu', 'wasm']);
  });
});
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `npx vitest run src/utils/yolo.test.ts`
Expected: FAIL — `_getRequestedExecutionProviders is not a function`.

- [ ] **Step 3: Export the helper and wire it into session creation**

In `src/utils/yolo.ts`:

```ts
export function _getRequestedExecutionProviders(): string[] {
  return ['webgpu', 'wasm'];
}

// inside getSession(), replace existing create call:
session = await ort.InferenceSession.create(modelUrl, {
  executionProviders: _getRequestedExecutionProviders(),
  graphOptimizationLevel: 'all',
});
console.log('[VoiceEye] ORT session EP:', (session as any).handler?._ep ?? 'unknown');
```

- [ ] **Step 4: Test passes + manual verify on device**

Run: `npx vitest run src/utils/yolo.test.ts` → PASS.
Run: `npm run dev -- --host`; check console for `ORT session EP: webgpu` on iPhone 15.

- [ ] **Step 5: Record benchmark and commit**

Append a new section to `docs/superpowers/benchmarks/fastlane-perf.md` labelled "After Task 1 (WebGPU)".

```bash
git add src/utils/yolo.ts src/utils/yolo.test.ts docs/superpowers/benchmarks/fastlane-perf.md
git commit -m "perf(fastlane): add webgpu execution provider with wasm fallback"
```

---

## Task 2: Audit WASM threads + SIMD

**Files:**
- Modify: `src/utils/yolo.ts` (thread config)
- Modify: `vite.config.ts` (COOP/COEP headers in dev)
- Test: `src/utils/yolo.test.ts`

**Why:** WebGPU is the headline win, but Safari quirks sometimes make it fall back silently. Multi-threaded SIMD WASM must also be fast. `SharedArrayBuffer` requires cross-origin isolation headers.

- [ ] **Step 1: Write failing test for thread count logic**

```ts
// append to src/utils/yolo.test.ts
import { _computeWasmThreads } from './yolo';

describe('wasm thread config', () => {
  it('caps at hardwareConcurrency', () => {
    expect(_computeWasmThreads(4)).toBe(4);
    expect(_computeWasmThreads(16)).toBe(8); // cap at 8
    expect(_computeWasmThreads(undefined)).toBe(2);
  });
});
```

- [ ] **Step 2: Run test, confirm it fails**

Run: `npx vitest run src/utils/yolo.test.ts` → FAIL (`_computeWasmThreads is not a function`).

- [ ] **Step 3: Implement thread cap and wire env flags**

In `src/utils/yolo.ts`, before `getSession()`:

```ts
export function _computeWasmThreads(hc: number | undefined): number {
  if (!hc || hc < 1) return 2;
  return Math.min(hc, 8);
}

// in module-level init (runs once):
ort.env.wasm.numThreads = _computeWasmThreads(navigator.hardwareConcurrency);
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = '/ort-wasm/';
```

- [ ] **Step 4: Add COOP/COEP headers in dev**

In `vite.config.ts`, inside `defineConfig({ server: { ... } })`:

```ts
headers: {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
},
```

- [ ] **Step 5: Test passes + manual verify**

Run: `npx vitest run` → all pass.
Run: `npm run dev`; in devtools check `crossOriginIsolated === true` in the JS console.

- [ ] **Step 6: Record benchmark and commit**

Append "After Task 2 (threads+SIMD)" to the benchmark log.

```bash
git add src/utils/yolo.ts src/utils/yolo.test.ts vite.config.ts docs/superpowers/benchmarks/fastlane-perf.md
git commit -m "perf(fastlane): enable multi-threaded SIMD wasm and COOP/COEP"
```

---

## Task 3: Make input size configurable

**Files:**
- Modify: `src/config.ts` (add `YOLO_INPUT_SIZE`)
- Modify: `src/utils/yolo.ts` (replace hard-coded 640 with config import)
- Modify: `src/components/SettingsPanel.tsx` (optional setting; keep advanced)
- Test: extend `src/utils/yolo.test.ts`

**Why:** 480×480 is ~1.8× faster than 640×640 with a small mAP drop — worth having as a user-tunable trade-off before the INT8 model lands.

- [ ] **Step 1: Write failing test for input-size consumer**

```ts
// append to yolo.test.ts
import { _letterboxSize } from './yolo';

describe('letterbox', () => {
  it('honours configured input size', () => {
    expect(_letterboxSize(480).w).toBe(480);
    expect(_letterboxSize(480).h).toBe(480);
  });
});
```

- [ ] **Step 2: Run → FAIL**

Run: `npx vitest run src/utils/yolo.test.ts` → FAIL.

- [ ] **Step 3: Add config + refactor preprocessing**

In `src/config.ts`:

```ts
export const YOLO_INPUT_SIZE = 640; // 384 | 480 | 640
```

In `src/utils/yolo.ts`:

```ts
import { YOLO_INPUT_SIZE } from '../config';

export function _letterboxSize(size = YOLO_INPUT_SIZE) {
  return { w: size, h: size };
}
```

Replace every literal `640` in `yolo.ts` (pre-processing canvas size, tensor dims) with `_letterboxSize().w` or the imported constant.

- [ ] **Step 4: Test + build**

Run: `npx vitest run && npm run build` → both pass, zero TS errors.

- [ ] **Step 5: Record benchmark and commit**

Run the app with `YOLO_INPUT_SIZE = 480`. Record metrics line. Revert to 640 for now.

```bash
git add src/config.ts src/utils/yolo.ts src/utils/yolo.test.ts docs/superpowers/benchmarks/fastlane-perf.md
git commit -m "perf(fastlane): parameterize YOLO input size via config"
```

---

## Task 4: Move inference into a Web Worker

**Files:**
- Create: `src/workers/yolo.worker.ts`
- Modify: `src/utils/yolo.ts` (split pure helpers from session code; expose a postMessage client)
- Modify: `src/hooks/useDetectionLoop.ts` (call worker client instead of inline `runYolo`)
- Test: `src/workers/yolo.worker.test.ts` (structured-clone contract test — pure logic only)

**Why:** Main thread owns the camera, React render, and speech. Offloading inference unblocks ~15–30 ms per frame of pre/post work and lets us safely raise the 100 ms throttle.

- [ ] **Step 1: Write failing contract test**

```ts
// src/workers/yolo.worker.test.ts
import { describe, it, expect } from 'vitest';
import { _encodeRequest, _decodeResponse } from './yolo.worker.protocol';

describe('worker protocol', () => {
  it('round-trips an inference request', () => {
    const req = _encodeRequest({ frameId: 1, width: 640, height: 640, pixels: new Uint8ClampedArray(640*640*4) });
    expect(req.frameId).toBe(1);
  });
  it('decodes detection response', () => {
    const res = _decodeResponse({ frameId: 1, detections: [] });
    expect(res.detections).toEqual([]);
  });
});
```

- [ ] **Step 2: Run → FAIL**

Run: `npx vitest run src/workers/` → FAIL.

- [ ] **Step 3: Define the protocol module**

Create `src/workers/yolo.worker.protocol.ts`:

```ts
import type { Detection } from '../utils/yolo';

export interface YoloRequest {
  frameId: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
}
export interface YoloResponse {
  frameId: number;
  detections: Detection[];
}

export function _encodeRequest(r: YoloRequest): YoloRequest { return r; }
export function _decodeResponse(r: YoloResponse): YoloResponse { return r; }
```

- [ ] **Step 4: Create the worker**

Create `src/workers/yolo.worker.ts`:

```ts
/// <reference lib="webworker" />
import { runYoloFromPixels, warmup } from '../utils/yolo';
import type { YoloRequest, YoloResponse } from './yolo.worker.protocol';

declare const self: DedicatedWorkerGlobalScope;

let ready: Promise<void> | null = null;

self.onmessage = async (e: MessageEvent<YoloRequest>) => {
  if (!ready) ready = warmup();
  await ready;
  const { frameId, pixels, width, height } = e.data;
  const detections = await runYoloFromPixels(pixels, width, height);
  const res: YoloResponse = { frameId, detections };
  self.postMessage(res);
};
```

- [ ] **Step 5: Split `yolo.ts` so it runs in a worker**

Ensure `yolo.ts` does not import anything DOM-only (no `document`, no `window`). Replace `OffscreenCanvas` usage with the `self.OffscreenCanvas` global, which exists in workers. Expose:

```ts
export async function warmup(): Promise<void> { /* create session */ }
export async function runYoloFromPixels(px: Uint8ClampedArray, w: number, h: number): Promise<Detection[]> { /* ... */ }
```

- [ ] **Step 6: Wire the worker into the detection hook**

In `src/hooks/useDetectionLoop.ts`:

```ts
const workerRef = useRef<Worker | null>(null);
useEffect(() => {
  workerRef.current = new Worker(new URL('../workers/yolo.worker.ts', import.meta.url), { type: 'module' });
  return () => workerRef.current?.terminate();
}, []);

// inside the rAF callback, replace direct runYolo() with:
const pixels = ctx.getImageData(0, 0, w, h).data;
workerRef.current!.postMessage({ frameId, width: w, height: h, pixels }, [pixels.buffer]);
```

Use a pending-frame guard: drop new frames while a frame is in flight (backpressure). Handle the `message` event, then run tracker + attention on the main thread.

- [ ] **Step 7: Type-check + vitest + manual device run**

Run: `npm run build` → zero errors.
Run: `npx vitest run` → all pass.
Manual: on iPhone 15, confirm bounding boxes still appear and metrics log still emits.

- [ ] **Step 8: Record benchmark and commit**

Append "After Task 4 (Web Worker)" to the log.

```bash
git add src/workers/ src/utils/yolo.ts src/hooks/useDetectionLoop.ts docs/superpowers/benchmarks/fastlane-perf.md
git commit -m "perf(fastlane): move inference to Web Worker"
```

---

## Task 5: INT8 quantized model variant

**Files:**
- Create: `training/quantize_onnx.py`
- Modify: `training/requirements.txt` (add `onnxruntime` if missing)
- Modify: `public/models/` (add `yolo26n.int8.onnx` artifact — not committed to git; .gitignored like the fp32 one)
- Modify: `src/utils/yolo.ts` (pick quantized model when available)
- Modify: `src/config.ts` (`YOLO_MODEL_VARIANT: 'fp32' | 'int8'`)

**Why:** INT8 is typically 2–4× faster on WASM and ~25% of the size, with 1–2 mAP drop. It's the single biggest win once WebGPU is in.

- [ ] **Step 1: Write the quantizer**

Create `training/quantize_onnx.py`:

```python
"""Dynamic INT8 quantization for YOLO ONNX models."""
from __future__ import annotations
import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize(src: Path, dst: Path) -> None:
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )
    print(f"[quantize] wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--dst', type=Path, required=True)
    args = p.parse_args()
    quantize(args.src, args.dst)
```

- [ ] **Step 2: Produce the quantized artifact**

```bash
python training/quantize_onnx.py --src public/models/yolo26n.onnx --dst public/models/yolo26n.int8.onnx
```

Expect: `yolo26n.int8.onnx` is ~25% the size of the fp32 version.

- [ ] **Step 3: Add variant selection**

In `src/config.ts`:

```ts
export const YOLO_MODEL_VARIANT: 'fp32' | 'int8' = 'int8';
```

In `src/utils/yolo.ts`, pick the URL:

```ts
import { YOLO_MODEL_VARIANT } from '../config';
const modelUrl = YOLO_MODEL_VARIANT === 'int8'
  ? '/models/yolo26n.int8.onnx'
  : '/models/yolo26n.onnx';
```

- [ ] **Step 4: Validate mAP before shipping**

Run `training/train.py --val-only --weights public/models/yolo26n.int8.onnx` (or equivalent Ultralytics val on the ONNX). Record mAP50 and mAP50-95 in the benchmark log. Gate: mAP50 drop ≤ 0.02.

- [ ] **Step 5: Manual device run + record benchmark**

Verify detections still look correct on iPhone 15. Append "After Task 5 (INT8)" to the log.

```bash
git add training/quantize_onnx.py src/config.ts src/utils/yolo.ts docs/superpowers/benchmarks/fastlane-perf.md
git commit -m "perf(fastlane): add INT8 model variant selectable via config"
```

---

## Task 6: Raise the inference throttle safely

**Files:**
- Modify: `src/config.ts` (`YOLO_INFERENCE_INTERVAL_MS`)
- Modify: `src/hooks/useDetectionLoop.ts` (use the constant; remove literal `100`)

**Why:** With Worker + WebGPU + INT8 the per-frame cost should drop to ~25–40 ms. The 100 ms throttle was set for WASM-only fp32. But don't go below a sustained-safe value — A17 throttles after ~5 min if p95 > 50 ms at 30 fps.

- [ ] **Step 1: Extract the throttle constant**

In `src/config.ts`:

```ts
export const YOLO_INFERENCE_INTERVAL_MS = 50; // ~20 fps
```

In `useDetectionLoop.ts`, replace the literal `100` in the `performance.now()` gate with the imported constant.

- [ ] **Step 2: 5-minute thermal soak on iPhone 15**

Run continuously on a busy scene for 5 minutes. Record metrics every minute. If p95 climbs above 80 ms or fps sags by >20%, raise the interval to 66 ms (~15 fps) and re-soak.

- [ ] **Step 3: Record final benchmark and commit**

Append "After Task 6 (throttle)" with the stable value chosen.

```bash
git add src/config.ts src/hooks/useDetectionLoop.ts docs/superpowers/benchmarks/fastlane-perf.md
git commit -m "perf(fastlane): raise inference throttle to 20fps after thermal soak"
```

---

## Task 7: Final review

- [ ] **Step 1: Read the full benchmark log**

Confirm cumulative gain meets the goal: ≥20 fps / ≤60 ms p95 on iPhone 15, mAP50 drop ≤ 0.02.

- [ ] **Step 2: Check nothing regressed**

Run the attention + tracker + yolo test suites:

```bash
npx vitest run
npm run build
npm run lint
```

All must be clean.

- [ ] **Step 3: Update README and CLAUDE.md**

In `CLAUDE.md`, update the Fast Lane row of the architecture table:
> YOLO26n INT8 via ONNX Runtime Web (WebGPU EP, WASM fallback), inference in a Web Worker, throttled to ~20 fps.

Update the "Memory & Performance" section to reference the worker and WebGPU.

- [ ] **Step 4: Commit docs**

```bash
git add README.md CLAUDE.md
git commit -m "docs: reflect Fast Lane perf changes"
```

---

## Out of scope

- Training a pruned / distilled YOLO on the 48-class dataset. That's a separate plan that depends on `voiceeye-fastlane-v1` being uploaded and a first full-precision training run completing.
- Device-class profiles (lower-end Android, iPad). Same constants work for now; profiles become a plan when we have telemetry.
- Moving the Slow Lane off local Ollama. Tracked separately as the "cloud VLM" spec you'll write later.
