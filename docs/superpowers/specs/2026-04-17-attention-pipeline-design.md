# Attention Pipeline — Design Spec

**Date:** 2026-04-17
**Status:** Design approved, pending user review before plan writing
**Target:** iPhone 15 (v1); device-agnostic seams for weaker hardware later

## Problem

On a crowded street or in a cluttered room, YOLO26n can emit 15+ detections per frame. The current pipeline fans all of them into the tracker, the TTS queue, the haptic motor, and the bbox overlay without bound. Three failure modes result:

1. **Cognitive overload (primary).** The user hears "person, person, bicycle, person…" faster than they can process. The product becomes net-negative — worse than no announcements — because the chatter masks useful signals.
2. **Compute / FPS (secondary).** Tracker matching is O(tracks × detections); TTS queue orchestration and React re-renders grow with track count. On weaker hardware these costs compound.
3. **Thermal / battery (follow-on).** Relevant for future smart-glasses deployment; out of scope for v1 on iPhone 15.

This spec addresses #1 directly and #2 as a secondary benefit. #3 is deferred.

## Goals

- Bound announcements to a rate the user can actually act on.
- Make hazards (cars, bikes) always reach the user when they matter.
- Collapse same-class clutter ("3 people" instead of 3 names).
- Give users a single, legible verbosity control.
- Cap downstream compute (TTS, haptic, render) without touching the YOLO model or tracker algorithm.

## Non-goals

- Changing the ONNX model or swapping YOLO variants.
- Indoor/outdoor mode detection (rejected: single adaptive policy handles both).
- Adaptive inference-rate throttling (deferred to post-benchmark on weaker devices).
- Spatial audio / stereo panning (future work).

## Architecture

Insert a new stage, the **attention pipeline**, between the tracker and the announcement/render side effects in `App.tsx`.

```
runYolo() → updateTracks() → runAttention() → speak / vibrate / render
```

New module: `src/utils/attention.ts`. Pure function with external mutable state (mirrors the `tracker.ts` pattern). All "what to say" rules live here; `App.tsx` shrinks to orchestration.

### Public interface

```ts
export interface Announcement {
  kind: 'single' | 'group';
  trackIds: number[];                 // group has >1
  class: string;
  displayLabel: string;               // "person #3 ~1.2m" or "3 people ahead ~2m"
  bbox: [number, number, number, number];
  zone: ProximityZone;
  priority: number;
  reason: 'new' | 'zone-escalation' | 'approaching' | 'sustained';
}

export interface AttentionOutput {
  toAnnounce: Announcement[];         // already top-K filtered, ready to speak
  renderTracks: Track[];              // tracks that should draw bboxes
  deescalationTones: number[];        // track IDs that just exited `danger`
  suppressedCount: number;            // metrics / debug
}

export interface AttentionState {
  cooldowns: Map<number, CooldownEntry>;      // keyed by track ID
  clusters:  Map<number, ClusterEntry>;       // keyed by synthetic cluster ID
  windowStart: number;                        // ms; budget window boundary
  windowCount: number;                        // announcements emitted this window
  nextClusterId: number;
}

export function runAttention(
  tracks: Track[],
  now: number,
  state: AttentionState,              // mutated in place
  config: AttentionConfig
): AttentionOutput;
```

State is held in a ref in `App.tsx` alongside `tracksRef`. Single output struct (`Announcement`) covers both singles and groups — distinguished by `kind` + `trackIds.length`.

## The five filters

Executed in order every frame.

### 1. Hazard tier — static class map

```ts
// src/config.ts
export const HAZARD_TIER: Record<string, 1 | 2 | 3> = {
  car: 1, truck: 1, bus: 1, motorcycle: 1, bicycle: 1, train: 1,
  person: 2, dog: 2, cat: 2, horse: 2,
};
export const DEFAULT_TIER: 1 | 2 | 3 = 3;     // everything else
```

Tier 3 is not gated by a hard floor; its low weights simply cause it to lose against tier-1 and tier-2 candidates under the top-K budget. In `normal` mode (K=3) a tier-3 track will almost never emit in a busy scene. In `detailed` mode (K=6) tier-3 may emit when the scene is otherwise empty — this is intentional. COCO-80 lacks stairs/doors; custom-trained models add those to the tier map without touching other code.

### 2. Priority score — single scalar per track

```
priority = TIER_WEIGHT[tier] × ZONE_WEIGHT[zone] × (1 + approachingBoost)
```

- `TIER_WEIGHT  = { 1: 3.0, 2: 1.5, 3: 0.3 }`
- `ZONE_WEIGHT  = { danger: 3.0, near: 1.5, safe: 0.5 }`
- `approachingBoost = areaGrowthRate > 0.05 ? 1.0 : 0`

Examples: tier-1 danger approaching = 18.0; tier-3 safe = 0.15. Budget cutoff lands naturally around 1.0.

### 3. Spatial clustering — same-class collapse

- Trigger only when ≥3 tracks of the same class lie within `CLUSTER_RADIUS_FRAC` (0.15) of frame diagonal. Pairs stay individual.
- Cluster bbox = union of members; centroid = mean of member centroids; priority = max of member priorities.
- Cluster persistence across frames: match previous cluster by class + centroid distance < `CLUSTER_MATCH_FRAC` (0.10 of frame diagonal).
- Re-announce only when member count changes by ≥ `CLUSTER_COUNT_DELTA` (2). Prevents "4 people → 5 → 4" chatter.

### 4. Top-K budget — the output cap

- Sort all candidates (singles + clusters) by priority desc.
- Take top K per budget window (default 2s).
- **Verbosity setting** maps to K: `quiet → 1`, `normal → 3`, `detailed → 6`. Default: `normal`.
- **Hazard override:** a tier-1 track with `zone-escalation` reason AND a `safe → danger` transition (skipping `near`) bypasses the budget. `near → danger` does *not* bypass — only step-jumps count.

### 5. Cooldown / dedup — reason-gated re-announcements

```ts
interface CooldownEntry {
  lastZone: ProximityZone;
  lastSpokeAt: number;
  sustainedCount: number;       // capped at SUSTAINED_MAX_COUNT = 3
  lastReason: Announcement['reason'];
  lastSeenAt: number;           // for cleanup
}
```

Emission rules per track:

| Reason | Fires when | Cooldown |
|--------|------------|----------|
| `new` | Never announced before | Always |
| `zone-escalation` | `safe → near`, `near → danger`, or `safe → danger` | Always; resets `sustainedCount` |
| `approaching` | `areaGrowthRate > 0.05` and not already approaching | > 2s since last spoke |
| `sustained` | In `danger` zone for ≥ 3s with no other event | > 3s since last spoke AND `sustainedCount < 3` |

Zone de-escalation (`danger → near → safe`) emits **no TTS**. Instead, a short 440 Hz tone (50 ms) plays via the existing `AudioContext` — signals "threat passed" without consuming a budget slot. Distinct from the 880 Hz wake-word beep.

Cooldown entries are garbage-collected when `now - lastSeenAt > 30_000` ms (track gone for 30 s).

## Config surface — three layers

### Layer 1 — `src/config.ts` (developer-tunable)

```ts
export const TIER_WEIGHT:  Record<1|2|3, number>       = { 1: 3.0, 2: 1.5, 3: 0.3 };
export const ZONE_WEIGHT:  Record<ProximityZone, number> = { danger: 3.0, near: 1.5, safe: 0.5 };
export const APPROACHING_GROWTH_THRESHOLD = 0.05;
export const APPROACHING_BOOST = 1.0;

export const CLUSTER_MIN_MEMBERS  = 3;
export const CLUSTER_RADIUS_FRAC  = 0.15;
export const CLUSTER_MATCH_FRAC   = 0.10;
export const CLUSTER_COUNT_DELTA  = 2;

export const BUDGET_WINDOW_MS        = 2000;
export const COOLDOWN_APPROACHING_MS = 2000;
export const COOLDOWN_SUSTAINED_MS   = 3000;
export const SUSTAINED_ENTRY_DELAY_MS = 3000;
export const SUSTAINED_MAX_COUNT      = 3;
export const COOLDOWN_GC_MS           = 30_000;

export const DEESCALATION_TONE_HZ = 440;
export const DEESCALATION_TONE_MS = 50;

export const MAX_DETECTIONS_PER_FRAME = 40;
export const MAX_TRACKS               = 30;
export const MAX_RENDERED_BOXES       = 8;

export const VERBOSITY_K: Record<'quiet' | 'normal' | 'detailed', number> = {
  quiet: 1, normal: 3, detailed: 6,
};
```

### Layer 2 — User settings (persisted to `voiceeye_settings` in localStorage)

Add one field:

```ts
verbosity: 'quiet' | 'normal' | 'detailed';   // default: 'normal'
```

Rendered as a segmented control (three buttons) in `SettingsPanel.tsx`, under the existing haptics toggle. Mirrored into `settingsRef` for rAF-loop access.

### Layer 3 — Hardcoded (non-tunable)

- Reason hierarchy: `new` > `zone-escalation` > `approaching` > `sustained`.
- Override rule: only `safe → danger` bypasses the budget.
- Priority formula, cluster algorithm, cooldown algorithm themselves.
- Render-K independence: bboxes render for top `MAX_RENDERED_BOXES` tracks regardless of speech budget.

## Compute backpressure — four caps

All dormant on iPhone 15; load-bearing on weaker hardware. Cap 4 is deferred.

### Cap 1 — Detection count after NMS (`yolo.ts`)

```ts
return nms(detections, YOLO_IOU_THRESHOLD).slice(0, MAX_DETECTIONS_PER_FRAME); // 40
```

NMS is already score-sorted; truncation keeps highest-confidence detections.

### Cap 2 — Active tracks (`tracker.ts`)

```ts
if (updated.length > MAX_TRACKS) {
  updated.sort(byEvictionScore);     // ascending: lowest priority first
  updated.length = MAX_TRACKS;
}
```

Eviction score reuses `HAZARD_TIER` + `ZONE_WEIGHT` + track age. Tier-1 danger tracks never evict before tier-3 safe tracks.

### Cap 3 — Render-K (bbox overlay in `App.tsx`)

Draw at most `MAX_RENDERED_BOXES` (8) tracks per frame, sorted by priority desc. Independent of the speech budget — a Quiet-mode user still sees 8 boxes but hears 1.

### Cap 4 — Adaptive inference throttle (deferred)

Not in v1. Revisit after benchmarks on weaker target hardware.

## Seams for future scaling

- **Quantized / smaller model:** swap `public/models/yolo26n.onnx`; no downstream changes.
- **Region-of-interest cropping:** add in `preprocessFrame()` in `yolo.ts`.
- **Glasses (tethered) deployment:** `runYolo()` becomes an RPC boundary; attention pipeline unchanged.
- **IMU-based walking-speed scaling:** inject motion signal into the priority formula via a new term; no architectural change.

## Testing plan

### Layer 1 — unit tests (Vitest, stack already configured)

- **New file:** `src/utils/attention.test.ts`
  - Priority formula: tier-1 + danger + approaching → 18.0; tier-3 + safe → 0.15.
  - Clustering: 2 same-class → individual; 3 in radius → cluster; member drift ±1 does not re-announce; ±2 does.
  - Cooldown paths: `new` always; `zone-escalation` always; `approaching` respects 2s; `sustained` caps at 3.
  - De-escalation: never emits TTS but emits tone on `danger → near` transition.
  - Top-K: K=1 Quiet — only top priority emits; `safe → danger` bypasses; `near → danger` does not.
  - GC: cooldown entry removed after `COOLDOWN_GC_MS` of absence.

- **Extend:** `src/utils/tracker.test.ts`
  - At `MAX_TRACKS + 1`, a tier-3 safe track evicted before a tier-1 danger track.
  - Eviction stable: equal-priority tracks break ties by age.

### Layer 2 — synthetic stress scenes

Generate `Detection[]` arrays representing canonical scenarios, run 100 simulated frames through `updateTracks → runAttention`, assert on the announcement stream.

| Scene | Assertion |
|-------|-----------|
| Crowded sidewalk (15 people + 3 cars, Normal mode) | ≤ 3 announcements per 2s window; cluster "N people" appears; cars announced before people |
| Empty room (2 tier-3 chairs) | 0 announcements; bboxes render |
| Approaching car (area growing 0.08/frame) | Announced within 1s; `approaching` reason; not double-counted by `new` + `approaching` |
| Quiet mode + pedestrian safe→danger step | Bypasses K=1 budget; emits even with another announcement this window |
| Quiet mode + pedestrian near→danger gradual | Does *not* bypass; queued to next window |

Runs headless. Added to CI alongside existing Vitest tests.

### Layer 3 — real-world manual validation (pre-ship gate)

1. Residential block walk, Normal mode — log TTS count, rate overwhelm 1–5.
2. Cluttered indoor room — verify tier-3 silent; voice query "what's around?" routes correctly.
3. Busy street crossing, Quiet mode — verify tier-1 still announces on step-jump.
4. 3-min café sit — thermal check, battery drain, announcements/min.

**Pass bar:** ≤ 20 announcements/min in Normal outdoor; ≤ 30 in Detailed indoor; 0 missed tier-1 step-jumps; no thermal throttle.

### Metrics — extend `inferenceMetrics.ts`

Add to the existing 300-frame console log:

```
[VoiceEye] fps=8.2 p95=145ms dets/frame=3.1 announcements/min=14 clusters=1 suppressed/min=62
```

`suppressed/min` directly measures how much chatter the pipeline prevented. In a crowded scene it should dominate `announcements/min`.

## Open questions / future work

- **Voice-query routing.** "What's around me?" currently routes to Slow Lane (Qwen-VL); that behavior is unchanged in v1. A future enhancement could let voice queries pull a ranked tier-3 read-out from the attention pipeline directly (faster than round-tripping through Qwen). Not wired in v1, but `AttentionOutput.suppressedCount` can be extended to `suppressedTracks: Track[]` when that hook is needed.
- **Cluster labels for non-homogeneous groups.** "3 chairs and 2 bags" — ignored in v1; clusters are strictly same-class.
- **Per-class priority overrides.** Could one day tune tier weights per user (e.g. users who care more about dogs than people) — not a v1 concern.

## Files touched

| File | Change |
|------|--------|
| `src/utils/attention.ts` | **new** — pipeline, state, types |
| `src/utils/attention.test.ts` | **new** — Layer 1 + 2 tests |
| `src/utils/tracker.ts` | Add `MAX_TRACKS` eviction |
| `src/utils/tracker.test.ts` | Extend with eviction tests |
| `src/utils/yolo.ts` | Add `MAX_DETECTIONS_PER_FRAME` truncation |
| `src/utils/inferenceMetrics.ts` | Add `announcementsPerMin`, `suppressedPerMin`, `activeClusters` |
| `src/utils/inferenceMetrics.test.ts` | Update for new fields |
| `src/config.ts` | Add all Layer 1 constants and `VERBOSITY_K` |
| `src/App.tsx` | Wire `runAttention`, move announcement logic out, apply render cap |
| `src/components/SettingsPanel.tsx` | Add verbosity segmented control |
| (settings type / schema wherever defined) | Add `verbosity: 'quiet' \| 'normal' \| 'detailed'` |
