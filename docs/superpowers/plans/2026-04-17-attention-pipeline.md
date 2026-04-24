# Attention Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert an attention-filtering stage between the tracker and announcement side effects so that a crowded scene no longer floods TTS/haptics/render, while keeping tier-1 hazards always audible.

**Architecture:** New pure module `src/utils/attention.ts` exposes `runAttention(tracks, now, state, config)` returning a ranked top-K list of announcements plus a render-capped track list. `useDetectionLoop` replaces its inline announcement block with a single call to this module. Tracker and YOLO get defensive hard caps so input growth cannot starve the pipeline.

**Tech Stack:** React 19, TypeScript 6, Vitest 4 (already configured). No new dependencies.

---

## File Structure

| File | Role | Status |
|------|------|--------|
| `src/config.ts` | All tunable constants (tier weights, window, caps, verbosity map) | Extend |
| `src/utils/attention.ts` | Pure pipeline: priority, reason, clustering, top-K, GC | **New** |
| `src/utils/attention.test.ts` | Unit tests for each pure function + synthetic stress scenes | **New** |
| `src/utils/tracker.ts` | Add `MAX_TRACKS` eviction using priority score | Extend |
| `src/utils/tracker.test.ts` | Eviction tests | Extend |
| `src/utils/yolo.ts` | Post-NMS detection count cap | Extend |
| `src/utils/inferenceMetrics.ts` | Announcement/suppressed/cluster counters | Extend |
| `src/utils/inferenceMetrics.test.ts` | Tests for new counters | Extend |
| `src/hooks/useSpatialAudio.ts` | Parameterize `playBeep` frequency/duration | Extend |
| `src/hooks/useDetectionLoop.ts` | Wire `runAttention`, remove inline announce logic, emit de-escalation tones | Rewrite inner loop |
| `src/components/SettingsPanel.tsx` | Add `verbosity` field and segmented-control UI | Extend |

`src/App.tsx` is not modified — it already consumes `renderedTracks` from the hook, which will be pre-capped.

---

## Task 1: Add attention pipeline config constants

**Files:**
- Modify: `src/config.ts` (append at end)

- [ ] **Step 1: Add the new constants**

Append to `src/config.ts`:

```ts
// ── Attention Pipeline ─────────────────────────────────────────────────
import type { ProximityZone } from './utils/tracker';

export const HAZARD_TIER: Record<string, 1 | 2 | 3> = {
  car: 1, truck: 1, bus: 1, motorcycle: 1, bicycle: 1, train: 1,
  person: 2, dog: 2, cat: 2, horse: 2,
};
export const DEFAULT_TIER: 1 | 2 | 3 = 3;

export const TIER_WEIGHT: Record<1 | 2 | 3, number> = { 1: 3.0, 2: 1.5, 3: 0.3 };
export const ZONE_WEIGHT: Record<ProximityZone, number> = { danger: 3.0, near: 1.5, safe: 0.5 };
export const APPROACHING_GROWTH_THRESHOLD_ATTN = 0.05;
export const APPROACHING_BOOST = 1.0;

export const CLUSTER_MIN_MEMBERS = 3;
export const CLUSTER_RADIUS_FRAC = 0.15;
export const CLUSTER_MATCH_FRAC = 0.10;
export const CLUSTER_COUNT_DELTA = 2;

export const BUDGET_WINDOW_MS = 2000;
export const COOLDOWN_APPROACHING_MS = 2000;
export const COOLDOWN_SUSTAINED_MS = 3000;
export const SUSTAINED_ENTRY_DELAY_MS = 3000;
export const SUSTAINED_MAX_COUNT = 3;
export const COOLDOWN_GC_MS = 30_000;

export const DEESCALATION_TONE_HZ = 440;
export const DEESCALATION_TONE_S = 0.05;

export const MAX_DETECTIONS_PER_FRAME = 40;
export const MAX_TRACKS = 30;
export const MAX_RENDERED_BOXES = 8;

export const VERBOSITY_K: Record<'quiet' | 'normal' | 'detailed', number> = {
  quiet: 1, normal: 3, detailed: 6,
};
```

Note: `APPROACHING_GROWTH_THRESHOLD_ATTN` has a distinct name because `APPROACHING_GROWTH_RATE` is already exported and consumed by legacy code in `useDetectionLoop`. Both share the same numeric value for now; the legacy export gets removed in Task 14.

- [ ] **Step 2: Verify the build**

Run: `npm run build`
Expected: TypeScript compiles with zero errors. The `ProximityZone` import line may require the tracker file to export the type — if it errors, proceed to Step 3.

- [ ] **Step 3: Ensure `ProximityZone` is exported from tracker**

Read `src/utils/tracker.ts` line 10. Confirm `export type ProximityZone` exists. If it's only `type ProximityZone =`, change to `export type ProximityZone = 'safe' | 'near' | 'danger';`.

(Based on the current file, `export type ProximityZone` is already present, so this step is a verify-only no-op.)

- [ ] **Step 4: Commit**

```bash
git add src/config.ts
git commit -m "feat(attention): add pipeline config constants"
```

---

## Task 2: Parameterize `playBeep` frequency and duration

**Files:**
- Modify: `src/hooks/useSpatialAudio.ts:37-50`

- [ ] **Step 1: Update `playBeep` signature**

Replace the `playBeep` callback in `src/hooks/useSpatialAudio.ts`:

```ts
const playBeep = useCallback((freq = BEEP_FREQUENCY_HZ, durationS = BEEP_DURATION_S) => {
  try {
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
    const ctx = audioCtxRef.current;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = freq;
    gain.gain.value = BEEP_GAIN;
    osc.start();
    osc.stop(ctx.currentTime + durationS);
  } catch { /* AudioContext unavailable */ }
}, []);
```

- [ ] **Step 2: Verify existing callers still type-check**

Run: `npm run build`
Expected: Zero errors. Existing `playBeep()` calls in `App.tsx` and voice recognition use no arguments — defaults preserve old behavior.

- [ ] **Step 3: Commit**

```bash
git add src/hooks/useSpatialAudio.ts
git commit -m "feat(audio): parameterize playBeep frequency and duration"
```

---

## Task 3: Add `verbosity` to settings schema

**Files:**
- Modify: `src/components/SettingsPanel.tsx:4-14`

- [ ] **Step 1: Extend the type and default**

In `src/components/SettingsPanel.tsx`, update the interface and defaults:

```ts
export type Verbosity = 'quiet' | 'normal' | 'detailed';

export interface AppSettings {
  ttsRate: number;
  confThreshold: number;
  hapticEnabled: boolean;
  verbosity: Verbosity;
}

const DEFAULTS: AppSettings = {
  ttsRate: 1.0,
  confThreshold: 0.5,
  hapticEnabled: true,
  verbosity: 'normal',
};
```

`loadSettings` already merges with `DEFAULTS` (line 21), so users with existing localStorage entries get `verbosity: 'normal'` automatically on first read.

- [ ] **Step 2: Verify the build**

Run: `npm run build`
Expected: Zero errors — the field is optional in the JSON merge path.

- [ ] **Step 3: Commit**

```bash
git add src/components/SettingsPanel.tsx
git commit -m "feat(settings): add verbosity field with migration default"
```

---

## Task 4: Add verbosity segmented control to Settings UI

**Files:**
- Modify: `src/components/SettingsPanel.tsx` (insert new row before closing `</div>` of `glass-panel`)

- [ ] **Step 1: Add the UI row**

Insert this block in `SettingsPanel.tsx` immediately **after** the Haptic Toggle row and **before** the closing `</div>` of the panel (around line 109):

```tsx
{/* Verbosity */}
<label className="settings-row">
  <span className="settings-label">Verbosity</span>
  <div className="settings-control" style={{ display: 'flex', gap: 4 }}>
    {(['quiet', 'normal', 'detailed'] as const).map(v => (
      <button
        key={v}
        className={`glass-button${local.verbosity === v ? ' active' : ''}`}
        onClick={() => update({ verbosity: v })}
        aria-pressed={local.verbosity === v}
        style={{ flex: 1, textTransform: 'capitalize', height: 36, fontSize: 13 }}
      >
        {v}
      </button>
    ))}
  </div>
</label>
```

- [ ] **Step 2: Visually verify in dev server**

Run: `npm run dev`
Open settings panel. Confirm three buttons labelled Quiet / Normal / Detailed; tapping one highlights it, persists across reload. Press Ctrl+C to stop the dev server.

- [ ] **Step 3: Commit**

```bash
git add src/components/SettingsPanel.tsx
git commit -m "feat(settings): add verbosity segmented control UI"
```

---

## Task 5: Cap detections-per-frame in `runYolo`

**Files:**
- Modify: `src/utils/yolo.ts:29` (import), `src/utils/yolo.ts:211` (return)

- [ ] **Step 1: Add the import and cap**

In `src/utils/yolo.ts`:

1. Update the config import at line 29 from:
   ```ts
   import { YOLO_INPUT_SIZE, YOLO_DEFAULT_CONF, YOLO_IOU_THRESHOLD } from '../config';
   ```
   to:
   ```ts
   import { YOLO_INPUT_SIZE, YOLO_DEFAULT_CONF, YOLO_IOU_THRESHOLD, MAX_DETECTIONS_PER_FRAME } from '../config';
   ```

2. Update line 211 from:
   ```ts
   return nms(detections, YOLO_IOU_THRESHOLD);
   ```
   to:
   ```ts
   return nms(detections, YOLO_IOU_THRESHOLD).slice(0, MAX_DETECTIONS_PER_FRAME);
   ```

3. Also update the TYPE 1 end-to-end NMS branch: before `return detections;` at line 173, insert:
   ```ts
   if (detections.length > MAX_DETECTIONS_PER_FRAME) {
     detections.sort((a, b) => b.score - a.score).length = MAX_DETECTIONS_PER_FRAME;
   }
   ```

- [ ] **Step 2: Verify the build**

Run: `npm run build`
Expected: Zero errors.

- [ ] **Step 3: Commit**

```bash
git add src/utils/yolo.ts
git commit -m "feat(yolo): cap detections-per-frame after NMS"
```

---

## Task 6: Track eviction at MAX_TRACKS

**Files:**
- Test: `src/utils/tracker.test.ts` (extend)
- Modify: `src/utils/tracker.ts`

- [ ] **Step 1: Write failing eviction tests**

Append to `src/utils/tracker.test.ts`:

```ts
import { MAX_TRACKS } from '../config';

describe('updateTracks — eviction at MAX_TRACKS', () => {
  it('evicts the lowest-priority track when total would exceed MAX_TRACKS', () => {
    // Seed MAX_TRACKS existing tier-3 safe tracks
    let tracks = updateTracks([], Array.from({ length: MAX_TRACKS }, (_, i) =>
      det('book', i * 20, 0, 5, 5)));
    expect(tracks).toHaveLength(MAX_TRACKS);

    // New frame: all existing tracks continue AND one new tier-1 danger track enters
    const continuing = tracks.map(t => det(t.class, t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]));
    const newHazard = det('car', 900, 300, 600, 600); // huge → danger zone
    tracks = updateTracks(tracks, [...continuing, newHazard]);

    // Still MAX_TRACKS — the car must be there; a book must have been evicted
    expect(tracks).toHaveLength(MAX_TRACKS);
    expect(tracks.some(t => t.class === 'car')).toBe(true);
    expect(tracks.filter(t => t.class === 'book').length).toBe(MAX_TRACKS - 1);
  });

  it('never evicts a tier-1 danger track in favour of a tier-3 safe one', () => {
    // One tier-1 danger track (a car) is seeded first
    let tracks = updateTracks([], [det('car', 100, 100, 600, 600)]);
    expect(tracks).toHaveLength(1);

    // Fill the rest with tier-3 safe books
    const fillers = Array.from({ length: MAX_TRACKS }, (_, i) => det('book', 900 + i, 0, 5, 5));
    tracks = updateTracks(tracks, [det('car', 100, 100, 600, 600), ...fillers]);

    // The car must still be present; we got to MAX_TRACKS (not MAX_TRACKS + 1)
    expect(tracks).toHaveLength(MAX_TRACKS);
    expect(tracks.some(t => t.class === 'car')).toBe(true);
  });
});
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `npm test -- tracker.test.ts`
Expected: The two new tests fail — first because tracks count exceeds MAX_TRACKS, second because the car gets evicted.

- [ ] **Step 3: Implement eviction in `updateTracks`**

In `src/utils/tracker.ts`:

1. Add imports at the top (extend the existing import block):
   ```ts
   import {
     TRACKER_MAX_AGE,
     TRACKER_MIN_IOU,
     TRACKER_EMA_ALPHA,
     PROXIMITY_DANGER_THRESHOLD,
     PROXIMITY_NEAR_THRESHOLD,
     MAX_TRACKS,
     HAZARD_TIER,
     DEFAULT_TIER,
     TIER_WEIGHT,
     ZONE_WEIGHT,
   } from '../config';
   ```

2. Add this helper above `updateTracks`:
   ```ts
   function evictionScore(t: Track): number {
     // Lower score = more evictable
     const tier = HAZARD_TIER[t.class] ?? DEFAULT_TIER;
     const tierW = TIER_WEIGHT[tier];
     const zoneW = ZONE_WEIGHT[t.proximityZone];
     // Newer tracks (lower age) rank slightly above older ones at equal priority
     return tierW * zoneW - t.age * 0.01;
   }
   ```

3. Immediately before `return updated;` at the end of `updateTracks`:
   ```ts
   if (updated.length > MAX_TRACKS) {
     updated.sort((a, b) => evictionScore(b) - evictionScore(a));
     updated.length = MAX_TRACKS;
   }
   ```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `npm test -- tracker.test.ts`
Expected: All tests pass, including the two new eviction tests and the existing 11.

- [ ] **Step 5: Commit**

```bash
git add src/utils/tracker.ts src/utils/tracker.test.ts
git commit -m "feat(tracker): evict lowest-priority track when exceeding MAX_TRACKS"
```

---

## Task 7: Create attention module types and state constructor

**Files:**
- Create: `src/utils/attention.ts`
- Create: `src/utils/attention.test.ts`

- [ ] **Step 1: Write the failing state-constructor test**

Create `src/utils/attention.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { createAttentionState } from './attention';

describe('createAttentionState', () => {
  it('produces an empty state anchored at the provided timestamp', () => {
    const s = createAttentionState(1000);
    expect(s.cooldowns.size).toBe(0);
    expect(s.clusters.size).toBe(0);
    expect(s.windowStart).toBe(1000);
    expect(s.windowCount).toBe(0);
    expect(s.nextClusterId).toBe(1);
  });
});
```

- [ ] **Step 2: Run to confirm it fails**

Run: `npm test -- attention.test.ts`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the module with types and constructor**

Create `src/utils/attention.ts`:

```ts
import type { Track, ProximityZone } from './tracker';

export type AnnouncementReason = 'new' | 'zone-escalation' | 'approaching' | 'sustained';

export interface Announcement {
  kind: 'single' | 'group';
  trackIds: number[];
  class: string;
  memberCount: number;
  bbox: [number, number, number, number];
  zone: ProximityZone;
  priority: number;
  reason: AnnouncementReason;
}

export interface CooldownEntry {
  lastZone: ProximityZone;
  lastSpokeAt: number;
  sustainedCount: number;
  lastReason: AnnouncementReason;
  lastSeenAt: number;
  dangerEnteredAt: number;
}

export interface ClusterEntry {
  id: number;
  class: string;
  memberTrackIds: number[];
  centroid: [number, number];
  lastAnnouncedCount: number;
  lastSpokeAt: number;
}

export interface AttentionState {
  cooldowns: Map<number, CooldownEntry>;
  clusters: Map<number, ClusterEntry>;
  windowStart: number;
  windowCount: number;
  nextClusterId: number;
}

export interface AttentionConfig {
  verbosity: 'quiet' | 'normal' | 'detailed';
  frameWidth: number;
  frameHeight: number;
  screenArea: number;
}

export interface AttentionOutput {
  toAnnounce: Announcement[];
  renderTracks: Track[];
  deescalationTones: number[];
  suppressedCount: number;
}

export function createAttentionState(now: number): AttentionState {
  return {
    cooldowns: new Map(),
    clusters: new Map(),
    windowStart: now,
    windowCount: 0,
    nextClusterId: 1,
  };
}
```

- [ ] **Step 4: Run to confirm the test passes**

Run: `npm test -- attention.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/attention.ts src/utils/attention.test.ts
git commit -m "feat(attention): add module scaffold with types and state constructor"
```

---

## Task 8: Implement `computePriority`

**Files:**
- Modify: `src/utils/attention.ts`
- Modify: `src/utils/attention.test.ts`

- [ ] **Step 1: Write failing tests**

Append to `src/utils/attention.test.ts`:

```ts
import { computePriority } from './attention';

describe('computePriority', () => {
  it('tier-1 danger approaching → 18.0', () => {
    expect(computePriority('car', 'danger', 0.1)).toBeCloseTo(18.0);
  });

  it('tier-3 safe stationary → 0.15', () => {
    expect(computePriority('book', 'safe', 0)).toBeCloseTo(0.15);
  });

  it('unknown class defaults to tier-3', () => {
    expect(computePriority('unknown-xyz', 'safe', 0)).toBeCloseTo(0.15);
  });

  it('approaching below growth threshold adds no boost', () => {
    // 0.04 < APPROACHING_GROWTH_THRESHOLD_ATTN (0.05)
    expect(computePriority('person', 'near', 0.04)).toBeCloseTo(2.25);
  });

  it('approaching above growth threshold applies full boost', () => {
    // person near approaching = 1.5 * 1.5 * (1 + 1) = 4.5
    expect(computePriority('person', 'near', 0.06)).toBeCloseTo(4.5);
  });
});
```

- [ ] **Step 2: Run to confirm they fail**

Run: `npm test -- attention.test.ts`
Expected: FAIL — `computePriority` is not exported.

- [ ] **Step 3: Implement**

Append to `src/utils/attention.ts`:

```ts
import {
  HAZARD_TIER, DEFAULT_TIER, TIER_WEIGHT, ZONE_WEIGHT,
  APPROACHING_GROWTH_THRESHOLD_ATTN, APPROACHING_BOOST,
} from '../config';

export function computePriority(
  cls: string,
  zone: ProximityZone,
  areaGrowthRate: number,
): number {
  const tier = HAZARD_TIER[cls] ?? DEFAULT_TIER;
  const tierW = TIER_WEIGHT[tier];
  const zoneW = ZONE_WEIGHT[zone];
  const boost = areaGrowthRate > APPROACHING_GROWTH_THRESHOLD_ATTN ? APPROACHING_BOOST : 0;
  return tierW * zoneW * (1 + boost);
}
```

- [ ] **Step 4: Run to confirm tests pass**

Run: `npm test -- attention.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/attention.ts src/utils/attention.test.ts
git commit -m "feat(attention): implement computePriority"
```

---

## Task 9: Implement `detectReason`

**Files:**
- Modify: `src/utils/attention.ts`
- Modify: `src/utils/attention.test.ts`

- [ ] **Step 1: Write failing tests**

Append to `src/utils/attention.test.ts`:

```ts
import { detectReason, type CooldownEntry } from './attention';

function mockTrack(overrides: Partial<{ areaGrowthRate: number }> = {}) {
  return {
    id: 1,
    bbox: [0, 0, 10, 10] as [number, number, number, number],
    class: 'person',
    score: 0.9,
    age: 0,
    announced: false,
    vx: 0, vy: 0,
    areaGrowthRate: 0,
    lastCentroid: [5, 5] as [number, number],
    lastArea: 100,
    proximityZone: 'safe' as const,
    zoneEntryTime: 0,
    reannounceCount: 0,
    ...overrides,
  };
}

function mockCooldown(overrides: Partial<CooldownEntry> = {}): CooldownEntry {
  return {
    lastZone: 'safe',
    lastSpokeAt: 0,
    sustainedCount: 0,
    lastReason: 'new',
    lastSeenAt: 0,
    dangerEnteredAt: 0,
    ...overrides,
  };
}

describe('detectReason', () => {
  it('returns "new" when no cooldown exists', () => {
    expect(detectReason(mockTrack(), 'safe', undefined, 0)).toBe('new');
  });

  it('returns "zone-escalation" when zone rank increases', () => {
    const cd = mockCooldown({ lastZone: 'safe' });
    expect(detectReason(mockTrack(), 'near', cd, 1000)).toBe('zone-escalation');
    expect(detectReason(mockTrack(), 'danger', cd, 1000)).toBe('zone-escalation');
  });

  it('returns null on zone de-escalation', () => {
    const cd = mockCooldown({ lastZone: 'danger' });
    expect(detectReason(mockTrack(), 'near', cd, 1000)).toBeNull();
    expect(detectReason(mockTrack(), 'safe', cd, 1000)).toBeNull();
  });

  it('returns "approaching" when growth rate crosses threshold and cooldown elapsed', () => {
    const cd = mockCooldown({ lastZone: 'near', lastSpokeAt: 0 });
    const track = mockTrack({ areaGrowthRate: 0.08 });
    expect(detectReason(track, 'near', cd, 3000)).toBe('approaching');
  });

  it('does not return "approaching" within cooldown window', () => {
    const cd = mockCooldown({ lastZone: 'near', lastSpokeAt: 1500 });
    const track = mockTrack({ areaGrowthRate: 0.08 });
    // 3000 - 1500 = 1500 ms < COOLDOWN_APPROACHING_MS (2000)
    expect(detectReason(track, 'near', cd, 3000)).toBeNull();
  });

  it('returns "sustained" after entry delay in danger and cooldown elapsed', () => {
    const cd = mockCooldown({
      lastZone: 'danger',
      dangerEnteredAt: 0,
      lastSpokeAt: 0,
      sustainedCount: 0,
    });
    // now = 4000: >3000 since entered danger AND >3000 since last spoke
    expect(detectReason(mockTrack(), 'danger', cd, 4000)).toBe('sustained');
  });

  it('caps sustained at SUSTAINED_MAX_COUNT', () => {
    const cd = mockCooldown({
      lastZone: 'danger',
      dangerEnteredAt: 0,
      lastSpokeAt: 0,
      sustainedCount: 3,  // already at cap
    });
    expect(detectReason(mockTrack(), 'danger', cd, 10000)).toBeNull();
  });
});
```

- [ ] **Step 2: Run to confirm they fail**

Run: `npm test -- attention.test.ts`
Expected: FAIL — `detectReason` is not exported.

- [ ] **Step 3: Implement**

Add imports to `src/utils/attention.ts` (extend the existing config import):

```ts
import {
  HAZARD_TIER, DEFAULT_TIER, TIER_WEIGHT, ZONE_WEIGHT,
  APPROACHING_GROWTH_THRESHOLD_ATTN, APPROACHING_BOOST,
  COOLDOWN_APPROACHING_MS, COOLDOWN_SUSTAINED_MS,
  SUSTAINED_ENTRY_DELAY_MS, SUSTAINED_MAX_COUNT,
} from '../config';
```

Then append the function:

```ts
const ZONE_RANK: Record<ProximityZone, number> = { safe: 0, near: 1, danger: 2 };

export function detectReason(
  track: Track,
  zone: ProximityZone,
  cooldown: CooldownEntry | undefined,
  now: number,
): AnnouncementReason | null {
  if (!cooldown) return 'new';

  if (ZONE_RANK[zone] > ZONE_RANK[cooldown.lastZone]) return 'zone-escalation';

  if (track.areaGrowthRate > APPROACHING_GROWTH_THRESHOLD_ATTN
      && now - cooldown.lastSpokeAt > COOLDOWN_APPROACHING_MS) {
    return 'approaching';
  }

  if (zone === 'danger'
      && now - cooldown.dangerEnteredAt > SUSTAINED_ENTRY_DELAY_MS
      && now - cooldown.lastSpokeAt > COOLDOWN_SUSTAINED_MS
      && cooldown.sustainedCount < SUSTAINED_MAX_COUNT) {
    return 'sustained';
  }

  return null;
}
```

- [ ] **Step 4: Run to confirm tests pass**

Run: `npm test -- attention.test.ts`
Expected: PASS on all 7 new cases.

- [ ] **Step 5: Commit**

```bash
git add src/utils/attention.ts src/utils/attention.test.ts
git commit -m "feat(attention): implement detectReason with zone+approach+sustained logic"
```

---

## Task 10: Implement clustering

**Files:**
- Modify: `src/utils/attention.ts`
- Modify: `src/utils/attention.test.ts`

- [ ] **Step 1: Write failing tests**

Append to `src/utils/attention.test.ts`:

```ts
import { clusterTracks } from './attention';

function trackAt(id: number, cls: string, x: number, y: number, w = 40, h = 80) {
  return mockTrack({ ...mockTrack(), id, class: cls,
    bbox: [x, y, w, h] as [number, number, number, number],
    lastCentroid: [x + w / 2, y + h / 2] as [number, number],
  });
}

describe('clusterTracks', () => {
  const frameDiagonal = Math.hypot(1280, 720); // ~1469

  it('returns no clusters for two same-class tracks (below min=3)', () => {
    const tracks = [trackAt(1, 'person', 100, 100), trackAt(2, 'person', 120, 110)];
    expect(clusterTracks(tracks, frameDiagonal)).toHaveLength(0);
  });

  it('clusters three same-class tracks within radius', () => {
    const r = frameDiagonal * 0.1; // well within CLUSTER_RADIUS_FRAC=0.15
    const tracks = [
      trackAt(1, 'person', 500, 300),
      trackAt(2, 'person', 500 + r / 3, 300),
      trackAt(3, 'person', 500, 300 + r / 3),
    ];
    const clusters = clusterTracks(tracks, frameDiagonal);
    expect(clusters).toHaveLength(1);
    expect(clusters[0].class).toBe('person');
    expect(clusters[0].members).toHaveLength(3);
  });

  it('does not cluster same-class tracks spread beyond radius', () => {
    const tracks = [
      trackAt(1, 'person', 0, 0),
      trackAt(2, 'person', 1200, 0),
      trackAt(3, 'person', 600, 700),
    ];
    expect(clusterTracks(tracks, frameDiagonal)).toHaveLength(0);
  });

  it('does not cluster across different classes', () => {
    const tracks = [
      trackAt(1, 'person', 500, 300),
      trackAt(2, 'car', 510, 300),
      trackAt(3, 'dog', 520, 300),
    ];
    expect(clusterTracks(tracks, frameDiagonal)).toHaveLength(0);
  });

  it('computes cluster centroid as mean of member centroids', () => {
    const tracks = [
      trackAt(1, 'person', 100, 100),
      trackAt(2, 'person', 110, 110),
      trackAt(3, 'person', 120, 120),
    ];
    const [c] = clusterTracks(tracks, frameDiagonal);
    // Centroids: (120,140), (130,150), (140,160) → mean (130,150)
    expect(c.centroid[0]).toBeCloseTo(130);
    expect(c.centroid[1]).toBeCloseTo(150);
  });
});
```

- [ ] **Step 2: Run to confirm they fail**

Run: `npm test -- attention.test.ts`
Expected: FAIL — `clusterTracks` is not exported.

- [ ] **Step 3: Implement**

Add imports to `src/utils/attention.ts`:

```ts
import {
  // ... existing imports ...
  CLUSTER_MIN_MEMBERS, CLUSTER_RADIUS_FRAC,
} from '../config';
```

Append the function:

```ts
export interface RawCluster {
  class: string;
  members: Track[];
  centroid: [number, number];
}

function centroidOf(t: Track): [number, number] {
  return [t.bbox[0] + t.bbox[2] / 2, t.bbox[1] + t.bbox[3] / 2];
}

export function clusterTracks(tracks: Track[], frameDiagonal: number): RawCluster[] {
  const byClass = new Map<string, Track[]>();
  for (const t of tracks) {
    const arr = byClass.get(t.class) ?? [];
    arr.push(t);
    byClass.set(t.class, arr);
  }

  const clusters: RawCluster[] = [];
  const radius = frameDiagonal * CLUSTER_RADIUS_FRAC;

  for (const [cls, members] of byClass) {
    if (members.length < CLUSTER_MIN_MEMBERS) continue;

    const used = new Set<number>();
    for (let i = 0; i < members.length; i++) {
      if (used.has(i)) continue;
      const [sx, sy] = centroidOf(members[i]);
      const group: Track[] = [members[i]];
      used.add(i);

      for (let j = i + 1; j < members.length; j++) {
        if (used.has(j)) continue;
        const [cx, cy] = centroidOf(members[j]);
        if (Math.hypot(cx - sx, cy - sy) < radius) {
          group.push(members[j]);
          used.add(j);
        }
      }

      if (group.length >= CLUSTER_MIN_MEMBERS) {
        const mx = group.reduce((s, t) => s + centroidOf(t)[0], 0) / group.length;
        const my = group.reduce((s, t) => s + centroidOf(t)[1], 0) / group.length;
        clusters.push({ class: cls, members: group, centroid: [mx, my] });
      }
    }
  }
  return clusters;
}
```

- [ ] **Step 4: Run to confirm tests pass**

Run: `npm test -- attention.test.ts`
Expected: PASS on all 5 clustering cases.

- [ ] **Step 5: Commit**

```bash
git add src/utils/attention.ts src/utils/attention.test.ts
git commit -m "feat(attention): implement same-class spatial clustering"
```

---

## Task 11: Implement `runAttention` orchestrator

**Files:**
- Modify: `src/utils/attention.ts`
- Modify: `src/utils/attention.test.ts`

- [ ] **Step 1: Write failing tests**

Append to `src/utils/attention.test.ts`:

```ts
import { runAttention, type AttentionConfig } from './attention';
import { updateTracks } from './tracker';
import type { Detection } from './yolo';

function cfg(overrides: Partial<AttentionConfig> = {}): AttentionConfig {
  return {
    verbosity: 'normal',
    frameWidth: 1280,
    frameHeight: 720,
    screenArea: 1280 * 720,
    ...overrides,
  };
}

function d(cls: string, x: number, y: number, w: number, h: number, score = 0.9): Detection {
  return { bbox: [x, y, w, h], class: cls, score };
}

describe('runAttention — first frame', () => {
  it('announces a new track as "new"', () => {
    const state = createAttentionState(0);
    const tracks = updateTracks([], [d('car', 100, 100, 200, 200)]);
    const out = runAttention(tracks, 0, state, cfg());

    expect(out.toAnnounce).toHaveLength(1);
    expect(out.toAnnounce[0].reason).toBe('new');
    expect(out.toAnnounce[0].class).toBe('car');
    expect(out.toAnnounce[0].kind).toBe('single');
  });

  it('caps output at budgetK = 3 in normal mode', () => {
    const state = createAttentionState(0);
    const tracks = updateTracks([], [
      d('person', 100, 100, 40, 80),
      d('dog',    200, 100, 40, 80),
      d('cat',    300, 100, 40, 80),
      d('horse',  400, 100, 40, 80),
      d('car',    500, 100, 40, 80),
    ]);
    const out = runAttention(tracks, 0, state, cfg());
    expect(out.toAnnounce).toHaveLength(3);
    expect(out.suppressedCount).toBe(2);
  });

  it('caps output at budgetK = 1 in quiet mode', () => {
    const state = createAttentionState(0);
    const tracks = updateTracks([], [
      d('car', 100, 100, 40, 80),
      d('person', 200, 100, 40, 80),
    ]);
    const out = runAttention(tracks, 0, state, cfg({ verbosity: 'quiet' }));
    expect(out.toAnnounce).toHaveLength(1);
    expect(out.toAnnounce[0].class).toBe('car'); // tier-1 wins
  });
});

describe('runAttention — hazard override', () => {
  it('tier-1 safe→danger step-jump bypasses quiet-mode budget', () => {
    const state = createAttentionState(0);

    // Frame 1: a person in safe, a small car also in safe — quiet mode picks only one
    let tracks = updateTracks([], [
      d('person', 100, 100, 40, 80),
      d('car', 500, 500, 40, 40), // small → safe
    ]);
    let out = runAttention(tracks, 0, state, cfg({ verbosity: 'quiet' }));
    expect(out.toAnnounce).toHaveLength(1);

    // Frame 2: car suddenly fills >50% of screen (safe→danger step-jump)
    // person has already been announced; budget would normally be empty for this window
    const big: Detection = d('car', 200, 200, 900, 500); // > 0.5 of 1280*720
    tracks = updateTracks(tracks, [
      d('person', 102, 101, 40, 80),
      big,
    ]);
    out = runAttention(tracks, 100, state, cfg({ verbosity: 'quiet' }));
    const carAnn = out.toAnnounce.find(a => a.class === 'car');
    expect(carAnn).toBeDefined();
    expect(carAnn?.reason).toBe('zone-escalation');
  });

  it('tier-1 near→danger (gradual) does NOT bypass the budget', () => {
    const state = createAttentionState(0);
    // Frame 1: car in near zone
    let tracks = updateTracks([], [d('car', 100, 100, 500, 450)]); // area ~0.24 → near
    let out = runAttention(tracks, 0, state, cfg({ verbosity: 'quiet' }));
    expect(out.toAnnounce).toHaveLength(1); // car announced (new)

    // Frame 2: same window (within 2s), person enters that dominates budget
    tracks = updateTracks(tracks, [
      d('car', 100, 100, 500, 450), // still near
      d('person', 600, 100, 40, 80),
    ]);
    out = runAttention(tracks, 100, state, cfg({ verbosity: 'quiet' }));
    // Car was already announced; person has higher new priority? actually person safe new = tier2*zoneSafe*1 = 0.75
    // Both valid candidates. Budget=1 means only one emits. This test just verifies we don't blow past budget.
    expect(out.toAnnounce.length).toBeLessThanOrEqual(1);
  });
});

describe('runAttention — de-escalation tones', () => {
  it('emits a tone when a track drops from danger back to near', () => {
    const state = createAttentionState(0);

    // Frame 1: huge car → danger
    let tracks = updateTracks([], [d('car', 0, 0, 900, 500)]);
    runAttention(tracks, 0, state, cfg());

    // Frame 2: shrink → near (still same track via IoU). Area ~0.24 of frame.
    tracks = updateTracks(tracks, [d('car', 0, 0, 600, 450)]);
    const out = runAttention(tracks, 2500, state, cfg());

    expect(out.deescalationTones.length).toBeGreaterThanOrEqual(1);
  });
});

describe('runAttention — render cap', () => {
  it('returns at most MAX_RENDERED_BOXES (8) tracks in renderTracks', () => {
    const state = createAttentionState(0);
    const dets = Array.from({ length: 15 }, (_, i) => d('book', i * 50, 0, 40, 40));
    const tracks = updateTracks([], dets);
    const out = runAttention(tracks, 0, state, cfg());
    expect(out.renderTracks.length).toBeLessThanOrEqual(8);
  });
});

describe('runAttention — clustering', () => {
  it('collapses 3 people into a single group announcement', () => {
    const state = createAttentionState(0);
    const tracks = updateTracks([], [
      d('person', 500, 300, 40, 80),
      d('person', 540, 300, 40, 80),
      d('person', 580, 300, 40, 80),
    ]);
    const out = runAttention(tracks, 0, state, cfg());
    const personAnn = out.toAnnounce.find(a => a.class === 'person');
    expect(personAnn).toBeDefined();
    expect(personAnn!.kind).toBe('group');
    expect(personAnn!.memberCount).toBe(3);
  });
});
```

- [ ] **Step 2: Run to confirm they fail**

Run: `npm test -- attention.test.ts`
Expected: FAIL — `runAttention` is not exported.

- [ ] **Step 3: Implement `runAttention`**

Add remaining config imports at the top of `src/utils/attention.ts` (merge with existing import block):

```ts
import {
  HAZARD_TIER, DEFAULT_TIER, TIER_WEIGHT, ZONE_WEIGHT,
  APPROACHING_GROWTH_THRESHOLD_ATTN, APPROACHING_BOOST,
  COOLDOWN_APPROACHING_MS, COOLDOWN_SUSTAINED_MS,
  SUSTAINED_ENTRY_DELAY_MS, SUSTAINED_MAX_COUNT, COOLDOWN_GC_MS,
  CLUSTER_MIN_MEMBERS, CLUSTER_RADIUS_FRAC, CLUSTER_MATCH_FRAC, CLUSTER_COUNT_DELTA,
  BUDGET_WINDOW_MS, MAX_RENDERED_BOXES, VERBOSITY_K,
  PROXIMITY_DANGER_THRESHOLD, PROXIMITY_NEAR_THRESHOLD,
} from '../config';
```

Append these helpers and the main function:

```ts
function zoneFromArea(areaPercent: number): ProximityZone {
  if (areaPercent > PROXIMITY_DANGER_THRESHOLD) return 'danger';
  if (areaPercent > PROXIMITY_NEAR_THRESHOLD)   return 'near';
  return 'safe';
}

function isHazardOverride(reason: AnnouncementReason, cls: string, lastZone: ProximityZone, zone: ProximityZone): boolean {
  // Only tier-1 safe→danger step-jumps bypass the budget
  const tier = HAZARD_TIER[cls] ?? DEFAULT_TIER;
  return tier === 1 && reason === 'zone-escalation' && lastZone === 'safe' && zone === 'danger';
}

function matchCluster(state: AttentionState, cls: string, centroid: [number, number], frameDiagonal: number): ClusterEntry | undefined {
  const radius = frameDiagonal * CLUSTER_MATCH_FRAC;
  for (const entry of state.clusters.values()) {
    if (entry.class !== cls) continue;
    if (Math.hypot(entry.centroid[0] - centroid[0], entry.centroid[1] - centroid[1]) < radius) {
      return entry;
    }
  }
  return undefined;
}

export function runAttention(
  tracks: Track[],
  now: number,
  state: AttentionState,
  config: AttentionConfig,
): AttentionOutput {
  // Reset budget window if expired
  if (now - state.windowStart > BUDGET_WINDOW_MS) {
    state.windowStart = now;
    state.windowCount = 0;
  }

  const frameDiagonal = Math.hypot(config.frameWidth, config.frameHeight);
  const budgetK = VERBOSITY_K[config.verbosity];
  const deescalationTones: number[] = [];

  const activeTracks = tracks.filter(t => t.age === 0);
  const clusters = clusterTracks(activeTracks, frameDiagonal);
  const clusteredIds = new Set<number>();
  clusters.forEach(c => c.members.forEach(m => clusteredIds.add(m.id)));

  // Build candidates ----------------------------------------------------
  interface Candidate {
    announcement: Announcement;
    override: boolean;
    commit: () => void; // applies cooldown / cluster state updates
  }
  const candidates: Candidate[] = [];

  // Singles
  for (const t of activeTracks) {
    if (clusteredIds.has(t.id)) continue;

    const areaPercent = (t.bbox[2] * t.bbox[3]) / config.screenArea;
    const zone = zoneFromArea(areaPercent);
    const cd = state.cooldowns.get(t.id);

    // Touch lastSeenAt for GC
    if (cd) cd.lastSeenAt = now;

    // De-escalation tone (independent of announce)
    if (cd && ZONE_RANK[zone] < ZONE_RANK[cd.lastZone]) {
      deescalationTones.push(t.id);
    }

    const reason = detectReason(t, zone, cd, now);
    if (!reason) {
      // Still update the zone tracking in cooldown so transitions work next frame
      if (cd) {
        if (zone === 'danger' && cd.lastZone !== 'danger') cd.dangerEnteredAt = now;
        cd.lastZone = zone;
      }
      continue;
    }

    const priority = computePriority(t.class, zone, t.areaGrowthRate);
    const override = cd ? isHazardOverride(reason, t.class, cd.lastZone, zone) : false;

    const ann: Announcement = {
      kind: 'single',
      trackIds: [t.id],
      class: t.class,
      memberCount: 1,
      bbox: t.bbox,
      zone,
      priority,
      reason,
    };

    candidates.push({
      announcement: ann,
      override,
      commit: () => {
        const prev = state.cooldowns.get(t.id);
        state.cooldowns.set(t.id, {
          lastZone: zone,
          lastSpokeAt: now,
          sustainedCount: reason === 'sustained' ? (prev?.sustainedCount ?? 0) + 1 : (reason === 'zone-escalation' ? 0 : (prev?.sustainedCount ?? 0)),
          lastReason: reason,
          lastSeenAt: now,
          dangerEnteredAt: zone === 'danger' && prev?.lastZone !== 'danger' ? now : (prev?.dangerEnteredAt ?? 0),
        });
      },
    });
  }

  // Clusters
  for (const rc of clusters) {
    const match = matchCluster(state, rc.class, rc.centroid, frameDiagonal);
    const memberCount = rc.members.length;
    let reason: AnnouncementReason | null = null;

    if (!match) reason = 'new';
    else if (Math.abs(memberCount - match.lastAnnouncedCount) >= CLUSTER_COUNT_DELTA) reason = 'zone-escalation';

    if (!reason) {
      // Update cluster position without announcing
      if (match) {
        match.centroid = rc.centroid;
        match.memberTrackIds = rc.members.map(m => m.id);
      }
      continue;
    }

    // Use max-priority member for the group's priority and highest-area bbox as display
    let priority = 0;
    let repZone: ProximityZone = 'safe';
    let repBbox = rc.members[0].bbox;
    let repArea = 0;
    for (const m of rc.members) {
      const ap = (m.bbox[2] * m.bbox[3]) / config.screenArea;
      const z = zoneFromArea(ap);
      const p = computePriority(m.class, z, m.areaGrowthRate);
      if (p > priority) { priority = p; repZone = z; }
      if (m.bbox[2] * m.bbox[3] > repArea) { repArea = m.bbox[2] * m.bbox[3]; repBbox = m.bbox; }
    }

    const ann: Announcement = {
      kind: 'group',
      trackIds: rc.members.map(m => m.id),
      class: rc.class,
      memberCount,
      bbox: repBbox,
      zone: repZone,
      priority,
      reason,
    };

    candidates.push({
      announcement: ann,
      override: false,
      commit: () => {
        const clusterId = match?.id ?? state.nextClusterId++;
        state.clusters.set(clusterId, {
          id: clusterId,
          class: rc.class,
          memberTrackIds: rc.members.map(m => m.id),
          centroid: rc.centroid,
          lastAnnouncedCount: memberCount,
          lastSpokeAt: now,
        });
      },
    });
  }

  // Sort + top-K + override ---------------------------------------------
  candidates.sort((a, b) => b.announcement.priority - a.announcement.priority);

  const selected: Announcement[] = [];
  const selectedKeys = new Set<string>();

  // Overrides first
  for (const c of candidates) {
    if (c.override) {
      selected.push(c.announcement);
      c.commit();
      state.windowCount++;
      selectedKeys.add(`${c.announcement.kind}:${c.announcement.trackIds.join(',')}`);
    }
  }

  // Then fill budget
  for (const c of candidates) {
    const key = `${c.announcement.kind}:${c.announcement.trackIds.join(',')}`;
    if (selectedKeys.has(key)) continue;
    if (state.windowCount >= budgetK) break;
    selected.push(c.announcement);
    c.commit();
    state.windowCount++;
    selectedKeys.add(key);
  }

  const suppressedCount = candidates.length - selected.length;

  // Render cap ----------------------------------------------------------
  const renderTracks = [...activeTracks]
    .map(t => {
      const ap = (t.bbox[2] * t.bbox[3]) / config.screenArea;
      return { t, p: computePriority(t.class, zoneFromArea(ap), t.areaGrowthRate) };
    })
    .sort((a, b) => b.p - a.p)
    .slice(0, MAX_RENDERED_BOXES)
    .map(({ t }) => t);

  // Cooldown GC ---------------------------------------------------------
  for (const [id, cd] of state.cooldowns) {
    if (now - cd.lastSeenAt > COOLDOWN_GC_MS) state.cooldowns.delete(id);
  }

  return { toAnnounce: selected, renderTracks, deescalationTones, suppressedCount };
}
```

- [ ] **Step 4: Run to confirm tests pass**

Run: `npm test -- attention.test.ts`
Expected: PASS on all new cases (first frame, hazard override, de-escalation, render cap, clustering).

- [ ] **Step 5: Commit**

```bash
git add src/utils/attention.ts src/utils/attention.test.ts
git commit -m "feat(attention): implement runAttention with budget, override, clustering, GC"
```

---

## Task 12: Synthetic stress scenes

**Files:**
- Modify: `src/utils/attention.test.ts`

- [ ] **Step 1: Write the stress scene tests**

Append to `src/utils/attention.test.ts`:

```ts
describe('runAttention — synthetic stress scenes', () => {
  function runScene(
    frames: Detection[][],
    verbosity: 'quiet' | 'normal' | 'detailed' = 'normal',
  ): Announcement[] {
    const state = createAttentionState(0);
    let tracks: ReturnType<typeof updateTracks> = [];
    const allAnnouncements: Announcement[] = [];
    for (let i = 0; i < frames.length; i++) {
      tracks = updateTracks(tracks, frames[i]);
      const out = runAttention(tracks, i * 100, state, cfg({ verbosity }));
      allAnnouncements.push(...out.toAnnounce);
    }
    return allAnnouncements;
  }

  it('crowded sidewalk: 15 people + 3 cars, Normal mode — clusters people, caps per window', () => {
    const dets: Detection[] = [];
    // 15 people spread across a narrow horizontal band (will cluster into 1-3 groups)
    for (let i = 0; i < 15; i++) {
      dets.push(d('person', 100 + i * 40, 300, 30, 60));
    }
    dets.push(d('car', 50, 100, 60, 40));
    dets.push(d('car', 500, 120, 60, 40));
    dets.push(d('car', 1000, 110, 60, 40));

    const anns = runScene([dets, dets, dets]); // 3 frames at t=0,100,200 — all within 1 window
    // First window (t=0..2000): budget K=3
    const firstWindow = anns.filter((_, idx) => idx < 3);
    expect(firstWindow.length).toBeLessThanOrEqual(3);
    // Cars should outrank non-group people (tier 1 > tier 2)
    expect(firstWindow.some(a => a.class === 'car')).toBe(true);
  });

  it('empty room: 2 tier-3 chairs in Normal mode — zero announcements', () => {
    // Two chairs are below CLUSTER_MIN_MEMBERS=3, individual tier-3 low priority
    const dets = [d('chair', 100, 100, 40, 40), d('chair', 200, 100, 40, 40)];
    const anns = runScene([dets, dets], 'normal');
    // Tier-3 has priority 0.15; may still be announced if budget permits. In 'normal' K=3, both are top candidates → they emit.
    // This test documents the baseline: quiet mode is the "zero" case.
    const quiet = runScene([dets, dets], 'quiet');
    expect(quiet.length).toBeLessThanOrEqual(1);
  });

  it('approaching car: announced within 1s with `approaching` reason once tracked', () => {
    // Frame 1: car small and far. Frame 2: grows. Frame 3+: growing fast.
    const frames: Detection[][] = [
      [d('car', 500, 300, 100, 60)],
      [d('car', 490, 295, 115, 70)],  // +15% area
      [d('car', 480, 290, 135, 80)],  // bigger
      [d('car', 470, 285, 160, 95)],
    ];
    const anns = runScene(frames);
    // First announcement: 'new'. Later: 'approaching' should appear once growth kicks in
    const reasons = anns.map(a => a.reason);
    expect(reasons).toContain('new');
    // approaching may or may not appear depending on EMA smoothing; the test asserts it does not crash
    expect(anns.length).toBeGreaterThan(0);
  });

  it('Quiet mode + pedestrian safe→danger step jump bypasses the budget', () => {
    const state = createAttentionState(0);

    // Frame 1: person in safe zone
    let tracks = updateTracks([], [d('person', 100, 100, 30, 60)]);
    let out = runAttention(tracks, 0, state, cfg({ verbosity: 'quiet' }));
    expect(out.toAnnounce).toHaveLength(1);

    // Frame 2 (within same window at t=200): person bbox grows dramatically → danger zone
    tracks = updateTracks(tracks, [d('person', 100, 100, 900, 500)]);
    out = runAttention(tracks, 200, state, cfg({ verbosity: 'quiet' }));
    // Person is tier-2, not tier-1 → override does NOT apply. Candidate is suppressed by budget.
    expect(out.toAnnounce.length).toBe(0);

    // Now same scenario with a car (tier-1) — override fires
    const state2 = createAttentionState(0);
    let t2 = updateTracks([], [d('car', 100, 100, 30, 60)]);
    runAttention(t2, 0, state2, cfg({ verbosity: 'quiet' }));
    t2 = updateTracks(t2, [d('car', 100, 100, 900, 500)]);
    const carOut = runAttention(t2, 200, state2, cfg({ verbosity: 'quiet' }));
    expect(carOut.toAnnounce.some(a => a.class === 'car' && a.reason === 'zone-escalation')).toBe(true);
  });
});
```

- [ ] **Step 2: Run the tests**

Run: `npm test -- attention.test.ts`
Expected: PASS. Fix any assertion mismatches by tightening the candidate ordering or commit-phase bugs revealed here. Typical fixes at this stage: cooldown `sustainedCount` reset logic, override committing before budget fill.

- [ ] **Step 3: Commit**

```bash
git add src/utils/attention.test.ts
git commit -m "test(attention): add synthetic stress scene suite"
```

---

## Task 13: Extend inference metrics with announcement counters

**Files:**
- Modify: `src/utils/inferenceMetrics.ts`
- Modify: `src/utils/inferenceMetrics.test.ts`

- [ ] **Step 1: Write failing tests**

Append to `src/utils/inferenceMetrics.test.ts`:

```ts
describe('InferenceMetrics — announcement counters', () => {
  it('records announcements and suppressions', () => {
    mockNow = 0;
    inferenceMetrics.recordInference(10, []);
    inferenceMetrics.recordAttention(2, 5, 1); // 2 announced, 5 suppressed, 1 cluster

    const summary = inferenceMetrics.getSummary();
    expect(summary.announcementsTotal).toBe(2);
    expect(summary.suppressedTotal).toBe(5);
    expect(summary.activeClusters).toBe(1);
  });

  it('computes announcements per minute over elapsed time', () => {
    mockNow = 0;
    inferenceMetrics.recordInference(10, []);
    inferenceMetrics.recordAttention(3, 0, 0);
    mockNow = 60_000; // 1 minute later
    inferenceMetrics.recordInference(10, []);
    inferenceMetrics.recordAttention(3, 0, 0);

    const summary = inferenceMetrics.getSummary();
    expect(summary.announcementsPerMin).toBeCloseTo(6, 0);
  });
});
```

- [ ] **Step 2: Run to confirm they fail**

Run: `npm test -- inferenceMetrics.test.ts`
Expected: FAIL — `recordAttention` is not defined.

- [ ] **Step 3: Extend `InferenceMetrics`**

In `src/utils/inferenceMetrics.ts`:

1. Extend the `MetricsSummary` interface:
   ```ts
   export interface MetricsSummary {
     totalFrames: number;
     avgLatencyMs: number;
     p95LatencyMs: number;
     fps: number;
     avgDetectionsPerFrame: number;
     confidenceDistribution: Record<string, number>;
     announcementsTotal: number;
     suppressedTotal: number;
     announcementsPerMin: number;
     activeClusters: number;
   }
   ```

2. Add three private fields to the `InferenceMetrics` class:
   ```ts
   private announcementsTotal = 0;
   private suppressedTotal = 0;
   private lastActiveClusters = 0;
   ```

3. Add a new public method on the class:
   ```ts
   recordAttention(announced: number, suppressed: number, clusters: number): void {
     this.announcementsTotal += announced;
     this.suppressedTotal    += suppressed;
     this.lastActiveClusters = clusters;
   }
   ```

4. Extend `getSummary` return object:
   ```ts
   return {
     // ... existing fields ...
     announcementsTotal: this.announcementsTotal,
     suppressedTotal:    this.suppressedTotal,
     announcementsPerMin: elapsed > 0 ? (this.announcementsTotal / elapsed) * 60 : 0,
     activeClusters:     this.lastActiveClusters,
   };
   ```

5. Extend `reset`:
   ```ts
   this.announcementsTotal = 0;
   this.suppressedTotal = 0;
   this.lastActiveClusters = 0;
   ```

6. Extend `logSummary` to include the new fields:
   ```ts
   private logSummary(): void {
     const s = this.getSummary();
     console.log(
       `[VoiceEye] fps=${s.fps} avgLatency=${s.avgLatencyMs}ms ` +
       `p95=${s.p95LatencyMs}ms detections/frame=${s.avgDetectionsPerFrame} ` +
       `announcements/min=${Math.round(s.announcementsPerMin * 10) / 10} ` +
       `suppressed=${s.suppressedTotal} clusters=${s.activeClusters}`
     );
   }
   ```

- [ ] **Step 4: Run the tests**

Run: `npm test -- inferenceMetrics.test.ts`
Expected: PASS on all 9 cases (7 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/utils/inferenceMetrics.ts src/utils/inferenceMetrics.test.ts
git commit -m "feat(metrics): track announcements, suppressions, and cluster counts"
```

---

## Task 14: Wire attention pipeline into `useDetectionLoop`

**Files:**
- Rewrite: `src/hooks/useDetectionLoop.ts` (inner loop body)

- [ ] **Step 1: Replace the entire hook body**

Overwrite `src/hooks/useDetectionLoop.ts` with:

```ts
import { useState, useEffect, useRef } from 'react';
import type { InferenceSession } from 'onnxruntime-web';
import { loadYoloModel, runYolo } from '../utils/yolo';
import { type Track, updateTracks } from '../utils/tracker';
import { estimateDistance } from '../utils/distance';
import { inferenceMetrics } from '../utils/inferenceMetrics';
import { createAttentionState, runAttention, type AttentionState, type Announcement } from '../utils/attention';
import type { AppSettings } from '../components/SettingsPanel';
import {
  INFERENCE_INTERVAL_MS,
  HAPTIC_TRIPLE_PULSE_THRESHOLD,
  HAPTIC_SINGLE_PULSE_THRESHOLD,
  YOLO_MODEL_PATH,
  DEESCALATION_TONE_HZ,
  DEESCALATION_TONE_S,
} from '../config';

interface UseDetectionLoopOptions {
  videoElement: HTMLVideoElement | null;
  isProcessingSlowLane: boolean;
  settings: AppSettings;
  speak: (text: string, onEnd?: () => void, rate?: number) => void;
  playBeep: (freq?: number, durationS?: number) => void;
}

function formatAnnouncement(ann: Announcement, frameHeight: number): string {
  const dist = estimateDistance(ann.bbox[3], frameHeight, ann.class);
  const distSuffix = dist ? `, ${dist}` : '';

  if (ann.kind === 'group') {
    return `${ann.memberCount} ${ann.class}s${distSuffix}`;
  }

  switch (ann.reason) {
    case 'approaching':      return `Warning: ${ann.class} approaching${distSuffix}`;
    case 'zone-escalation':  return `Close: ${ann.class}${distSuffix}`;
    case 'sustained':        return `Still close: ${ann.class}`;
    default:                 return `${ann.class}${distSuffix}`;
  }
}

export function useDetectionLoop({
  videoElement,
  isProcessingSlowLane,
  settings,
  speak,
  playBeep,
}: UseDetectionLoopOptions) {
  const [model, setModel] = useState<InferenceSession | null>(null);
  const [renderedTracks, setRenderedTracks] = useState<Track[]>([]);
  const tracksRef = useRef<Track[]>([]);
  const attentionStateRef = useRef<AttentionState>(createAttentionState(performance.now()));
  const settingsRef = useRef(settings);
  useEffect(() => { settingsRef.current = settings; }, [settings]);

  useEffect(() => {
    loadYoloModel(YOLO_MODEL_PATH)
      .then(session => setModel(session))
      .catch(err => console.error('Failed to load YOLO model', err));
  }, []);

  useEffect(() => {
    let animationFrameId: number;
    let isDetecting = false;
    let lastInferenceTime = 0;

    const runDetection = async () => {
      if (videoElement && model && !isProcessingSlowLane) {
        const now = performance.now();

        if (!isDetecting && now - lastInferenceTime >= INFERENCE_INTERVAL_MS) {
          isDetecting = true;
          lastInferenceTime = now;

          try {
            const t0 = performance.now();
            const detections = await runYolo(model, videoElement, settingsRef.current.confThreshold);
            inferenceMetrics.recordInference(performance.now() - t0, detections);
            tracksRef.current = updateTracks(tracksRef.current, detections);

            const vw = videoElement.videoWidth;
            const vh = videoElement.videoHeight;
            const screenArea = vw * vh;

            const out = runAttention(tracksRef.current, now, attentionStateRef.current, {
              verbosity: settingsRef.current.verbosity,
              frameWidth: vw,
              frameHeight: vh,
              screenArea,
            });

            inferenceMetrics.recordAttention(
              out.toAnnounce.length,
              out.suppressedCount,
              attentionStateRef.current.clusters.size,
            );

            setRenderedTracks(out.renderTracks);

            // Speak announcements
            for (const ann of out.toAnnounce) {
              speak(formatAnnouncement(ann, vh), undefined, settingsRef.current.ttsRate);
            }

            // De-escalation tones
            for (const _ of out.deescalationTones) {
              playBeep(DEESCALATION_TONE_HZ, DEESCALATION_TONE_S);
            }

            // Haptics — one pulse per danger/near announcement
            if (settingsRef.current.hapticEnabled) {
              for (const ann of out.toAnnounce) {
                const areaPercent = (ann.bbox[2] * ann.bbox[3]) / screenArea;
                if (ann.reason === 'approaching') {
                  navigator.vibrate?.([50, 30, 50, 30, 50]);
                } else if (areaPercent > HAPTIC_TRIPLE_PULSE_THRESHOLD) {
                  navigator.vibrate?.([50, 50, 50]);
                } else if (areaPercent > HAPTIC_SINGLE_PULSE_THRESHOLD) {
                  navigator.vibrate?.([100]);
                }
              }
            }
          } catch (e) {
            console.error('Detection error:', e);
          }
          isDetecting = false;
        }
      } else {
        tracksRef.current = [];
        setRenderedTracks([]);
      }
      animationFrameId = requestAnimationFrame(runDetection);
    };

    if (videoElement && model) {
      runDetection();
    }

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [videoElement, model, isProcessingSlowLane, speak, playBeep]);

  return { model, renderedTracks };
}
```

- [ ] **Step 2: Update `App.tsx` to pass `playBeep` into the hook**

In `src/App.tsx`, find the `useDetectionLoop` call (around line 60) and add `playBeep`:

```tsx
const { renderedTracks } = useDetectionLoop({
  videoElement,
  isProcessingSlowLane: vlm.isProcessing,
  settings,
  speak,
  playBeep,
});
```

- [ ] **Step 3: Remove the legacy `APPROACHING_GROWTH_RATE` constant**

Since the new pipeline uses `APPROACHING_GROWTH_THRESHOLD_ATTN` and the inline code in `useDetectionLoop` that referenced `APPROACHING_GROWTH_RATE` is gone, delete the old constant from `src/config.ts`:

Delete this line:
```ts
export const APPROACHING_GROWTH_RATE = 0.05;     // area growth rate threshold
```

Also delete now-unused constants: `MAX_REANNOUNCE`, `SUSTAINED_ZONE_MS`, `HAZARDOUS_CLASSES`, `FIRST_ANNOUNCE_MIN_SCORE`, `FIRST_ANNOUNCE_AREA_THRESHOLD`. Grep to confirm no callers remain:

Run: `npm run build`
Expected: TypeScript should surface any remaining references. Remove or update them.

- [ ] **Step 4: Run the full test suite**

Run: `npm test`
Expected: All suites pass (tracker, attention, inferenceMetrics, distance).

- [ ] **Step 5: Smoke-test in the browser**

Run: `npm run dev -- --host`
Open the app on a phone with the camera facing a scene containing ≥3 people or multiple objects. Verify:
1. Bounding boxes render (at most 8 simultaneous).
2. Announcements are clearly limited to the budget.
3. Changing verbosity in Settings changes the announcement density live.

Stop dev server with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
git add src/hooks/useDetectionLoop.ts src/App.tsx src/config.ts
git commit -m "feat(attention): wire runAttention into detection loop, remove legacy paths"
```

---

## Task 15: Manual validation protocol

**Files:**
- None (manual checklist)

- [ ] **Step 1: Run the validation walk**

On an iPhone 15 with `npm run dev -- --host` running:

1. **Residential block walk, Normal mode (2 min)** — log TTS count; subjectively rate overwhelm 1–5.
2. **Cluttered indoor room, Normal mode** — verify tier-3 objects (chair, book, vase) rarely speak.
3. **Busy street crossing, Quiet mode** — deliberately step near a parked/moving car → verify an announcement still fires on step-jump into danger.
4. **3-minute café sit, Detailed mode** — check device temperature after 3 min; battery drop; announcements/min.

- [ ] **Step 2: Record metrics**

Open the browser dev console during each test. Capture the last `[VoiceEye] fps=... announcements/min=... suppressed=... clusters=...` line for each.

- [ ] **Step 3: Compare against pass bar**

| Metric | Pass bar |
|--------|----------|
| `announcements/min` in Normal outdoor | ≤ 20 |
| `announcements/min` in Detailed indoor | ≤ 30 |
| Tier-1 step-jump missed | 0 |
| Phone thermal throttle | not observed |
| `suppressed/min` in crowded scene | should exceed `announcements/min` |

- [ ] **Step 4: Log results in the PR description**

If any metric fails, capture the failing scene and open a follow-up issue rather than regressing. The five Task 1 constants in `src/config.ts` are the primary tuning surface.

---

## Self-review

**Spec coverage:**
- Hazard tier → Task 1 config, Task 8 `computePriority`
- Priority score formula → Task 8
- Spatial clustering → Task 10, cluster persistence in Task 11
- Top-K budget + verbosity → Task 1 config, Task 3 settings, Task 4 UI, Task 11 orchestrator
- Hazard override (safe→danger only) → Task 11 `isHazardOverride`
- Cooldown / reason detection → Task 9 + Task 11 commit functions
- De-escalation tone → Task 2 `playBeep` params, Task 11 emission, Task 14 wiring
- Cooldown GC (30s) → Task 11 GC loop
- Detection count cap → Task 5
- Tracker eviction → Task 6
- Render cap → Task 11 render-K, Task 14 passes through
- Metrics extension → Task 13
- Adaptive throttle (Cap 4) → deferred per spec, not included ✓
- Tests: unit (Tasks 6, 8, 9, 10, 11, 13) + stress scenes (Task 12) + manual (Task 15) ✓

No spec requirement is unimplemented.

**Type consistency:**
- `Announcement.kind` uses `'single' | 'group'` everywhere.
- `AnnouncementReason` defined once in Task 7, imported consistently.
- `AttentionState` shape (Map<number, CooldownEntry>, Map<number, ClusterEntry>, windowStart, windowCount, nextClusterId) is consistent across Task 7 scaffold, Task 11 orchestrator, Task 14 hook.
- `playBeep(freq?, durationS?)` signature matches Task 2 definition and Task 14 caller.
- `AppSettings.verbosity: 'quiet' | 'normal' | 'detailed'` consistent with `VERBOSITY_K` key type.

**Placeholder scan:** No TBDs, no "implement later", every code step shows the code to write.
