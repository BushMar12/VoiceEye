import type { Detection } from './yolo';
import {
  TRACKER_MAX_AGE_MS,
  TRACKER_EMA_ALPHA,
  TRACKER_HIGH_CONF_THRESHOLD,
  TRACKER_COST_THRESHOLD_HIGH,
  TRACKER_COST_THRESHOLD_LOW,
  TRACKER_CLASS_HISTORY_LEN,
  TRACKER_CONFIRM_HITS,
  PROXIMITY_DANGER_THRESHOLD,
  PROXIMITY_NEAR_THRESHOLD,
  MAX_TRACKS,
  HAZARD_TIER,
  DEFAULT_TIER,
  TIER_WEIGHT,
  ZONE_WEIGHT,
  EVICTION_AGE_PENALTY,
  INFERENCE_INTERVAL_MS,
} from '../config';

export type ProximityZone = 'safe' | 'near' | 'danger';

export interface Track {
  id: number;
  bbox: [number, number, number, number]; // [x, y, w, h]
  class: string;
  /** Sliding window of raw class labels — mode is the stable class name. */
  classHistory: string[];
  score: number;
  /** Tentative until TRACKER_CONFIRM_HITS consecutive matches; never reaches attention/render. */
  status: 'tentative' | 'confirmed';
  /** Consecutive matched-frame count since spawn. */
  hits: number;
  /** Ms since last matched detection (0 = matched this frame). Replaces frame-count age. */
  ageMs: number;

  // EMA-smoothed velocity in px/ms
  vx: number;
  vy: number;
  vw: number; // width change velocity
  vh: number; // height change velocity

  /** Normalised d(area)/dt — positive = approaching camera. */
  areaGrowthRate: number;
  lastCentroid: [number, number];
  lastArea: number;

  /** Recomputed every frame from bbox / screenArea. Drives bbox colour and eviction weight. */
  proximityZone: ProximityZone;
}

let nextId = 1;

// ── Geometry helpers ──────────────────────────────────────────────────────────

function iou(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  const bx2 = b[0] + b[2], by2 = b[1] + b[3];
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(ax2, bx2),   iy2 = Math.min(ay2, by2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

export function classifyProximity(areaPercent: number): ProximityZone {
  if (areaPercent > PROXIMITY_DANGER_THRESHOLD) return 'danger';
  if (areaPercent > PROXIMITY_NEAR_THRESHOLD)   return 'near';
  return 'safe';
}

/** Linearly predict the bbox forward by dtMs using stored px/ms velocity. */
function predict(t: Track, dtMs: number): [number, number, number, number] {
  return [
    t.bbox[0] + t.vx * dtMs,
    t.bbox[1] + t.vy * dtMs,
    Math.max(1, t.bbox[2] + t.vw * dtMs),
    Math.max(1, t.bbox[3] + t.vh * dtMs),
  ];
}

/**
 * Assignment cost between a predicted track bbox and a candidate detection.
 *   0.70 × (1 − IoU)           — spatial overlap
 *   0.30 × (centreGap / diag)  — normalised centre distance
 *   0.15 class mismatch bonus  — soft class penalty; doesn't hard-reject
 *
 * Maximum possible cost: 1.15 (0 IoU, max distance, wrong class).
 * A wrong-class detection still matches if it is at essentially the same
 * position — intentional, since a single YOLO mis-label should not kill the
 * track. The class-history window stabilises the output label.
 */
function matchCost(
  pred: [number, number, number, number],
  det: Detection,
  trackClass: string,
  diagPx: number,
): number {
  const overlap  = 1 - iou(pred, det.bbox);
  const pcx = pred[0] + pred[2] / 2, pcy = pred[1] + pred[3] / 2;
  const dcx = det.bbox[0] + det.bbox[2] / 2, dcy = det.bbox[1] + det.bbox[3] / 2;
  const dist     = diagPx > 0 ? Math.hypot(pcx - dcx, pcy - dcy) / diagPx : 0;
  const classPen = trackClass !== det.class ? 0.15 : 0;
  return 0.7 * overlap + 0.3 * dist + classPen;
}

// ── Hungarian (Jonker-Volgenant shortest-path variant) ────────────────────────
//
// Solves the rectangular linear-assignment problem for cost[i][j].
// Returns assignment[i] = j for each row i, or −1 if unassigned or if
// cost[i][j] > maxCost.  O(n² · m) — negligible for n, m ≤ 30.

function hungarian(cost: number[][], maxCost: number): number[] {
  const n = cost.length;
  if (n === 0) return [];
  const m = cost[0].length;
  if (m === 0) return new Array(n).fill(-1);

  const size = Math.max(n, m);
  const c = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => (i < n && j < m) ? cost[i][j] : 0),
  );

  const u   = new Array<number>(size + 1).fill(0); // row potentials
  const v   = new Array<number>(size + 1).fill(0); // col potentials
  const p   = new Array<number>(size + 1).fill(0); // p[j] = row assigned to col j (1-indexed)
  const way = new Array<number>(size + 1).fill(0);

  for (let i = 1; i <= size; i++) {
    p[0] = i;
    let j0 = 0;
    const minDist = new Array<number>(size + 1).fill(Infinity);
    const used    = new Array<boolean>(size + 1).fill(false);

    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = -1;

      for (let j = 1; j <= size; j++) {
        if (!used[j]) {
          const cur = c[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minDist[j]) { minDist[j] = cur; way[j] = j0; }
          if (minDist[j] < delta) { delta = minDist[j]; j1 = j; }
        }
      }

      for (let j = 0; j <= size; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else         { minDist[j] -= delta; }
      }

      j0 = j1 as number;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const assignment = new Array<number>(n).fill(-1);
  for (let j = 1; j <= m; j++) {
    const row = p[j] - 1;
    if (row >= 0 && row < n && cost[row][j - 1] <= maxCost) {
      assignment[row] = j - 1;
    }
  }
  return assignment;
}

// ── Class voting ─────────────────────────────────────────────────────────────

/**
 * Returns the most-frequent class label in the history window.
 * Ties go to the most recent observation (iterate in reverse so newer labels
 * have priority when two counts are equal).
 */
function classMode(history: string[]): string {
  const freq = new Map<string, number>();
  for (const cls of history) freq.set(cls, (freq.get(cls) ?? 0) + 1);
  let best = history[history.length - 1];
  let max  = 0;
  for (let i = history.length - 1; i >= 0; i--) {
    const count = freq.get(history[i])!;
    if (count > max) { max = count; best = history[i]; }
  }
  return best;
}

// ── Eviction priority ─────────────────────────────────────────────────────────

function trackPriority(t: Track): number {
  const tier  = HAZARD_TIER[t.class] ?? DEFAULT_TIER;
  const tierW = TIER_WEIGHT[tier];
  const zoneW = ZONE_WEIGHT[t.proximityZone];
  // Normalise ageMs to frame-equivalent so the penalty magnitude matches the old formula.
  const agePenalty = (t.ageMs / INFERENCE_INTERVAL_MS) * EVICTION_AGE_PENALTY;
  return tierW * zoneW - agePenalty;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Predict-then-associate multi-object tracker.
 *
 * Improvements over the previous greedy IoU tracker:
 *   1. Velocity-based bbox prediction (px/ms EMA) before matching.
 *   2. Two-stage Hungarian assignment — high-conf dets first, unmatched
 *      tracks then compete for low-conf dets (occlusion recovery).
 *   3. Class voting over a sliding history window — single mis-label frames
 *      no longer kill or swap a track.
 *   4. Tentative → confirmed lifecycle — one-frame false positives never
 *      reach the attention pipeline.
 *   5. Time-based aging in ms — fps-invariant track lifetime.
 *
 * `dtMs` is the elapsed time since the previous call (defaults to
 * INFERENCE_INTERVAL_MS so existing callers that omit it are unaffected).
 *
 * Only `confirmed` tracks with `ageMs === 0` are active this frame.
 * The attention pipeline and renderer should filter on those two fields.
 */
export function updateTracks(
  tracks: Track[],
  detections: Detection[],
  screenArea = 0,
  dtMs = INFERENCE_INTERVAL_MS,
): Track[] {
  const zoneOf = (area: number): ProximityZone =>
    screenArea > 0 ? classifyProximity(area / screenArea) : 'safe';

  // Approximate frame diagonal for normalising centre-distance cost.
  // Use Infinity when screenArea is unknown so the centre-distance term contributes 0
  // and matching degrades gracefully to IoU-only (avoids blowing up the cost).
  const diagPx = screenArea > 0 ? Math.sqrt(2 * screenArea) : Infinity;

  // Predict every track's bbox forward by dtMs.
  const predicted = tracks.map(t => predict(t, dtMs));

  // Partition detections into high- and low-confidence sets.
  const highIdx: number[] = [];
  const lowIdx:  number[] = [];
  for (let d = 0; d < detections.length; d++) {
    (detections[d].score >= TRACKER_HIGH_CONF_THRESHOLD ? highIdx : lowIdx).push(d);
  }

  /**
   * Build a Hungarian assignment from a subset of tracks to a subset of
   * detections. Returns Map<trackIndex, detectionIndex>.
   */
  function buildAssignment(
    trackIndices: number[],
    detIndices: number[],
    threshold: number,
  ): Map<number, number> {
    if (trackIndices.length === 0 || detIndices.length === 0) return new Map();
    const cost = trackIndices.map(ti =>
      detIndices.map(di => matchCost(predicted[ti], detections[di], tracks[ti].class, diagPx)),
    );
    const assignment = hungarian(cost, threshold);
    const result = new Map<number, number>();
    for (let i = 0; i < assignment.length; i++) {
      if (assignment[i] >= 0) result.set(trackIndices[i], detIndices[assignment[i]]);
    }
    return result;
  }

  // Stage A — all tracks vs high-confidence detections.
  const allTrackIndices = tracks.map((_, i) => i);
  const stageA = buildAssignment(allTrackIndices, highIdx, TRACKER_COST_THRESHOLD_HIGH);

  // Stage B — still-unmatched tracks vs low-confidence detections.
  // Stricter cost threshold reduces false re-links.
  const unmatchedTracks = allTrackIndices.filter(i => !stageA.has(i));
  const stageB = buildAssignment(unmatchedTracks, lowIdx, TRACKER_COST_THRESHOLD_LOW);

  const matchedDets = new Set<number>();
  for (const di of stageA.values()) matchedDets.add(di);
  for (const di of stageB.values()) matchedDets.add(di);

  const combined = new Map([...stageA, ...stageB]);

  // Apply updates.
  const updated: Track[] = [];

  for (let i = 0; i < tracks.length; i++) {
    const track  = tracks[i];
    const detIdx = combined.get(i);

    if (detIdx !== undefined) {
      // Matched — update kinematics, class history, and lifecycle state.
      const det  = detections[detIdx];
      const cx   = det.bbox[0] + det.bbox[2] / 2;
      const cy   = det.bbox[1] + det.bbox[3] / 2;
      const area = det.bbox[2] * det.bbox[3];

      const rawVx = dtMs > 0 ? (cx - track.lastCentroid[0]) / dtMs : 0;
      const rawVy = dtMs > 0 ? (cy - track.lastCentroid[1]) / dtMs : 0;
      const rawVw = dtMs > 0 ? (det.bbox[2] - track.bbox[2]) / dtMs : 0;
      const rawVh = dtMs > 0 ? (det.bbox[3] - track.bbox[3]) / dtMs : 0;

      const α  = TRACKER_EMA_ALPHA;
      const vx = α * rawVx + (1 - α) * track.vx;
      const vy = α * rawVy + (1 - α) * track.vy;
      const vw = α * rawVw + (1 - α) * track.vw;
      const vh = α * rawVh + (1 - α) * track.vh;

      const areaGrowthRate = track.lastArea > 0
        ? α * ((area - track.lastArea) / track.lastArea) + (1 - α) * track.areaGrowthRate
        : 0;

      const newHistory  = [...track.classHistory, det.class].slice(-TRACKER_CLASS_HISTORY_LEN);
      const stableClass = classMode(newHistory);
      const newHits     = track.hits + 1;

      updated.push({
        ...track,
        bbox:         det.bbox,
        class:        stableClass,
        classHistory: newHistory,
        score:        det.score,
        status:       newHits >= TRACKER_CONFIRM_HITS ? 'confirmed' : track.status,
        hits:         newHits,
        ageMs:        0,
        vx, vy, vw, vh,
        areaGrowthRate,
        lastCentroid: [cx, cy],
        lastArea:     area,
        proximityZone: zoneOf(area),
      });
    } else {
      // Unmatched — age and drop when the limit is exceeded.
      const newAgeMs = track.ageMs + dtMs;
      if (newAgeMs <= TRACKER_MAX_AGE_MS) {
        updated.push({ ...track, ageMs: newAgeMs });
      }
    }
  }

  // Spawn new tracks only from unmatched high-confidence detections.
  // Low-confidence unmatched detections are too noisy to seed a new track.
  for (const di of highIdx) {
    if (!matchedDets.has(di)) {
      const det  = detections[di];
      const cx   = det.bbox[0] + det.bbox[2] / 2;
      const cy   = det.bbox[1] + det.bbox[3] / 2;
      const area = det.bbox[2] * det.bbox[3];
      updated.push({
        id:           nextId++,
        bbox:         det.bbox,
        class:        det.class,
        classHistory: [det.class],
        score:        det.score,
        status:       'tentative',
        hits:         1,
        ageMs:        0,
        vx: 0, vy: 0, vw: 0, vh: 0,
        areaGrowthRate: 0,
        lastCentroid: [cx, cy],
        lastArea:     area,
        proximityZone: zoneOf(area),
      });
    }
  }

  // Evict lowest-priority tracks if the pool exceeds the cap.
  if (updated.length > MAX_TRACKS) {
    updated.sort((a, b) => trackPriority(b) - trackPriority(a));
    updated.length = MAX_TRACKS;
  }

  return updated;
}
