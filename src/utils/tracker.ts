import type { Detection } from './yolo';
import {
  TRACKER_MAX_AGE,
  TRACKER_MIN_IOU,
  TRACKER_EMA_ALPHA,
  PROXIMITY_DANGER_THRESHOLD,
  PROXIMITY_NEAR_THRESHOLD,
} from '../config';

export type ProximityZone = 'safe' | 'near' | 'danger';

export interface Track {
  id: number;
  bbox: [number, number, number, number]; // [x, y, w, h]
  class: string;
  score: number;
  age: number;            // frames since last matched detection (0 = active this frame)
  announced: boolean;     // true once initial TTS has fired for this track

  // Velocity / motion
  vx: number;             // centroid x velocity (pixels/frame), EMA smoothed
  vy: number;             // centroid y velocity (pixels/frame), EMA smoothed
  areaGrowthRate: number; // normalised d(area)/dt — positive = approaching
  lastCentroid: [number, number];
  lastArea: number;

  // Proximity re-announcement
  proximityZone: ProximityZone;
  zoneEntryTime: number;  // timestamp (ms) when current zone was entered
  reannounceCount: number; // capped to prevent spam
}

let nextId = 1;

function iou(
  a: [number, number, number, number],
  b: [number, number, number, number]
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

/**
 * Greedy IoU-based tracker with velocity estimation.
 * Matches detections to existing tracks by class + IoU overlap.
 * Computes per-track centroid velocity and area growth rate via EMA.
 */
export function updateTracks(tracks: Track[], detections: Detection[]): Track[] {
  const usedDetections = new Set<number>();
  const updated: Track[] = [];

  for (const track of tracks) {
    let bestIou = TRACKER_MIN_IOU;
    let bestDet = -1;

    for (let d = 0; d < detections.length; d++) {
      if (usedDetections.has(d)) continue;
      if (detections[d].class !== track.class) continue;
      const overlap = iou(track.bbox, detections[d].bbox);
      if (overlap > bestIou) { bestIou = overlap; bestDet = d; }
    }

    if (bestDet >= 0) {
      usedDetections.add(bestDet);
      const det = detections[bestDet];

      // Compute new centroid and area
      const cx = det.bbox[0] + det.bbox[2] / 2;
      const cy = det.bbox[1] + det.bbox[3] / 2;
      const area = det.bbox[2] * det.bbox[3];

      // EMA-smoothed velocity
      const vx = TRACKER_EMA_ALPHA * (cx - track.lastCentroid[0]) + (1 - TRACKER_EMA_ALPHA) * track.vx;
      const vy = TRACKER_EMA_ALPHA * (cy - track.lastCentroid[1]) + (1 - TRACKER_EMA_ALPHA) * track.vy;
      const areaGrowthRate = track.lastArea > 0
        ? TRACKER_EMA_ALPHA * ((area - track.lastArea) / track.lastArea) + (1 - TRACKER_EMA_ALPHA) * track.areaGrowthRate
        : 0;

      updated.push({
        ...track,
        bbox: det.bbox,
        score: det.score,
        age: 0,
        vx,
        vy,
        areaGrowthRate,
        lastCentroid: [cx, cy],
        lastArea: area,
      });
    } else {
      const aged = track.age + 1;
      if (aged <= TRACKER_MAX_AGE) {
        updated.push({ ...track, age: aged });
      }
    }
  }

  // Spawn new tracks for unmatched detections
  for (let d = 0; d < detections.length; d++) {
    if (!usedDetections.has(d)) {
      const det = detections[d];
      const cx = det.bbox[0] + det.bbox[2] / 2;
      const cy = det.bbox[1] + det.bbox[3] / 2;
      updated.push({
        id: nextId++,
        bbox: det.bbox,
        class: det.class,
        score: det.score,
        age: 0,
        announced: false,
        vx: 0,
        vy: 0,
        areaGrowthRate: 0,
        lastCentroid: [cx, cy],
        lastArea: det.bbox[2] * det.bbox[3],
        proximityZone: 'safe',
        zoneEntryTime: Date.now(),
        reannounceCount: 0,
      });
    }
  }

  return updated;
}
