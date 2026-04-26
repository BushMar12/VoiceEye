import { describe, it, expect } from 'vitest';
import { updateTracks, classifyProximity } from './tracker';
import { MAX_TRACKS, INFERENCE_INTERVAL_MS } from '../config';
import type { Detection } from './yolo';

// Helper — score defaults to 0.9 (high-conf) so tracks spawn and match normally.
function det(cls: string, x: number, y: number, w: number, h: number, score = 0.9): Detection {
  return { bbox: [x, y, w, h], class: cls, score };
}

// ── classifyProximity ─────────────────────────────────────────────────────────

describe('classifyProximity', () => {
  it('returns "danger" for area > 0.5', () => {
    expect(classifyProximity(0.6)).toBe('danger');
    expect(classifyProximity(0.51)).toBe('danger');
    expect(classifyProximity(1.0)).toBe('danger');
  });

  it('returns "near" for area between 0.25 and 0.5', () => {
    expect(classifyProximity(0.3)).toBe('near');
    expect(classifyProximity(0.5)).toBe('near');
    expect(classifyProximity(0.26)).toBe('near');
  });

  it('returns "safe" for area <= 0.25', () => {
    expect(classifyProximity(0.1)).toBe('safe');
    expect(classifyProximity(0.25)).toBe('safe');
    expect(classifyProximity(0.0)).toBe('safe');
  });
});

// ── Basic spawning and properties ─────────────────────────────────────────────

describe('updateTracks — spawning', () => {
  it('creates a tentative track for the first unmatched high-conf detection', () => {
    const tracks = updateTracks([], [det('person', 10, 10, 50, 100)]);
    expect(tracks).toHaveLength(1);
    expect(tracks[0].class).toBe('person');
    expect(tracks[0].ageMs).toBe(0);
    expect(tracks[0].status).toBe('tentative');
    expect(tracks[0].hits).toBe(1);
    expect(tracks[0].proximityZone).toBe('safe');
  });

  it('promotes to confirmed after TRACKER_CONFIRM_HITS consecutive matches', () => {
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)]);
    expect(t1[0].status).toBe('tentative');
    // Second consecutive match → confirmed (TRACKER_CONFIRM_HITS = 2)
    const t2 = updateTracks(t1, [det('person', 102, 101, 50, 100)]);
    expect(t2[0].id).toBe(t1[0].id);
    expect(t2[0].status).toBe('confirmed');
    expect(t2[0].hits).toBe(2);
  });

  it('ignores low-confidence detections for spawning new tracks', () => {
    const lowConf = det('person', 10, 10, 50, 100, 0.3);
    const tracks = updateTracks([], [lowConf]);
    expect(tracks).toHaveLength(0);
  });

  it('classifies proximityZone from screenArea per frame', () => {
    const screenArea = 800 * 600; // 480 000

    // 600×600 = 360 000 → 75% → danger
    const danger = updateTracks([], [det('car', 0, 0, 600, 600)], screenArea);
    expect(danger[0].proximityZone).toBe('danger');

    // 400×350 = 140 000 → 29% → near
    const near = updateTracks([], [det('car', 0, 0, 400, 350)], screenArea);
    expect(near[0].proximityZone).toBe('near');

    // 100×100 = 10 000 → 2% → safe
    const safe = updateTracks([], [det('car', 0, 0, 100, 100)], screenArea);
    expect(safe[0].proximityZone).toBe('safe');
  });

  it('assigns unique IDs to new tracks', () => {
    const tracks = updateTracks([], [
      det('person', 10,  10, 50, 100),
      det('car',   200, 200, 80,  60),
    ]);
    expect(tracks).toHaveLength(2);
    expect(tracks[0].id).not.toBe(tracks[1].id);
  });

  it('computes centroid and area correctly for new tracks', () => {
    const tracks = updateTracks([], [det('person', 100, 200, 50, 80)]);
    expect(tracks[0].lastCentroid).toEqual([125, 240]); // x + w/2, y + h/2
    expect(tracks[0].lastArea).toBe(4000); // 50 * 80
  });
});

// ── Matching ──────────────────────────────────────────────────────────────────

describe('updateTracks — matching', () => {
  it('matches overlapping same-class detections to existing tracks', () => {
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)]);
    const id = t1[0].id;

    const t2 = updateTracks(t1, [det('person', 105, 102, 50, 100)]);
    expect(t2).toHaveLength(1);
    expect(t2[0].id).toBe(id);
    expect(t2[0].ageMs).toBe(0);
  });

  it('updates proximityZone on a matched track when bbox grows', () => {
    const screenArea = 800 * 600;
    const t1 = updateTracks([], [det('car', 100, 100, 100, 100)], screenArea);
    expect(t1[0].proximityZone).toBe('safe');

    // (200×200)/(800×600) ≈ 8% → still safe, but same track ID
    const t2 = updateTracks(t1, [det('car', 100, 100, 200, 200)], screenArea);
    expect(t2[0].id).toBe(t1[0].id);
    expect(t2[0].proximityZone).toBe('safe');
  });

  it('allows cross-class matching at high IoU and stabilises class via history', () => {
    // A single YOLO mis-label on the same bbox should not kill the track.
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)]);
    // Exact same bbox, different class — IoU=1.0, cost = class penalty only (0.15) < threshold
    const t2 = updateTracks(t1, [det('car', 100, 100, 50, 100)]);
    expect(t2).toHaveLength(1);
    expect(t2[0].id).toBe(t1[0].id);
    expect(t2[0].classHistory).toEqual(['person', 'car']);
    // Mode is a tie → most-recent wins → 'car'
    expect(t2[0].class).toBe('car');
  });

  it('does not match detections that are spatially far away with a different class', () => {
    // Person at left, car at right — IoU≈0, high centre distance, class mismatch → separate tracks
    const t1 = updateTracks([], [det('person', 0, 0, 50, 100)]);
    const t2 = updateTracks(t1, [det('car', 700, 0, 50, 100)]);
    expect(t2).toHaveLength(2);
    expect(t2.find(t => t.class === 'person')!.ageMs).toBe(INFERENCE_INTERVAL_MS);
    expect(t2.find(t => t.class === 'car')!.ageMs).toBe(0);
  });

  it('handles multiple detections with globally optimal (Hungarian) matching', () => {
    const t1 = updateTracks([], [
      det('person', 100, 100, 50, 100),
      det('person', 300, 100, 50, 100),
    ]);
    expect(t1).toHaveLength(2);

    const t2 = updateTracks(t1, [
      det('person', 103, 102, 50, 100),
      det('person', 303, 101, 50, 100),
    ]);
    expect(t2).toHaveLength(2);
    expect(t2.map(t => t.id).sort()).toEqual(t1.map(t => t.id).sort());
  });
});

// ── Aging and dropping ────────────────────────────────────────────────────────

describe('updateTracks — aging', () => {
  it('ages unmatched tracks by dtMs each call and drops them after TRACKER_MAX_AGE_MS', () => {
    let tracks = updateTracks([], [det('person', 100, 100, 50, 100)]);

    // 10 frames × 100 ms = 1 000 ms — track survives at the boundary
    for (let i = 1; i <= 10; i++) {
      tracks = updateTracks(tracks, []);
      expect(tracks).toHaveLength(1);
      expect(tracks[0].ageMs).toBe(i * INFERENCE_INTERVAL_MS);
    }

    // Frame 11: ageMs = 1 100 > 1 000 ms → dropped
    tracks = updateTracks(tracks, []);
    expect(tracks).toHaveLength(0);
  });

  it('respects a custom dtMs for fps-invariant aging', () => {
    // 200 ms per call — dropped after 5 calls (5 × 200 = 1000 ms, frame 6 → 1200 > 1000)
    let tracks = updateTracks([], [det('person', 100, 100, 50, 100)], 0, 200);
    for (let i = 1; i <= 5; i++) {
      tracks = updateTracks(tracks, [], 0, 200);
      expect(tracks).toHaveLength(1);
    }
    tracks = updateTracks(tracks, [], 0, 200);
    expect(tracks).toHaveLength(0);
  });
});

// ── Velocity / kinematics ─────────────────────────────────────────────────────

describe('updateTracks — velocity', () => {
  it('computes EMA-smoothed vx/vy in px/ms on matched tracks', () => {
    const dtMs = INFERENCE_INTERVAL_MS; // 100 ms
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)], 0, dtMs);

    // Move 20 px to the right → raw vx = 20 / 100 = 0.2 px/ms
    const t2 = updateTracks(t1, [det('person', 120, 100, 50, 100)], 0, dtMs);
    // EMA: α * 0.2 + (1-α) * 0 = 0.3 * 0.2 = 0.06 px/ms
    expect(t2[0].vx).toBeCloseTo(0.06, 3);
    expect(t2[0].vy).toBeCloseTo(0, 3);
  });

  it('computes positive areaGrowthRate for an approaching object', () => {
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)]);
    const t2 = updateTracks(t1, [det('person', 100, 100, 55, 110)]);

    // 6050 vs 5000 → 21% growth
    expect(t2[0].areaGrowthRate).toBeGreaterThan(0);
    expect(t2).toHaveLength(1);
    expect(t2[0].id).toBe(t1[0].id);
  });

  it('handles zero-area edge case without crashing', () => {
    const t1 = updateTracks([], [det('person', 100, 100, 0, 0)]);
    const t2 = updateTracks(t1, [det('person', 100, 100, 50, 100)]);
    expect(t2[0].areaGrowthRate).toBe(0);
  });
});

// ── Class voting ──────────────────────────────────────────────────────────────

describe('updateTracks — class voting', () => {
  it('returns the modal class from classHistory', () => {
    let tracks = updateTracks([], [det('person', 100, 100, 50, 100)]);
    tracks = updateTracks(tracks, [det('person', 101, 101, 50, 100)]);
    tracks = updateTracks(tracks, [det('bicycle', 102, 102, 50, 100)]); // mis-label
    // History = ['person','person','bicycle'] → mode = 'person'
    expect(tracks[0].class).toBe('person');
    expect(tracks[0].classHistory).toEqual(['person', 'person', 'bicycle']);
  });

  it('slides the window and forgets old labels after TRACKER_CLASS_HISTORY_LEN frames', () => {
    // 5-frame window: seed 3× person, then feed 5× car → history is all car
    let tracks = updateTracks([], [det('person', 100, 100, 50, 100)]);
    tracks = updateTracks(tracks, [det('person', 101, 101, 50, 100)]);
    tracks = updateTracks(tracks, [det('person', 102, 102, 50, 100)]);
    for (let i = 0; i < 5; i++) {
      tracks = updateTracks(tracks, [det('car', 103 + i, 103 + i, 50, 100)]);
    }
    // History window (length 5) should be all 'car' now
    expect(tracks[0].classHistory.every(c => c === 'car')).toBe(true);
    expect(tracks[0].class).toBe('car');
  });
});

// ── Eviction at MAX_TRACKS ────────────────────────────────────────────────────

describe('updateTracks — eviction at MAX_TRACKS', () => {
  it('evicts the lowest-priority track when total would exceed MAX_TRACKS', () => {
    // Seed MAX_TRACKS tier-3 safe books.
    let tracks = updateTracks([], Array.from({ length: MAX_TRACKS }, (_, i) =>
      det('book', i * 20, 0, 5, 5)));
    expect(tracks).toHaveLength(MAX_TRACKS);

    // Next frame: all books continue AND a new tier-1 danger car enters.
    const continuing = tracks.map(t => det(t.class, t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]));
    const newHazard  = det('car', 900, 300, 600, 600);
    tracks = updateTracks(tracks, [...continuing, newHazard]);

    expect(tracks).toHaveLength(MAX_TRACKS);
    expect(tracks.some(t => t.class === 'car')).toBe(true);
    expect(tracks.filter(t => t.class === 'book').length).toBe(MAX_TRACKS - 1);
  });

  it('never evicts a tier-1 danger track in favour of a tier-3 safe one', () => {
    let tracks = updateTracks([], [det('car', 100, 100, 600, 600)]);
    const fillers = Array.from({ length: MAX_TRACKS }, (_, i) => det('book', 900 + i, 0, 5, 5));
    tracks = updateTracks(tracks, [det('car', 100, 100, 600, 600), ...fillers]);

    expect(tracks).toHaveLength(MAX_TRACKS);
    expect(tracks.some(t => t.class === 'car')).toBe(true);
  });
});
