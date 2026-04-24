import { describe, it, expect } from 'vitest';
import { updateTracks, classifyProximity } from './tracker';
import { MAX_TRACKS } from '../config';
import type { Detection } from './yolo';

// Helper to create a detection
function det(cls: string, x: number, y: number, w: number, h: number, score = 0.9): Detection {
  return { bbox: [x, y, w, h], class: cls, score };
}

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

describe('updateTracks', () => {
  it('creates new tracks for unmatched detections', () => {
    const tracks = updateTracks([], [det('person', 10, 10, 50, 100)]);
    expect(tracks).toHaveLength(1);
    expect(tracks[0].class).toBe('person');
    expect(tracks[0].age).toBe(0);
    expect(tracks[0].proximityZone).toBe('safe');
  });

  it('classifies proximityZone from screenArea per frame', () => {
    const screenArea = 800 * 600; // 480000
    // 600*600 = 360000 → 75% → 'danger'
    const danger = updateTracks([], [det('car', 0, 0, 600, 600)], screenArea);
    expect(danger[0].proximityZone).toBe('danger');

    // 400*350 = 140000 → 29% → 'near'
    const near = updateTracks([], [det('car', 0, 0, 400, 350)], screenArea);
    expect(near[0].proximityZone).toBe('near');

    // 100*100 = 10000 → 2% → 'safe'
    const safe = updateTracks([], [det('car', 0, 0, 100, 100)], screenArea);
    expect(safe[0].proximityZone).toBe('safe');
  });

  it('updates proximityZone on a matched track when bbox grows', () => {
    const screenArea = 800 * 600;
    // Spawn a small safe car
    const t1 = updateTracks([], [det('car', 100, 100, 100, 100)], screenArea);
    expect(t1[0].proximityZone).toBe('safe');

    // Grow it slightly (IoU above 0.3 keeps it the same track), still safe
    // (200*200)/(800*600) ≈ 8% → safe
    const t2 = updateTracks(t1, [det('car', 100, 100, 200, 200)], screenArea);
    expect(t2[0].id).toBe(t1[0].id);
    expect(t2[0].proximityZone).toBe('safe');
  });

  it('assigns unique IDs to new tracks', () => {
    const tracks = updateTracks([], [
      det('person', 10, 10, 50, 100),
      det('car', 200, 200, 80, 60),
    ]);
    expect(tracks).toHaveLength(2);
    expect(tracks[0].id).not.toBe(tracks[1].id);
  });

  it('matches detections to existing tracks by class + IoU', () => {
    // Frame 1: create track
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)]);
    const id = t1[0].id;

    // Frame 2: same class, overlapping bbox → should match
    const t2 = updateTracks(t1, [det('person', 105, 102, 50, 100)]);
    expect(t2).toHaveLength(1);
    expect(t2[0].id).toBe(id);
    expect(t2[0].age).toBe(0);
  });

  it('does not match detections of different classes', () => {
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)]);
    // Same bbox but different class → new track + old one ages
    const t2 = updateTracks(t1, [det('car', 100, 100, 50, 100)]);
    expect(t2).toHaveLength(2);
    expect(t2.find(t => t.class === 'person')!.age).toBe(1);
    expect(t2.find(t => t.class === 'car')!.age).toBe(0);
  });

  it('ages unmatched tracks and drops them after MAX_AGE', () => {
    let tracks = updateTracks([], [det('person', 100, 100, 50, 100)]);

    // Run 10 frames with no detections — track should age each frame
    for (let i = 1; i <= 10; i++) {
      tracks = updateTracks(tracks, []);
      if (i <= 10) {
        expect(tracks).toHaveLength(1);
        expect(tracks[0].age).toBe(i);
      }
    }

    // Frame 11: track should be dropped (age > MAX_AGE=10)
    tracks = updateTracks(tracks, []);
    expect(tracks).toHaveLength(0);
  });

  it('computes centroid and area for new tracks', () => {
    const tracks = updateTracks([], [det('person', 100, 200, 50, 80)]);
    expect(tracks[0].lastCentroid).toEqual([125, 240]); // x + w/2, y + h/2
    expect(tracks[0].lastArea).toBe(4000); // 50 * 80
  });

  it('computes EMA-smoothed velocity on matched tracks', () => {
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)]);
    // Move 20px to the right
    const t2 = updateTracks(t1, [det('person', 120, 100, 50, 100)]);

    // vx should be EMA_ALPHA * 20 + (1 - EMA_ALPHA) * 0 = 0.3 * 20 = 6
    expect(t2[0].vx).toBeCloseTo(6, 1);
    expect(t2[0].vy).toBeCloseTo(0, 1);
  });

  it('computes area growth rate for approaching objects', () => {
    const t1 = updateTracks([], [det('person', 100, 100, 50, 100)]);
    // Slightly larger bbox (still high IoU overlap): 55 * 110 = 6050 vs 50 * 100 = 5000
    const t2 = updateTracks(t1, [det('person', 100, 100, 55, 110)]);

    // Growth rate = EMA_ALPHA * ((6050 - 5000) / 5000) = 0.3 * 0.21 = 0.063
    expect(t2[0].areaGrowthRate).toBeGreaterThan(0);
    // Verify matched track (same id, not a new track)
    expect(t2).toHaveLength(1);
    expect(t2[0].id).toBe(t1[0].id);
  });

  it('handles multiple detections with greedy matching', () => {
    // Two persons, close together
    const t1 = updateTracks([], [
      det('person', 100, 100, 50, 100),
      det('person', 300, 100, 50, 100),
    ]);
    expect(t1).toHaveLength(2);

    // Both move slightly
    const t2 = updateTracks(t1, [
      det('person', 103, 102, 50, 100),
      det('person', 303, 101, 50, 100),
    ]);
    expect(t2).toHaveLength(2);
    // IDs should persist
    expect(t2.map(t => t.id).sort()).toEqual(t1.map(t => t.id).sort());
  });

  it('handles zero-area edge case in growth rate', () => {
    // Detection with zero area shouldn't crash
    const t1 = updateTracks([], [det('person', 100, 100, 0, 0)]);
    const t2 = updateTracks(t1, [det('person', 100, 100, 50, 100)]);
    // lastArea was 0, so growth rate should be 0 (guarded)
    expect(t2[0].areaGrowthRate).toBe(0);
  });
});

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
