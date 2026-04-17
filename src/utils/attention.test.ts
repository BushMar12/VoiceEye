import { describe, it, expect } from 'vitest';
import { createAttentionState, computePriority } from './attention';

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
    expect(detectReason(track, 'near', cd, 3000)).toBeNull();
  });

  it('returns "sustained" after entry delay in danger and cooldown elapsed', () => {
    const cd = mockCooldown({
      lastZone: 'danger',
      dangerEnteredAt: 0,
      lastSpokeAt: 0,
      sustainedCount: 0,
    });
    expect(detectReason(mockTrack(), 'danger', cd, 4000)).toBe('sustained');
  });

  it('caps sustained at SUSTAINED_MAX_COUNT', () => {
    const cd = mockCooldown({
      lastZone: 'danger',
      dangerEnteredAt: 0,
      lastSpokeAt: 0,
      sustainedCount: 3,
    });
    expect(detectReason(mockTrack(), 'danger', cd, 10000)).toBeNull();
  });
});
