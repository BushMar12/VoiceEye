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
