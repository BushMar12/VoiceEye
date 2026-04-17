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
