import { describe, it, expect } from 'vitest';
import { estimateDistance } from './distance';

describe('estimateDistance', () => {
  it('returns empty string for zero bbox height', () => {
    expect(estimateDistance(0, 480, 'person')).toBe('');
  });

  it('returns empty string for negative bbox height', () => {
    expect(estimateDistance(-10, 480, 'person')).toBe('');
  });

  it('returns empty string for zero video height', () => {
    expect(estimateDistance(100, 0, 'person')).toBe('');
  });

  it('returns a valid distance string for person', () => {
    // Person (1.70m height) at roughly half-screen height in 480px frame
    const result = estimateDistance(240, 480, 'person');
    expect(result).toMatch(/^~\d+\.\d+m$/);
  });

  it('returns ">10m" for very distant objects', () => {
    // Very small bbox = far away
    const result = estimateDistance(5, 480, 'person');
    expect(result).toBe('>10m');
  });

  it('returns ">5m" for moderately distant objects', () => {
    // bbox=80px for person (1.70m) in 480px frame → ~7.3m
    const result = estimateDistance(80, 480, 'person');
    expect(result).toBe('>5m');
  });

  it('uses default height for unknown classes', () => {
    const result = estimateDistance(100, 480, 'unknown_object_xyz');
    expect(result).toMatch(/^~\d+\.\d+m$|^>5m$|^>10m$/);
  });

  it('handles case-insensitive class names', () => {
    const lower = estimateDistance(100, 480, 'person');
    const upper = estimateDistance(100, 480, 'Person');
    expect(lower).toBe(upper);
  });

  it('estimates closer distance for larger bbox', () => {
    // Bigger bbox = closer
    const close = estimateDistance(400, 480, 'person');
    const far = estimateDistance(100, 480, 'person');

    // Parse numeric values
    const parseDistance = (s: string) => {
      if (s.startsWith('>10')) return 11;
      if (s.startsWith('>5')) return 6;
      return parseFloat(s.replace('~', '').replace('m', ''));
    };

    expect(parseDistance(close)).toBeLessThan(parseDistance(far));
  });

  it('returns correct format with tilde and m suffix', () => {
    const result = estimateDistance(200, 480, 'car');
    // Should be ~X.Xm, >5m, or >10m
    expect(result).toMatch(/^(~\d+\.\d+m|>5m|>10m)$/);
  });

  it('uses known height for car (1.50m)', () => {
    // With known height of 1.50m, larger bbox → closer
    const result = estimateDistance(300, 480, 'car');
    expect(result).toMatch(/^~\d+\.\d+m$/);
  });

  it('uses known height for small objects like cup (0.12m)', () => {
    const result = estimateDistance(100, 480, 'cup');
    // Cup is tiny (0.12m), so even at 100px bbox it should be very close
    expect(result).toMatch(/^~\d+\.\d+m$/);
  });
});
