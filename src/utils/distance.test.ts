import { describe, it, expect } from 'vitest';
import {
  calibrateVerticalFovDeg,
  distanceRelativeError,
  estimateDistance,
  estimateDistanceMeters,
  KNOWN_HEIGHTS_M,
} from './distance';

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

  it('returns numeric metres for real-world validation', () => {
    const estimate = estimateDistanceMeters(320, 720, 'person');

    expect(estimate).not.toBeNull();
    expect(estimate?.distanceM).toBeGreaterThan(0);
    expect(estimate?.rawDistanceM).toBeGreaterThan(0);
    expect(estimate?.objectHeightM).toBe(KNOWN_HEIGHTS_M.person);
    expect(estimate?.focalLengthPx).toBeGreaterThan(0);
    expect(estimate?.label).toMatch(/^(~\d+\.\d+m|>5m|>10m)$/);
  });

  it('returns null numeric estimate for invalid inputs', () => {
    expect(estimateDistanceMeters(0, 720, 'person')).toBeNull();
    expect(estimateDistanceMeters(100, 0, 'person')).toBeNull();
    expect(estimateDistanceMeters(100, 720, 'person', { verticalFovDeg: 0 })).toBeNull();
    expect(estimateDistanceMeters(100, 720, 'person', { verticalFovDeg: 180 })).toBeNull();
  });

  it('supports measured object-height override for real-world tests', () => {
    const defaultEstimate = estimateDistanceMeters(250, 720, 'door');
    const measuredDoorEstimate = estimateDistanceMeters(250, 720, 'door', {
      objectHeightM: 2.10,
    });

    expect(defaultEstimate).not.toBeNull();
    expect(measuredDoorEstimate).not.toBeNull();
    expect(measuredDoorEstimate?.objectHeightM).toBe(2.10);
    expect(measuredDoorEstimate!.distanceM).toBeGreaterThan(defaultEstimate!.distanceM);
  });

  it('supports per-class height overrides for validation fixtures', () => {
    const estimate = estimateDistanceMeters(120, 720, 'test target', {
      classHeightsM: { 'test target': 1.20 },
    });

    expect(estimate).not.toBeNull();
    expect(estimate?.objectHeightM).toBe(1.20);
  });

  it('calibrates vertical FOV from a measured real-world sample', () => {
    // A 1.7m person standing 3m away occupying 291px in a 720px frame
    // corresponds to roughly a 70 degree vertical FOV camera.
    const calibratedFov = calibrateVerticalFovDeg({
      bboxHeightPx: 291,
      frameHeightPx: 720,
      objectHeightM: 1.70,
      measuredDistanceM: 3,
    });

    expect(calibratedFov).not.toBeNull();
    expect(calibratedFov!).toBeGreaterThan(69);
    expect(calibratedFov!).toBeLessThan(71);
  });

  it('uses calibrated FOV to recover a measured distance within tolerance', () => {
    const calibratedFov = calibrateVerticalFovDeg({
      bboxHeightPx: 291,
      frameHeightPx: 720,
      objectHeightM: 1.70,
      measuredDistanceM: 3,
    });
    const estimate = estimateDistanceMeters(291, 720, 'person', {
      verticalFovDeg: calibratedFov!,
      objectHeightM: 1.70,
    });
    const error = distanceRelativeError(estimate!.distanceM, 3);

    expect(error).not.toBeNull();
    expect(error!).toBeLessThan(0.01);
  });

  it('returns null for invalid calibration samples', () => {
    expect(calibrateVerticalFovDeg({
      bboxHeightPx: 0,
      frameHeightPx: 720,
      objectHeightM: 1.70,
      measuredDistanceM: 3,
    })).toBeNull();
  });

  it('calculates relative error for real-world test reports', () => {
    expect(distanceRelativeError(2.7, 3)).toBeCloseTo(0.1);
    expect(distanceRelativeError(0, 3)).toBeNull();
    expect(distanceRelativeError(3, 0)).toBeNull();
  });
});
