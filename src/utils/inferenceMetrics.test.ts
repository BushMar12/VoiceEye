import { describe, it, expect, beforeEach, vi } from 'vitest';
import { inferenceMetrics } from './inferenceMetrics';

// Mock performance.now for deterministic testing
let mockNow = 0;
vi.stubGlobal('performance', {
  now: () => mockNow,
});

beforeEach(() => {
  inferenceMetrics.reset();
  mockNow = 0;
});

describe('InferenceMetrics', () => {
  it('starts with zero state', () => {
    const summary = inferenceMetrics.getSummary();
    expect(summary.totalFrames).toBe(0);
    expect(summary.avgLatencyMs).toBe(0);
    expect(summary.p95LatencyMs).toBe(0);
    expect(summary.fps).toBe(0);
    expect(summary.avgDetectionsPerFrame).toBe(0);
  });

  it('records a single inference correctly', () => {
    mockNow = 1000;
    inferenceMetrics.recordInference(50, [
      { bbox: [0, 0, 10, 10], class: 'person', score: 0.85 },
    ]);

    const summary = inferenceMetrics.getSummary();
    expect(summary.totalFrames).toBe(1);
    expect(summary.avgLatencyMs).toBe(50);
    expect(summary.avgDetectionsPerFrame).toBe(1);
  });

  it('computes average latency across frames', () => {
    mockNow = 0;
    inferenceMetrics.recordInference(40, []);
    mockNow = 100;
    inferenceMetrics.recordInference(60, []);

    const summary = inferenceMetrics.getSummary();
    expect(summary.avgLatencyMs).toBe(50);
  });

  it('computes p95 latency', () => {
    // Record 20 frames: 19 at 10ms, 1 at 200ms
    for (let i = 0; i < 19; i++) {
      mockNow = i * 100;
      inferenceMetrics.recordInference(10, []);
    }
    mockNow = 1900;
    inferenceMetrics.recordInference(200, []);

    const summary = inferenceMetrics.getSummary();
    // p95 index = floor(20 * 0.95) = 19 → the 200ms entry
    expect(summary.p95LatencyMs).toBe(200);
  });

  it('computes FPS from elapsed time', () => {
    mockNow = 0;
    inferenceMetrics.recordInference(10, []);
    mockNow = 1000; // 1 second later
    inferenceMetrics.recordInference(10, []);

    const summary = inferenceMetrics.getSummary();
    // 2 frames over 1 second = 2 fps
    expect(summary.fps).toBe(2);
  });

  it('tracks detection counts', () => {
    mockNow = 0;
    inferenceMetrics.recordInference(10, [
      { bbox: [0, 0, 10, 10], class: 'person', score: 0.9 },
      { bbox: [20, 20, 10, 10], class: 'car', score: 0.8 },
    ]);
    mockNow = 100;
    inferenceMetrics.recordInference(10, [
      { bbox: [0, 0, 10, 10], class: 'person', score: 0.9 },
    ]);

    const summary = inferenceMetrics.getSummary();
    expect(summary.avgDetectionsPerFrame).toBe(1.5);
  });

  it('buckets confidence scores correctly', () => {
    mockNow = 0;
    inferenceMetrics.recordInference(10, [
      { bbox: [0, 0, 10, 10], class: 'a', score: 0.55 },
      { bbox: [0, 0, 10, 10], class: 'b', score: 0.65 },
      { bbox: [0, 0, 10, 10], class: 'c', score: 0.75 },
      { bbox: [0, 0, 10, 10], class: 'd', score: 0.85 },
      { bbox: [0, 0, 10, 10], class: 'e', score: 0.95 },
    ]);

    const dist = inferenceMetrics.getSummary().confidenceDistribution;
    expect(dist['0.5-0.6']).toBe(1);
    expect(dist['0.6-0.7']).toBe(1);
    expect(dist['0.7-0.8']).toBe(1);
    expect(dist['0.8-0.9']).toBe(1);
    expect(dist['0.9-1.0']).toBe(1);
  });

  it('uses circular buffer — older entries are evicted', () => {
    // Record 110 frames (buffer size is 100)
    for (let i = 0; i < 110; i++) {
      mockNow = i * 100;
      inferenceMetrics.recordInference(i, []);
    }

    const summary = inferenceMetrics.getSummary();
    expect(summary.totalFrames).toBe(110);
    // Average should be over last 100 entries (10..109)
    // avg = (10 + 11 + ... + 109) / 100 = (100 * 59.5) / 100 = 59.5
    expect(summary.avgLatencyMs).toBe(59.5);
  });

  it('resets to zero state', () => {
    mockNow = 0;
    inferenceMetrics.recordInference(50, [
      { bbox: [0, 0, 10, 10], class: 'person', score: 0.9 },
    ]);
    inferenceMetrics.reset();

    const summary = inferenceMetrics.getSummary();
    expect(summary.totalFrames).toBe(0);
    expect(summary.avgLatencyMs).toBe(0);
  });
});
