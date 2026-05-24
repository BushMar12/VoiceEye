import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// All heavy ML modules are mocked — the goal of this test file is to verify
// the speak() gating around isCommandWindowOpen, not the YOLO pipeline.
vi.mock('../utils/yolo', () => ({
  loadYoloModel: vi.fn(async () => ({ /* fake InferenceSession */ })),
  runYolo: vi.fn(async () => [
    { bbox: [0, 0, 100, 100] as [number, number, number, number], score: 0.9, class: 'person' },
  ]),
}));

vi.mock('../utils/tracker', () => ({
  updateTracks: vi.fn(() => [
    {
      id: 1,
      class: 'person',
      bbox: [0, 0, 100, 100] as [number, number, number, number],
      score: 0.9,
      status: 'confirmed',
      proximityZone: 'danger',
      ageMs: 0,
    },
  ]),
}));

vi.mock('../utils/distance', () => ({
  estimateDistance: () => '~1.2m',
  estimateDistanceForSpeech: () => 'about 1 meter',
}));

vi.mock('../utils/inferenceMetrics', () => ({
  inferenceMetrics: {
    recordInference: vi.fn(),
    recordAttention: vi.fn(),
  },
}));

// The attention pipeline returns one announcement and one de-escalation tone
// every call. The gating under test sits AROUND this output, so this is the
// only mock shape that matters.
vi.mock('../utils/attention', () => ({
  createAttentionState: () => ({ clusters: new Map() }),
  runAttention: vi.fn(() => ({
    toAnnounce: [{
      kind: 'single',
      class: 'person',
      bbox: [0, 0, 100, 100],
      reason: 'zone-escalation',
    }],
    deescalationTones: ['1'],
    renderTracks: [],
    suppressedCount: 0,
  })),
}));

import { useDetectionLoop } from './useDetectionLoop';

// Minimal HTMLVideoElement stand-in — useDetectionLoop only reads .videoWidth
// and .videoHeight, and passes the element through to runYolo (already mocked).
function makeVideoElement(): HTMLVideoElement {
  const v = document.createElement('video');
  Object.defineProperty(v, 'videoWidth', { value: 640 });
  Object.defineProperty(v, 'videoHeight', { value: 480 });
  return v;
}

function defaultSettings() {
  return {
    ttsRate: 1.0,
    confThreshold: 0.25,
    verbosity: 'normal' as const,
    hapticEnabled: true,
    cameraVfovDeg: 70,
  };
}

describe('useDetectionLoop — command-window mute', () => {
  let rafCallbacks: FrameRequestCallback[] = [];
  let perfNow = 0;

  beforeEach(() => {
    rafCallbacks = [];
    perfNow = 1000;
    vi.spyOn(performance, 'now').mockImplementation(() => perfNow);
    // Drive the rAF loop manually so we can advance the simulated clock
    // between frames.
    vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation((cb: FrameRequestCallback) => {
      rafCallbacks.push(cb);
      return rafCallbacks.length;
    });
    vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  async function flushFrame(): Promise<void> {
    const cbs = rafCallbacks.splice(0, rafCallbacks.length);
    await act(async () => {
      for (const cb of cbs) cb(perfNow);
      // Let the awaited runYolo + everything after resolve.
      await Promise.resolve();
      await Promise.resolve();
    });
  }

  it('calls speak when the command window is closed', async () => {
    const speak = vi.fn();
    const playBeep = vi.fn();
    const videoElement = makeVideoElement();

    renderHook(() => useDetectionLoop({
      videoElement,
      isProcessingSlowLane: false,
      isCommandWindowOpen: false,
      settings: defaultSettings(),
      speak,
      playBeep,
    }));

    // Frame 1: model load awaits; loop hasn't fired runDetection yet because
    // model state is still null. After the load promise resolves, the effect
    // re-runs and runDetection schedules its first frame.
    await act(async () => {
      // Let loadYoloModel's promise resolve so setModel fires.
      await Promise.resolve();
      await Promise.resolve();
    });
    // Drive the rAF loop forward. INFERENCE_INTERVAL_MS = 100 ms; first call
    // sees lastInferenceTime=0 so it always runs.
    await flushFrame();

    expect(speak).toHaveBeenCalledTimes(1);
    expect(speak.mock.calls[0][0]).toContain('person');
    // De-escalation tone fires unconditionally when present.
    expect(playBeep).toHaveBeenCalled();
  });

  it('suppresses speak but still fires the de-escalation tone while the command window is open', async () => {
    const speak = vi.fn();
    const playBeep = vi.fn();
    const videoElement = makeVideoElement();

    renderHook(() => useDetectionLoop({
      videoElement,
      isProcessingSlowLane: false,
      isCommandWindowOpen: true,
      settings: defaultSettings(),
      speak,
      playBeep,
    }));

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    await flushFrame();

    expect(speak).not.toHaveBeenCalled();
    expect(playBeep).toHaveBeenCalled();
  });
});
