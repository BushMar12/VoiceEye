import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { parseVoiceCommand, isSpeechRecognitionSupported, useVoiceRecognition } from './useVoiceRecognition';

// ─── parseVoiceCommand — describe ────────────────────────────────────────────
// The parser used to also strip the "voice eye" wake word. The hook itself no
// longer requires that prefix (press-and-hold replaces the wake word) but the
// parser is still tolerant of a residual wake phrase from old habits.

describe('parseVoiceCommand — describe', () => {
  it('matches the literal word "describe"', () => {
    expect(parseVoiceCommand('describe')).toEqual({ type: 'describe' });
  });

  it('matches "what\'s there"', () => {
    expect(parseVoiceCommand("what's there")).toEqual({ type: 'describe' });
  });

  it('matches "what do you see"', () => {
    expect(parseVoiceCommand('what do you see')).toEqual({ type: 'describe' });
  });

  it('matches "look around"', () => {
    expect(parseVoiceCommand('look around')).toEqual({ type: 'describe' });
  });

  it('matches "description"', () => {
    expect(parseVoiceCommand('description')).toEqual({ type: 'describe' });
  });

  it('tolerates a residual "voice eye" prefix', () => {
    expect(parseVoiceCommand('voice eye describe')).toEqual({ type: 'describe' });
  });

  it('does NOT match "reads" as describe', () => {
    const result = parseVoiceCommand('reads the sign');
    expect(result.type).not.toBe('describe');
  });
});

// ─── parseVoiceCommand — read ─────────────────────────────────────────────────

describe('parseVoiceCommand — read', () => {
  it('matches the literal word "read"', () => {
    expect(parseVoiceCommand('read')).toEqual({ type: 'read' });
  });

  it('matches "text"', () => {
    expect(parseVoiceCommand('text')).toEqual({ type: 'read' });
  });

  it('matches "what does the sign say"', () => {
    expect(parseVoiceCommand('what does the sign say')).toEqual({ type: 'read' });
  });

  it('does NOT match "ready" as a read command', () => {
    const result = parseVoiceCommand('ready');
    expect(result.type).not.toBe('read');
  });

  it('does NOT match "already" as a read command', () => {
    const result = parseVoiceCommand('already done');
    expect(result.type).not.toBe('read');
  });
});

// ─── parseVoiceCommand — search ───────────────────────────────────────────────

describe('parseVoiceCommand — search', () => {
  it('matches "find my keys" and extracts the query', () => {
    const result = parseVoiceCommand('find my keys');
    expect(result).toEqual({ type: 'search', query: 'my keys' });
  });

  it('matches "search for a cup" and extracts query', () => {
    const result = parseVoiceCommand('search for a cup');
    expect(result).toEqual({ type: 'search', query: 'a cup' });
  });

  it('matches "look for the door" and extracts query', () => {
    const result = parseVoiceCommand('look for the door');
    expect(result).toEqual({ type: 'search', query: 'the door' });
  });

  it('matches "where is the exit" and extracts query', () => {
    const result = parseVoiceCommand('where is the exit');
    expect(result).toEqual({ type: 'search', query: 'the exit' });
  });

  it("matches \"where's the bathroom\"", () => {
    const result = parseVoiceCommand("where's the bathroom");
    expect(result).toEqual({ type: 'search', query: 'the bathroom' });
  });

  it('falls back to "object" when no query follows "find"', () => {
    const result = parseVoiceCommand('find');
    expect(result).toEqual({ type: 'search', query: 'object' });
  });
});

// ─── parseVoiceCommand — none ─────────────────────────────────────────────────

describe('parseVoiceCommand — none', () => {
  it('returns none for residual wake word alone', () => {
    expect(parseVoiceCommand('voice eye')).toEqual({ type: 'none' });
  });

  it('returns none for unrelated speech', () => {
    expect(parseVoiceCommand('hello there')).toEqual({ type: 'none' });
  });

  it('returns none for empty string', () => {
    expect(parseVoiceCommand('')).toEqual({ type: 'none' });
  });
});

// ─── isSpeechRecognitionSupported — capability probe for graceful fallback ──

describe('isSpeechRecognitionSupported', () => {
  // Each test mutates window.{SpeechRecognition,webkitSpeechRecognition}; reset
  // around every case so cross-test contamination can't hide a regression.
  const saved = {
    standard: (globalThis as { SpeechRecognition?: unknown }).SpeechRecognition,
    webkit: (globalThis as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition,
  };

  beforeEach(() => {
    delete (globalThis as { SpeechRecognition?: unknown }).SpeechRecognition;
    delete (globalThis as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition;
  });

  afterEach(() => {
    if (saved.standard !== undefined) {
      (globalThis as { SpeechRecognition?: unknown }).SpeechRecognition = saved.standard;
    }
    if (saved.webkit !== undefined) {
      (globalThis as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition = saved.webkit;
    }
  });

  it('returns false when neither global is present (jsdom default)', () => {
    expect(isSpeechRecognitionSupported()).toBe(false);
  });

  it('returns true when window.SpeechRecognition is defined', () => {
    (globalThis as { SpeechRecognition?: unknown }).SpeechRecognition = class FakeSR {};
    expect(isSpeechRecognitionSupported()).toBe(true);
  });

  it('returns true when only window.webkitSpeechRecognition is defined', () => {
    (globalThis as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition = class FakeWebkitSR {};
    expect(isSpeechRecognitionSupported()).toBe(true);
  });

  it('accepts an explicit window-like argument (no global mutation)', () => {
    expect(isSpeechRecognitionSupported({} as Window)).toBe(false);
    expect(
      isSpeechRecognitionSupported({ webkitSpeechRecognition: class {} } as unknown as Window),
    ).toBe(true);
  });
});

// ─── useVoiceRecognition — on-demand activate / deactivate ─────────────────

// FakeSpeechRecognition mirrors the spec subset we use. Tests grab the
// constructed instance off `instances` so they can drive onstart/onresult/onend.
class FakeSR {
  static instances: FakeSR[] = [];
  continuous = false;
  interimResults = false;
  lang = '';
  onstart: (() => void) | null = null;
  onerror: ((e: { error: string }) => void) | null = null;
  onend: (() => void) | null = null;
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null = null;
  start = vi.fn(() => { this.onstart?.(); });
  stop = vi.fn(() => { this.onend?.(); });
  constructor() {
    FakeSR.instances.push(this);
  }
  static reset() {
    FakeSR.instances = [];
  }
  // Helper for tests to drive a result event.
  fireResult(transcript: string) {
    this.onresult?.({
      results: [[{ transcript }]] as unknown as ArrayLike<ArrayLike<{ transcript: string }>>,
    });
  }
}

function defaultOpts() {
  return {
    enabled: true,
    isVLMBusy: false,
    onDescribe: vi.fn(),
    onRead: vi.fn(),
    onSearch: vi.fn(),
    onHeard: vi.fn(),
    onResolved: vi.fn(),
    playBeep: vi.fn(),
    speakQuick: vi.fn(),
  };
}

describe('useVoiceRecognition — on-demand lifecycle', () => {
  beforeEach(() => {
    FakeSR.reset();
    (globalThis as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition = FakeSR;
  });

  afterEach(() => {
    delete (globalThis as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition;
  });

  it('does not start recognition automatically on mount', () => {
    const opts = defaultOpts();
    renderHook(() => useVoiceRecognition(opts));
    // A FakeSR instance is constructed eagerly so start() is available, but
    // start() must NOT be called until activate() is invoked.
    expect(FakeSR.instances.length).toBe(1);
    expect(FakeSR.instances[0].start).not.toHaveBeenCalled();
  });

  it('activate() starts the recognition session exactly once', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    expect(FakeSR.instances[0].start).toHaveBeenCalledTimes(1);
    // Repeat activate() while active should be a no-op (no overlapping sessions).
    act(() => { result.current.activate(); });
    expect(FakeSR.instances[0].start).toHaveBeenCalledTimes(1);
  });

  it('dispatches onDescribe and onResolved(true) when a describe phrase is heard', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    act(() => { FakeSR.instances[0].fireResult('describe the room'); });

    expect(opts.onDescribe).toHaveBeenCalledTimes(1);
    expect(opts.onRead).not.toHaveBeenCalled();
    expect(opts.onSearch).not.toHaveBeenCalled();
    expect(opts.playBeep).toHaveBeenCalled();
    expect(opts.onHeard).toHaveBeenCalledWith('describe the room');
    expect(opts.onResolved).toHaveBeenCalledWith(true);
  });

  it('dispatches onRead when "read" is heard', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    act(() => { FakeSR.instances[0].fireResult('read the sign'); });
    expect(opts.onRead).toHaveBeenCalledTimes(1);
    expect(opts.onResolved).toHaveBeenCalledWith(true);
  });

  it('dispatches onSearch with the extracted query', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    act(() => { FakeSR.instances[0].fireResult('find my keys'); });
    expect(opts.onSearch).toHaveBeenCalledWith('my keys');
  });

  it('does NOT speak the long "I am listening" prompt on activate', () => {
    // Regression guard for the bug that triggered this refactor: the old
    // hook spoke a multi-second prompt that bled into the mic.
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    expect(opts.speakQuick).not.toHaveBeenCalled();
  });

  it('signals onResolved(false) when recognition ends without a match', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    // onend without a prior fireResult
    act(() => { FakeSR.instances[0].onend?.(); });
    expect(opts.onResolved).toHaveBeenCalledWith(false);
    expect(opts.onDescribe).not.toHaveBeenCalled();
  });

  it('treats unrecognised speech as no-match and does not fire any command', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    act(() => { FakeSR.instances[0].fireResult('hello there'); });
    expect(opts.onDescribe).not.toHaveBeenCalled();
    expect(opts.onRead).not.toHaveBeenCalled();
    expect(opts.onSearch).not.toHaveBeenCalled();
    expect(opts.onHeard).toHaveBeenCalledWith('hello there');
  });

  it('deactivate() stops the in-flight session', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    act(() => { result.current.deactivate(); });
    expect(FakeSR.instances[0].stop).toHaveBeenCalledTimes(1);
    // onend (driven by FakeSR.stop) should report no-match.
    expect(opts.onResolved).toHaveBeenCalledWith(false);
  });

  it('reports speakQuick "still analysing" when the VLM is busy', () => {
    const opts = defaultOpts();
    opts.isVLMBusy = true;
    const { result } = renderHook(() => useVoiceRecognition(opts));
    act(() => { result.current.activate(); });
    act(() => { FakeSR.instances[0].fireResult('describe'); });
    expect(opts.speakQuick).toHaveBeenCalledWith('Still analysing, one moment.');
    expect(opts.onDescribe).not.toHaveBeenCalled();
  });
});

describe('useVoiceRecognition — unsupported browser', () => {
  beforeEach(() => {
    delete (globalThis as { SpeechRecognition?: unknown }).SpeechRecognition;
    delete (globalThis as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition;
  });

  it('activate() resolves immediately as no-match when speech is unavailable', () => {
    const opts = defaultOpts();
    const { result } = renderHook(() => useVoiceRecognition(opts));
    expect(result.current.isSupported).toBe(false);
    act(() => { result.current.activate(); });
    expect(opts.onResolved).toHaveBeenCalledWith(false);
  });
});
