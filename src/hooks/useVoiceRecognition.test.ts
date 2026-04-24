import { describe, it, expect } from 'vitest';
import { parseVoiceCommand, hasWakeWord } from './useVoiceRecognition';

// ─── hasWakeWord ──────────────────────────────────────────────────────────────

describe('hasWakeWord', () => {
  it('recognises the canonical wake phrase', () => {
    expect(hasWakeWord('voice eye describe the scene')).toBe(true);
  });

  it('recognises common mishear variants', () => {
    expect(hasWakeWord('voice i describe')).toBe(true);
    expect(hasWakeWord('voice ai describe')).toBe(true);
    expect(hasWakeWord('voice hi describe')).toBe(true);
    expect(hasWakeWord('boy eye describe')).toBe(true);
    expect(hasWakeWord('boy i describe')).toBe(true);
  });

  it('returns false when no wake word is present', () => {
    expect(hasWakeWord('describe the room')).toBe(false);
    expect(hasWakeWord('already there')).toBe(false);
    expect(hasWakeWord('')).toBe(false);
  });
});

// ─── parseVoiceCommand — describe ────────────────────────────────────────────

describe('parseVoiceCommand — describe', () => {
  it('matches the literal word "describe"', () => {
    expect(parseVoiceCommand('voice eye describe')).toEqual({ type: 'describe' });
  });

  it('matches "what\'s there"', () => {
    expect(parseVoiceCommand("voice eye what's there")).toEqual({ type: 'describe' });
  });

  it('matches "what do you see"', () => {
    expect(parseVoiceCommand('voice eye what do you see')).toEqual({ type: 'describe' });
  });

  it('matches "what is this"', () => {
    expect(parseVoiceCommand('voice eye what is this')).toEqual({ type: 'describe' });
  });

  it('matches "look around"', () => {
    expect(parseVoiceCommand('voice eye look around')).toEqual({ type: 'describe' });
  });

  it('matches "tell me what you see"', () => {
    expect(parseVoiceCommand('voice eye tell me what you see')).toEqual({ type: 'describe' });
  });

  it('matches "description"', () => {
    expect(parseVoiceCommand('voice eye description')).toEqual({ type: 'describe' });
  });

  it('does NOT match "reads" as describe', () => {
    const result = parseVoiceCommand('voice eye reads the sign');
    expect(result.type).not.toBe('describe');
  });
});

// ─── parseVoiceCommand — read ─────────────────────────────────────────────────

describe('parseVoiceCommand — read', () => {
  it('matches the literal word "read"', () => {
    expect(parseVoiceCommand('voice eye read')).toEqual({ type: 'read' });
  });

  it('matches "text"', () => {
    expect(parseVoiceCommand('voice eye text')).toEqual({ type: 'read' });
  });

  it('matches "what does the sign say"', () => {
    expect(parseVoiceCommand('what does the sign say')).toEqual({ type: 'read' });
  });

  it('does NOT match "ready" as a read command', () => {
    // "ready" contains "read" as a substring — word-boundary regex should exclude it
    const result = parseVoiceCommand('voice eye ready');
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
    const result = parseVoiceCommand('voice eye find my keys');
    expect(result).toEqual({ type: 'search', query: 'my keys' });
  });

  it('matches "search for a cup" and extracts query', () => {
    const result = parseVoiceCommand('voice eye search for a cup');
    expect(result).toEqual({ type: 'search', query: 'a cup' });
  });

  it('matches "look for the door" and extracts query', () => {
    const result = parseVoiceCommand('voice eye look for the door');
    expect(result).toEqual({ type: 'search', query: 'the door' });
  });

  it('matches "where is the exit" and extracts query', () => {
    const result = parseVoiceCommand("voice eye where is the exit");
    expect(result).toEqual({ type: 'search', query: 'the exit' });
  });

  it("matches \"where's the bathroom\"", () => {
    const result = parseVoiceCommand("voice eye where's the bathroom");
    expect(result).toEqual({ type: 'search', query: 'the bathroom' });
  });

  it('falls back to "object" when no query follows "find"', () => {
    const result = parseVoiceCommand('voice eye find');
    expect(result).toEqual({ type: 'search', query: 'object' });
  });
});

// ─── parseVoiceCommand — none ─────────────────────────────────────────────────

describe('parseVoiceCommand — none (wake word only or unrecognised)', () => {
  it('returns none for wake word alone', () => {
    expect(parseVoiceCommand('voice eye')).toEqual({ type: 'none' });
  });

  it('returns none for unrelated speech', () => {
    expect(parseVoiceCommand('hello there')).toEqual({ type: 'none' });
  });

  it('returns none for empty string', () => {
    expect(parseVoiceCommand('')).toEqual({ type: 'none' });
  });
});
