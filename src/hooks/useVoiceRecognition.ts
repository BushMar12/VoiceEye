import { useCallback, useEffect, useRef, useState } from 'react';

// ─── Minimal Web Speech API typings ───────────────────────────────────────────
// lib.dom.d.ts ships incomplete types for SpeechRecognition (still flagged
// as "experimental"). Declare the shape our code actually uses.

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  start: () => void;
  stop: () => void;
}

interface SpeechRecognitionErrorEventLike {
  error: string;
}

interface SpeechRecognitionEventLike {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

interface WindowWithSpeech extends Window {
  SpeechRecognition?: SpeechRecognitionConstructor;
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
}

// ─── Capability probe ────────────────────────────────────────────────────────
// Exported so callers can decide which UI to render before the hook even runs,
// and so it can be unit-tested without exercising the hook lifecycle.
export function isSpeechRecognitionSupported(
  win: WindowWithSpeech = (typeof window === 'undefined'
    ? ({} as WindowWithSpeech)
    : (window as WindowWithSpeech)),
): boolean {
  return !!(win.SpeechRecognition || win.webkitSpeechRecognition);
}

// ─── Command parser ───────────────────────────────────────────────────────────
// No wake-word stripping any more — the hold-to-talk gesture is the explicit
// activation signal, so the user just says "describe" / "read" / "find my
// keys". A residual "voice eye" is still tolerated in case a user uses it
// out of habit; it just becomes whitespace.

export type ParsedCommand =
  | { type: 'describe' }
  | { type: 'read' }
  | { type: 'search'; query: string }
  | { type: 'none' };

const RESIDUAL_WAKE_PHRASES = /\bvoice\s+(eye|i|ai|hi)\b/g;

function normaliseTranscript(transcript: string): string {
  return transcript
    .toLowerCase()
    .replace(RESIDUAL_WAKE_PHRASES, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

const DESCRIBE_PATTERNS: RegExp[] = [
  /\bdescri(be|bes|bed|bing|ption)\b/,
  /what'?s?\s+(there|around|in front|happening|going on)/,
  /what (do|can) you see/,
  /what is (this|it|that)/,
  /look around/,
  /\bscene\b/,
  /tell me (what|about)/,
];

const READ_PATTERNS: RegExp[] = [
  /\bread\b/,
  /\btext\b/,
  /what does (it|this|that|the sign|the label|the screen) say/,
];

// Search patterns return the remaining query text as capture group 1
const SEARCH_PATTERNS: RegExp[] = [
  /\b(?:find|search(?: for)?|look for|where(?:'?s| is))\s+(.+)/,
  /\b(?:find|search)\s*$/,
];

export function parseVoiceCommand(transcript: string): ParsedCommand {
  const cleaned = normaliseTranscript(transcript);

  for (const p of SEARCH_PATTERNS) {
    const m = cleaned.match(p);
    if (m) {
      const query = m[1]?.trim();
      return { type: 'search', query: query || 'object' };
    }
  }

  if (READ_PATTERNS.some(p => p.test(cleaned))) {
    return { type: 'read' };
  }

  if (DESCRIBE_PATTERNS.some(p => p.test(cleaned))) {
    return { type: 'describe' };
  }

  return { type: 'none' };
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

interface UseVoiceRecognitionOptions {
  // Permission probe — same gating useDetectionLoop used. Used to decide
  // whether to even attempt SpeechRecognition.
  enabled: boolean;
  // True if the VLM is mid-call; we still listen so the user can be told
  // "still analysing", but we don't fire the same command twice.
  isVLMBusy: boolean;
  onDescribe: () => void;
  onRead: () => void;
  onSearch: (query: string) => void;
  // Surface what the engine heard for the visible "Heard: ..." flash.
  onHeard: (transcript: string) => void;
  // Called after onresult dispatches (or after a recognition end with no
  // dispatch). Lets the parent (App.tsx via useHoldToTalk) close the
  // visual command window without waiting for the 8 s timeout.
  onResolved: (matched: boolean) => void;
  playBeep: () => void;
  speakQuick: (text: string) => void;
}

export function useVoiceRecognition({
  enabled,
  isVLMBusy,
  onDescribe,
  onRead,
  onSearch,
  onHeard,
  onResolved,
  playBeep,
  speakQuick,
}: UseVoiceRecognitionOptions) {
  const [isListening, setIsListening] = useState(false);
  // Probed once on mount; never changes for the life of the page.
  const [isSupported] = useState<boolean>(() => isSpeechRecognitionSupported());

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  // True between activate() and the next onend. Prevents activate() from
  // creating overlapping recognition sessions if the user lock-ins twice.
  const isActiveRef = useRef(false);
  // Latched on activate to make sure onResolved fires exactly once per
  // session, regardless of which path (onresult OR onend OR error) runs first.
  const resolvedRef = useRef(false);
  // Guard so the "not supported" log only ever fires once per page load
  const unsupportedLoggedRef = useRef(false);

  const stateRef = useRef({
    isVLMBusy, onDescribe, onRead, onSearch, onHeard, onResolved, playBeep, speakQuick,
  });
  useEffect(() => {
    stateRef.current = {
      isVLMBusy, onDescribe, onRead, onSearch, onHeard, onResolved, playBeep, speakQuick,
    };
  }, [isVLMBusy, onDescribe, onRead, onSearch, onHeard, onResolved, playBeep, speakQuick]);

  // ── Construct the SpeechRecognition instance once we know we have a
  // ── supported browser and the parent is enabled. The instance is reused
  // ── across activate() calls — we just call start()/stop() on it.
  useEffect(() => {
    if (!enabled) return;

    if (!isSupported) {
      if (!unsupportedLoggedRef.current) {
        unsupportedLoggedRef.current = true;
        console.info('[voice] SpeechRecognition unavailable — hold-to-talk falls back to tap-only describe');
      }
      return;
    }

    const w = window as WindowWithSpeech;
    const SpeechRecognition = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    // continuous=false means the engine stops on the first sentence — that
    // matches the on-demand model: the user has already opened the window
    // with a hold, so we want a single sentence and then auto-close.
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognitionRef.current = recognition;

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onerror = (event: SpeechRecognitionErrorEventLike) => {
      // `no-speech` and `aborted` are routine: no-speech fires if the user
      // didn't talk during the window, aborted fires when we deactivate().
      // Anything else is logged for diagnosis.
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        console.warn('[voice] recognition error:', event.error);
      }
    };

    recognition.onend = () => {
      setIsListening(false);
      isActiveRef.current = false;
      // If a result already arrived, resolvedRef is true and we've already
      // notified the parent. If the user said nothing, this is where we
      // signal "no match" so the visible window can close.
      if (!resolvedRef.current) {
        resolvedRef.current = true;
        stateRef.current.onResolved(false);
      }
    };

    recognition.onresult = (event: SpeechRecognitionEventLike) => {
      const transcript: string =
        event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
      const s = stateRef.current;

      const parsed = parseVoiceCommand(transcript);
      console.info('[voice] heard:', JSON.stringify(transcript), { parsed });
      s.onHeard(transcript);

      if (parsed.type === 'none') {
        // Heard speech but no recognised command — let the window close
        // through onend without an extra beep or spoken reply.
        return;
      }

      if (s.isVLMBusy) {
        s.playBeep();
        s.speakQuick('Still analysing, one moment.');
      } else {
        s.playBeep();
        if (parsed.type === 'describe') s.onDescribe();
        else if (parsed.type === 'read')   s.onRead();
        else if (parsed.type === 'search') s.onSearch(parsed.query);
      }

      // Mark resolved here so onend doesn't double-fire onResolved(false).
      resolvedRef.current = true;
      s.onResolved(true);
    };

    return () => {
      recognition.onstart = null;
      recognition.onend = null;
      recognition.onerror = null;
      recognition.onresult = null;
      try { recognition.stop(); } catch { /* already stopped */ }
      recognitionRef.current = null;
      isActiveRef.current = false;
      resolvedRef.current = false;
      setIsListening(false);
    };
  }, [enabled, isSupported]);

  // ── activate(): start a single recognition session. Called by the
  // ── press-and-hold gesture lock-in.
  const activate = useCallback(() => {
    if (!enabled) return;

    if (!isSupported) {
      // No SpeechRecognition: report "no match" immediately so the visible
      // command window closes. App.tsx's lock-in path can then fall back
      // to firing vlm.trigger() (Describe) on its own.
      stateRef.current.onResolved(false);
      return;
    }

    const rec = recognitionRef.current;
    if (!rec) {
      // useEffect hasn't run yet (shouldn't happen in practice given that
      // activate() is invoked after a 3 s hold). Best effort: close cleanly.
      stateRef.current.onResolved(false);
      return;
    }

    if (isActiveRef.current) return;
    isActiveRef.current = true;
    resolvedRef.current = false;
    try {
      rec.start();
    } catch {
      // Some browsers throw if start() is called before a previous session
      // has fully ended. Reset state and surface as no-match.
      isActiveRef.current = false;
      resolvedRef.current = true;
      stateRef.current.onResolved(false);
    }
  }, [enabled, isSupported]);

  // ── deactivate(): cancel an in-flight recognition session (e.g. user
  // ── lifted off without the hold completing, or the parent timed out).
  const deactivate = useCallback(() => {
    if (!isActiveRef.current) return;
    const rec = recognitionRef.current;
    if (!rec) return;
    try {
      rec.stop();
    } catch {
      // Best effort. onend will handle cleanup.
    }
  }, []);

  return { isListening, isSupported, activate, deactivate };
}
