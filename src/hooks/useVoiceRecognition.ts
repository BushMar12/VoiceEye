import { useState, useEffect, useRef } from 'react';

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

// ─── Wake-word detection ──────────────────────────────────────────────────────
const WAKE_WORDS = [
  'voice eye',
  'voice i',
  'voice ai',
  'voice hi',
  'boy eye',
  'boy i',
];

export function hasWakeWord(transcript: string): boolean {
  return WAKE_WORDS.some(w => transcript.includes(w));
}

// Strip the wake phrase so downstream parsing isn't distracted by it
function stripWakeWord(transcript: string): string {
  let t = transcript;
  for (const w of WAKE_WORDS) {
    t = t.split(w).join(' ');
  }
  return t.replace(/\s+/g, ' ').trim();
}

// ─── Command parser ───────────────────────────────────────────────────────────

export type ParsedCommand =
  | { type: 'describe' }
  | { type: 'read' }
  | { type: 'search'; query: string }
  | { type: 'none' };

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
  const cleaned = stripWakeWord(transcript.toLowerCase().trim());

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
  enabled: boolean;
  isAssistantSpeaking: boolean;
  isVLMBusy: boolean;
  onDescribe: () => void;
  onRead: () => void;
  onSearch: (query: string) => void;
  onHeard: (transcript: string) => void;
  playBeep: () => void;
  speakQuick: (text: string) => void;
}

export function useVoiceRecognition({
  enabled,
  isAssistantSpeaking,
  isVLMBusy,
  onDescribe,
  onRead,
  onSearch,
  onHeard,
  playBeep,
  speakQuick,
}: UseVoiceRecognitionOptions) {
  const [isListening, setIsListening] = useState(false);
  const awakeUntilRef = useRef<number>(0);

  // Ref to the live recognition object so effects outside the setup effect can stop/start it
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  // When true, recognition.onend should NOT auto-restart — the caller intentionally paused it
  const intentionalStopRef = useRef(false);

  const stateRef = useRef({ isVLMBusy, onDescribe, onRead, onSearch, onHeard, playBeep, speakQuick });
  useEffect(() => {
    stateRef.current = { isVLMBusy, onDescribe, onRead, onSearch, onHeard, playBeep, speakQuick };
  }, [isVLMBusy, onDescribe, onRead, onSearch, onHeard, playBeep, speakQuick]);

  // Hard-mute: stop recognition while the assistant is speaking; restart when it finishes
  useEffect(() => {
    const rec = recognitionRef.current;
    if (!rec) return;
    if (isAssistantSpeaking) {
      intentionalStopRef.current = true;
      try { rec.stop(); } catch { /* already stopped */ }
    } else if (intentionalStopRef.current) {
      intentionalStopRef.current = false;
      // Small delay lets the engine flush any buffered frames from the TTS tail
      const t = setTimeout(() => {
        try { rec.start(); } catch { /* already running */ }
      }, 300);
      return () => clearTimeout(t);
    }
  }, [isAssistantSpeaking]);

  useEffect(() => {
    if (!enabled) return;

    const w = window as WindowWithSpeech;
    const SpeechRecognition = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn('[voice] SpeechRecognition not supported in this browser');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognitionRef.current = recognition;

    recognition.onstart = () => {
      console.info('[voice] recognition started');
      setIsListening(true);
    };

    recognition.onerror = (event: SpeechRecognitionErrorEventLike) => {
      console.warn('[voice] recognition error:', event.error);
    };

    recognition.onend = () => {
      setIsListening(false);
      if (intentionalStopRef.current) {
        console.info('[voice] recognition ended (intentional pause)');
        return;
      }
      try {
        recognition.start();
      } catch { /* already running */ }
    };

    recognition.onresult = (event: SpeechRecognitionEventLike) => {
      const transcript: string =
        event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
      const s = stateRef.current;

      const woke = hasWakeWord(transcript);
      const awake = Date.now() < awakeUntilRef.current;
      const parsed = parseVoiceCommand(transcript);

      console.info('[voice] heard:', JSON.stringify(transcript), { woke, awake, parsed });

      if (!woke && !awake) return;

      if (parsed.type !== 'none') {
        if (s.isVLMBusy) {
          s.playBeep();
          s.speakQuick('Still analysing, one moment.');
        } else {
          s.playBeep();
          s.speakQuick('Got it');
          if (parsed.type === 'describe') s.onDescribe();
          else if (parsed.type === 'read')   s.onRead();
          else if (parsed.type === 'search') s.onSearch(parsed.query);
        }
        awakeUntilRef.current = 0;
      } else if (woke) {
        s.playBeep();
        s.speakQuick("I'm listening. Say describe, read, or find something.");
        s.onHeard(transcript);
        awakeUntilRef.current = Date.now() + 8000;
      } else {
        s.onHeard(transcript);
      }
    };

    intentionalStopRef.current = false;
    try { recognition.start(); } catch { /* ignore */ }

    return () => {
      recognition.onend = null;
      recognition.onerror = null;
      recognition.onresult = null;
      try { recognition.stop(); } catch { /* ignore */ }
      recognitionRef.current = null;
      setIsListening(false);
    };
  }, [enabled]);

  return { isListening };
}
