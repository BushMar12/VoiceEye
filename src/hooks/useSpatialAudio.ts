import { useCallback, useRef } from 'react';
import {
  BEEP_FREQUENCY_HZ,
  BEEP_DURATION_S,
  BEEP_GAIN,
  QUICK_TTS_RATE,
  QUICK_TTS_VOLUME,
} from '../config';

/**
 * Provides TTS, beep, and haptic feedback primitives.
 * All audio functions are stable refs — safe for use in rAF loops and event listeners.
 */
export function useSpatialAudio() {
  const audioCtxRef = useRef<AudioContext | null>(null);

  const speak = useCallback((text: string, onEnd?: () => void, rate = 1.0) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = rate;
      u.pitch = 1.0;
      if (onEnd) u.onend = onEnd;
      window.speechSynthesis.speak(u);
    }
  }, []);

  const speakQuick = useCallback((text: string) => {
    if ('speechSynthesis' in window) {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = QUICK_TTS_RATE;
      u.volume = QUICK_TTS_VOLUME;
      window.speechSynthesis.speak(u);
    }
  }, []);

  const playBeep = useCallback(() => {
    try {
      if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = BEEP_FREQUENCY_HZ;
      gain.gain.value = BEEP_GAIN;
      osc.start();
      osc.stop(ctx.currentTime + BEEP_DURATION_S);
    } catch { /* AudioContext unavailable */ }
  }, []);

  return { speak, speakQuick, playBeep };
}
