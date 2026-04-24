import { useCallback, useRef, useState } from 'react';
import {
  BEEP_FREQUENCY_HZ,
  BEEP_DURATION_S,
  BEEP_GAIN,
  QUICK_TTS_RATE,
  QUICK_TTS_VOLUME,
} from '../config';

export function useSpatialAudio() {
  const audioCtxRef = useRef<AudioContext | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const activeUtterancesRef = useRef(0);

  const markStart = useCallback(() => {
    activeUtterancesRef.current += 1;
    setIsSpeaking(true);
  }, []);

  const markEnd = useCallback(() => {
    activeUtterancesRef.current = Math.max(0, activeUtterancesRef.current - 1);
    if (activeUtterancesRef.current === 0) setIsSpeaking(false);
  }, []);

  const unlockAudio = useCallback(() => {
    try {
      if (!audioCtxRef.current) audioCtxRef.current = new window.AudioContext();
      if (audioCtxRef.current.state === 'suspended') {
        audioCtxRef.current.resume();
      }
      if ('speechSynthesis' in window) {
        const u = new SpeechSynthesisUtterance('');
        u.volume = 0;
        window.speechSynthesis.speak(u);
      }
    } catch { /* ignore */ }
  }, []);

  const speak = useCallback((text: string, onEnd?: () => void, rate = 1.0) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    activeUtterancesRef.current = 0;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = rate;
    u.pitch = 1.0;
    u.onstart = () => markStart();
    u.onend = () => { markEnd(); onEnd?.(); };
    u.onerror = () => { markEnd(); onEnd?.(); };
    window.speechSynthesis.speak(u);
  }, [markStart, markEnd]);

  const speakQuick = useCallback((text: string, onEnd?: () => void) => {
    if (!('speechSynthesis' in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = QUICK_TTS_RATE;
    u.volume = QUICK_TTS_VOLUME;
    u.onstart = () => markStart();
    u.onend = () => { markEnd(); onEnd?.(); };
    u.onerror = () => { markEnd(); onEnd?.(); };
    window.speechSynthesis.speak(u);
  }, [markStart, markEnd]);

  const playBeep = useCallback((freq = BEEP_FREQUENCY_HZ, durationS = BEEP_DURATION_S) => {
    try {
      if (!audioCtxRef.current) audioCtxRef.current = new window.AudioContext();
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        ctx.resume();
      }
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = freq;
      gain.gain.value = BEEP_GAIN;
      osc.start();
      osc.stop(ctx.currentTime + durationS);
    } catch { /* AudioContext unavailable */ }
  }, []);

  return { unlockAudio, speak, speakQuick, playBeep, isSpeaking };
}
