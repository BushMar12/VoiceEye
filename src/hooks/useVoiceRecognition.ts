import { useState, useEffect, useRef } from 'react';

interface UseVoiceRecognitionOptions {
  onDescribe: () => void;
  onRead: () => void;
  onSearch: (query: string) => void;
  playBeep: () => void;
  speakQuick: (text: string) => void;
}

/**
 * Continuous voice recognition with "Voice Eye" wake word detection.
 * Parses commands: describe (default), read, find/search <query>.
 */
export function useVoiceRecognition({
  onDescribe,
  onRead,
  onSearch,
  playBeep,
  speakQuick,
}: UseVoiceRecognitionOptions) {
  const [isListening, setIsListening] = useState(false);

  // Keep callbacks current without restarting recognition
  const callbacksRef = useRef({ onDescribe, onRead, onSearch, playBeep, speakQuick });
  useEffect(() => {
    callbacksRef.current = { onDescribe, onRead, onSearch, playBeep, speakQuick };
  }, [onDescribe, onRead, onSearch, playBeep, speakQuick]);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => {
      setIsListening(false);
      try { recognition.start(); } catch { }
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase();
      const cb = callbacksRef.current;

      if (transcript.includes('voice eye') || transcript.includes('voice i') || transcript.includes('boy i')) {
        cb.playBeep();
        cb.speakQuick('Got it');

        if (transcript.includes('read')) {
          cb.onRead();
        } else if (transcript.includes('find') || transcript.includes('search')) {
          const words = transcript.split(' ');
          const findIndex = words.findIndex((w: string) => w === 'find' || w === 'search');
          const query = words.slice(findIndex + 1).join(' ') || 'object';
          cb.onSearch(query);
        } else {
          cb.onDescribe();
        }
      }
    };

    try { recognition.start(); } catch { }

    return () => {
      recognition.onend = null;
      recognition.stop();
    };
  }, []);

  return { isListening };
}
