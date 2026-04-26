import { useState, useCallback, useRef } from 'react';
import {
  VLM_TIMEOUT_MS,
  VLM_ENDPOINT,
  VLM_MODEL,
  VLM_UNAVAILABLE_MESSAGE,
  VLM_IMAGE_QUALITY,
  VLM_IMAGE_MAX_EDGE,
  VLM_MAX_TOKENS,
  VLM_MAX_WORDS,
} from '../config';

export type VLMMode = 'Describe' | 'Read' | 'Search';

async function processWithVLM(
  videoElement: HTMLVideoElement,
  mode: string,
  query: string,
  signal: AbortSignal,
): Promise<string> {
  const sourceWidth = videoElement.videoWidth || 640;
  const sourceHeight = videoElement.videoHeight || 480;
  const scale = VLM_IMAGE_MAX_EDGE > 0
    ? Math.min(1, VLM_IMAGE_MAX_EDGE / Math.max(sourceWidth, sourceHeight))
    : 1;
  const targetWidth = Math.max(1, Math.round(sourceWidth * scale));
  const targetHeight = Math.max(1, Math.round(sourceHeight * scale));
  const startedAt = performance.now();

  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return 'Failed to process image.';

  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  const base64Image = canvas.toDataURL('image/jpeg', VLM_IMAGE_QUALITY).split(',')[1];

  const responseLimit = `Answer in ${VLM_MAX_WORDS} words or fewer. Use complete sentences.`;
  let dynamicPrompt = `You are an AI assistant for a visually impaired person. Briefly but accurately describe the scene in front of them. ${responseLimit}`;
  if (mode === 'Read') {
    dynamicPrompt = `Extract and read aloud the most important visible text in this image. Do not describe the general scene. ${responseLimit}`;
  } else if (mode === 'Search' && query) {
    dynamicPrompt = `Look carefully at this image. Is there a ${query} in it? If yes, describe exactly where it is. If not, say it is not visible. ${responseLimit}`;
  }

  const response = await fetch(VLM_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: VLM_MODEL,
      prompt: dynamicPrompt,
      stream: false,
      images: [base64Image],
      keep_alive: '30m',
      options: {
        num_predict: VLM_MAX_TOKENS,
        temperature: 0.2,
      },
    }),
    signal,
  });

  if (!response.ok) return `API Error: ${response.statusText}. ${VLM_UNAVAILABLE_MESSAGE}`;
  const data = await response.json();
  console.info('Slow Lane completed', {
    ms: Math.round(performance.now() - startedAt),
    source: `${sourceWidth}x${sourceHeight}`,
    sent: `${targetWidth}x${targetHeight}`,
    imageKB: Math.round((base64Image.length * 3) / 4 / 1024),
  });
  return data.response || 'No description provided.';
}

interface UseVLMEngineOptions {
  videoElement: HTMLVideoElement | null;
  speak: (text: string, onEnd?: () => void, rate?: number) => void;
  ttsRate: number;
  defaultMessage: string;
  setLatestMessage: (msg: string) => void;
}

export function useVLMEngine({
  videoElement,
  speak,
  ttsRate,
  defaultMessage,
  setLatestMessage,
}: UseVLMEngineOptions) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [vlmMode, setVlmMode] = useState<VLMMode>('Describe');
  const [searchQuery, setSearchQuery] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  const trigger = useCallback(async (overrideMode?: string, overrideQuery?: string) => {
    if (isProcessing) return;

    abortRef.current?.abort();

    setIsProcessing(true);
    const activeMode = overrideMode || vlmMode;
    const activeQuery = overrideQuery || searchQuery;

    if (activeMode === 'Search') {
      setLatestMessage(`Searching for ${activeQuery}...`);
      speak(`Searching for ${activeQuery}`, undefined, ttsRate);
    } else if (activeMode === 'Read') {
      setLatestMessage('Reading text...');
      speak('Reading text', undefined, ttsRate);
    } else {
      setLatestMessage('Analyzing the scene...');
      speak('Analyzing scene', undefined, ttsRate);
    }

    const controller = new AbortController();
    abortRef.current = controller;
    const timeoutId = setTimeout(() => controller.abort(), VLM_TIMEOUT_MS);

    try {
      if (!videoElement) throw new Error('Video not ready');
      const description = await processWithVLM(videoElement, activeMode, activeQuery, controller.signal);
      setLatestMessage(description);
      
      setIsSpeaking(true);
      const fallbackTime = Math.max(3000, description.length * 100);
      const safetyTimer = setTimeout(() => setIsSpeaking(false), fallbackTime);

      speak(description, () => {
        clearTimeout(safetyTimer);
        setIsSpeaking(false);
        setTimeout(() => setLatestMessage(defaultMessage), 1500);
      }, ttsRate);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        const msg = 'Scene description timed out. Please try again.';
        setLatestMessage(msg);
        speak(msg, undefined, ttsRate);
      } else {
        const isNetworkError = err instanceof TypeError && (err.message.includes('fetch') || err.message.includes('network'));
        const msg = isNetworkError
          ? VLM_UNAVAILABLE_MESSAGE
          : 'Sorry, I encountered an error analyzing the scene.';
        setLatestMessage(msg);
        speak(msg, undefined, ttsRate);
      }
    } finally {
      clearTimeout(timeoutId);
      abortRef.current = null;
      setIsProcessing(false);
    }
  }, [isProcessing, videoElement, vlmMode, searchQuery, defaultMessage, speak, ttsRate, setLatestMessage]);

  return {
    isProcessing,
    isSpeaking,
    vlmMode,
    setVlmMode,
    searchQuery,
    setSearchQuery,
    trigger,
  };
}
