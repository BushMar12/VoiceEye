import { useState, useCallback, useRef } from 'react';
import { VLM_TIMEOUT_MS, VLM_ENDPOINT, VLM_MODEL, VLM_IMAGE_QUALITY } from '../config';

export type VLMMode = 'Describe' | 'Read' | 'Search';

async function processWithVLM(
  videoElement: HTMLVideoElement,
  mode: string,
  query: string,
  signal: AbortSignal,
): Promise<string> {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth || 640;
  canvas.height = videoElement.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  if (!ctx) return 'Failed to process image.';

  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  const base64Image = canvas.toDataURL('image/jpeg', VLM_IMAGE_QUALITY).split(',')[1];

  let dynamicPrompt = 'You are an AI assistant for a visually impaired person. Briefly but accurately describe the scene in front of them.';
  if (mode === 'Read') {
    dynamicPrompt = 'Extract and read aloud all text visible in this image. Do not describe the general scene, focus only on reading text.';
  } else if (mode === 'Search' && query) {
    dynamicPrompt = `Look carefully at this image. Is there a ${query} in it? If yes, describe exactly where it is. If not, say it is not visible.`;
  }

  const response = await fetch(VLM_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: VLM_MODEL, prompt: dynamicPrompt, stream: false, images: [base64Image] }),
    signal,
  });

  if (!response.ok) return `API Error: ${response.statusText}. Please ensure Qwen is running.`;
  const data = await response.json();
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
  const [vlmMode, setVlmMode] = useState<VLMMode>('Describe');
  const [searchQuery, setSearchQuery] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  const trigger = useCallback(async (overrideMode?: string, overrideQuery?: string) => {
    if (isProcessing) return;

    // Abort any lingering request
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
      speak(description, () => {
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
          ? 'Scene description unavailable. The local AI engine is not running.'
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
    vlmMode,
    setVlmMode,
    searchQuery,
    setSearchQuery,
    trigger,
  };
}
