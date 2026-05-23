import { useState, useEffect, useRef } from 'react';
import type { InferenceSession } from 'onnxruntime-web';
import { loadYoloModel, runYolo } from '../utils/yolo';
import { type Track, updateTracks } from '../utils/tracker';
import { estimateDistanceForSpeech } from '../utils/distance';
import { inferenceMetrics } from '../utils/inferenceMetrics';
import { createAttentionState, runAttention, type AttentionState, type Announcement } from '../utils/attention';
import type { AppSettings } from '../components/SettingsPanel';
import {
  INFERENCE_INTERVAL_MS,
  HAPTIC_TRIPLE_PULSE_THRESHOLD,
  HAPTIC_SINGLE_PULSE_THRESHOLD,
  YOLO_MODEL_PATH,
  DEESCALATION_TONE_HZ,
  DEESCALATION_TONE_S,
} from '../config';

interface UseDetectionLoopOptions {
  videoElement: HTMLVideoElement | null;
  isProcessingSlowLane: boolean;
  settings: AppSettings;
  speak: (text: string, onEnd?: () => void, rate?: number) => void;
  playBeep: (freq?: number, durationS?: number) => void;
}

function formatAnnouncement(ann: Announcement, frameHeight: number, verticalFovDeg: number): string {
  const dist = estimateDistanceForSpeech(ann.bbox[3], frameHeight, ann.class, { verticalFovDeg });
  const distSuffix = dist ? `, ${dist}` : '';

  if (ann.kind === 'group') {
    return `${ann.memberCount} ${ann.class}s${distSuffix}`;
  }

  switch (ann.reason) {
    case 'approaching':      return `Warning: ${ann.class} approaching${distSuffix}`;
    case 'zone-escalation':  return `Close: ${ann.class}${distSuffix}`;
    case 'sustained':        return `Still close: ${ann.class}`;
    default:                 return `${ann.class}${distSuffix}`;
  }
}

export function useDetectionLoop({
  videoElement,
  isProcessingSlowLane,
  settings,
  speak,
  playBeep,
}: UseDetectionLoopOptions) {
  const [model, setModel] = useState<InferenceSession | null>(null);
  const [renderedTracks, setRenderedTracks] = useState<Track[]>([]);
  const tracksRef = useRef<Track[]>([]);
  const attentionStateRef = useRef<AttentionState>(createAttentionState(performance.now()));
  const settingsRef = useRef(settings);
  useEffect(() => { settingsRef.current = settings; }, [settings]);

  useEffect(() => {
    loadYoloModel(YOLO_MODEL_PATH)
      .then(session => setModel(session))
      .catch(err => console.error('Failed to load YOLO model', err));
  }, []);

  useEffect(() => {
    let animationFrameId: number;
    let isDetecting = false;
    let lastInferenceTime = 0;

    const runDetection = async () => {
      // Pause entirely when the tab is backgrounded — Safari keeps charging memory
      // even for throttled rAF calls, which can trigger an OOM kill on return.
      if (document.hidden) {
        animationFrameId = requestAnimationFrame(runDetection);
        return;
      }

      if (videoElement && model && !isProcessingSlowLane) {
        const now = performance.now();

        if (!isDetecting && now - lastInferenceTime >= INFERENCE_INTERVAL_MS) {
          // Real elapsed time since the last inference — used by the tracker for
          // velocity-based bbox prediction and fps-invariant track aging.
          const dtMs = lastInferenceTime > 0 ? now - lastInferenceTime : INFERENCE_INTERVAL_MS;
          isDetecting = true;
          lastInferenceTime = now;

          try {
            const t0 = performance.now();
            const detections = await runYolo(model, videoElement, settingsRef.current.confThreshold);
            inferenceMetrics.recordInference(performance.now() - t0, detections);

            const vw = videoElement.videoWidth;
            const vh = videoElement.videoHeight;
            const screenArea = vw * vh;

            tracksRef.current = updateTracks(tracksRef.current, detections, screenArea, dtMs);

            const out = runAttention(tracksRef.current, now, attentionStateRef.current, {
              verbosity: settingsRef.current.verbosity,
              frameWidth: vw,
              frameHeight: vh,
              screenArea,
            });

            inferenceMetrics.recordAttention(
              out.toAnnounce.length,
              out.suppressedCount,
              attentionStateRef.current.clusters.size,
            );

            setRenderedTracks(out.renderTracks);

            // Speak announcements — batched into one utterance per frame because
            // useSpatialAudio.speak() cancels any in-flight speech, which would
            // otherwise collapse the budget (K=3 in Normal) down to "last only".
            if (out.toAnnounce.length > 0) {
              const phrase = out.toAnnounce
                .map(ann => formatAnnouncement(ann, vh, settingsRef.current.cameraVfovDeg))
                .join('. ');
              speak(phrase, undefined, settingsRef.current.ttsRate);
            }

            // De-escalation tones — one beep per de-escalating track
            for (let i = 0; i < out.deescalationTones.length; i++) {
              playBeep(DEESCALATION_TONE_HZ, DEESCALATION_TONE_S);
            }

            // Haptics — one pulse per danger/near announcement
            if (settingsRef.current.hapticEnabled) {
              for (const ann of out.toAnnounce) {
                const areaPercent = (ann.bbox[2] * ann.bbox[3]) / screenArea;
                if (ann.reason === 'approaching') {
                  navigator.vibrate?.([50, 30, 50, 30, 50]);
                } else if (areaPercent > HAPTIC_TRIPLE_PULSE_THRESHOLD) {
                  navigator.vibrate?.([50, 50, 50]);
                } else if (areaPercent > HAPTIC_SINGLE_PULSE_THRESHOLD) {
                  navigator.vibrate?.([100]);
                }
              }
            }
          } catch (e) {
            console.error('Detection error:', e);
          }
          isDetecting = false;
        }
      } else {
        tracksRef.current = [];
        setRenderedTracks([]);
      }
      animationFrameId = requestAnimationFrame(runDetection);
    };

    if (videoElement && model) {
      runDetection();
    }

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [videoElement, model, isProcessingSlowLane, speak, playBeep]);

  return { model, renderedTracks };
}
