import { useState, useEffect, useRef } from 'react';
import type { InferenceSession } from 'onnxruntime-web';
import { loadYoloModel, runYolo } from '../utils/yolo';
import { type Track, updateTracks, classifyProximity } from '../utils/tracker';
import { estimateDistance } from '../utils/distance';
import { inferenceMetrics } from '../utils/inferenceMetrics';
import type { AppSettings } from '../components/SettingsPanel';
import {
  INFERENCE_INTERVAL_MS,
  MAX_REANNOUNCE,
  SUSTAINED_ZONE_MS,
  HAZARDOUS_CLASSES,
  HAPTIC_TRIPLE_PULSE_THRESHOLD,
  HAPTIC_SINGLE_PULSE_THRESHOLD,
  APPROACHING_GROWTH_RATE,
  FIRST_ANNOUNCE_MIN_SCORE,
  FIRST_ANNOUNCE_AREA_THRESHOLD,
  YOLO_MODEL_PATH,
} from '../config';

interface UseDetectionLoopOptions {
  videoElement: HTMLVideoElement | null;
  isProcessingSlowLane: boolean;
  settings: AppSettings;
  speak: (text: string, onEnd?: () => void, rate?: number) => void;
}

export function useDetectionLoop({
  videoElement,
  isProcessingSlowLane,
  settings,
  speak,
}: UseDetectionLoopOptions) {
  const [model, setModel] = useState<InferenceSession | null>(null);
  const [renderedTracks, setRenderedTracks] = useState<Track[]>([]);
  const tracksRef = useRef<Track[]>([]);
  const settingsRef = useRef(settings);
  useEffect(() => { settingsRef.current = settings; }, [settings]);

  // Load YOLO model on mount
  useEffect(() => {
    loadYoloModel(YOLO_MODEL_PATH)
      .then(session => setModel(session))
      .catch(err => console.error('Failed to load YOLO model', err));
  }, []);

  // Detection + tracking + alerts loop
  useEffect(() => {
    let animationFrameId: number;
    let isDetecting = false;
    let lastInferenceTime = 0;

    const runDetection = async () => {
      if (videoElement && model && !isProcessingSlowLane) {
        const now = performance.now();

        if (!isDetecting && now - lastInferenceTime >= INFERENCE_INTERVAL_MS) {
          isDetecting = true;
          lastInferenceTime = now;

          try {
            const t0 = performance.now();
            const detections = await runYolo(model, videoElement, settingsRef.current.confThreshold);
            inferenceMetrics.recordInference(performance.now() - t0, detections);
            tracksRef.current = updateTracks(tracksRef.current, detections);
            setRenderedTracks([...tracksRef.current]);

            const vw = videoElement.videoWidth;
            const vh = videoElement.videoHeight;
            const screenArea = vw * vh;

            for (const track of tracksRef.current) {
              if (track.age > 0) continue;

              const areaPercent = (track.bbox[2] * track.bbox[3]) / screenArea;
              const newZone = classifyProximity(areaPercent);
              const oldZone = track.proximityZone;

              // Haptic feedback (continuous, independent of TTS)
              if (settingsRef.current.hapticEnabled) {
                if (areaPercent > HAPTIC_TRIPLE_PULSE_THRESHOLD) {
                  navigator.vibrate?.([50, 50, 50]);
                } else if (areaPercent > HAPTIC_SINGLE_PULSE_THRESHOLD) {
                  navigator.vibrate?.([100]);
                }
              }

              // Approaching alert
              const isApproaching = track.areaGrowthRate > APPROACHING_GROWTH_RATE && HAZARDOUS_CLASSES.has(track.class);
              if (isApproaching && !track.announced) {
                track.announced = true;
                const dist = estimateDistance(track.bbox[3], vh, track.class);
                speak(`Warning: ${track.class} approaching${dist ? `, ${dist}` : ''}`, undefined, settingsRef.current.ttsRate);
                if (settingsRef.current.hapticEnabled) navigator.vibrate?.([50, 30, 50, 30, 50]);
              }

              // First announcement (existing, non-approaching)
              if (!track.announced && track.score > FIRST_ANNOUNCE_MIN_SCORE) {
                const isClose = areaPercent > FIRST_ANNOUNCE_AREA_THRESHOLD;
                const isHazardous = HAZARDOUS_CLASSES.has(track.class);
                if (isClose || isHazardous) {
                  track.announced = true;
                  const dist = estimateDistance(track.bbox[3], vh, track.class);
                  const prefix = isClose ? 'Close ' : '';
                  speak(`${prefix}${track.class}${dist ? `, ${dist}` : ''}`, undefined, settingsRef.current.ttsRate);
                }
              }

              // Zone transition re-announcement
              if (newZone === 'danger' && oldZone !== 'danger' && track.reannounceCount < MAX_REANNOUNCE) {
                track.reannounceCount++;
                track.zoneEntryTime = Date.now();
                const dist = estimateDistance(track.bbox[3], vh, track.class);
                speak(`Close: ${track.class}${dist ? `, ${dist}` : ''}`, undefined, settingsRef.current.ttsRate);
                if (settingsRef.current.hapticEnabled) navigator.vibrate?.([50, 50, 50]);
              }

              // Sustained proximity re-announcement
              if (newZone === 'danger' && oldZone === 'danger'
                  && Date.now() - track.zoneEntryTime > SUSTAINED_ZONE_MS
                  && track.reannounceCount < MAX_REANNOUNCE) {
                track.reannounceCount++;
                track.zoneEntryTime = Date.now();
                speak(`Still close: ${track.class}`, undefined, settingsRef.current.ttsRate);
              }

              track.proximityZone = newZone;
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
  }, [videoElement, model, isProcessingSlowLane, speak]);

  return { model, renderedTracks };
}
