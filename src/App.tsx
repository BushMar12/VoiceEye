import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Settings, Eye, Zap, Image as ImageIcon } from 'lucide-react';
import CameraView from './components/CameraView';
import SettingsPanel, { type AppSettings, loadSettings } from './components/SettingsPanel';
import { estimateDistance } from './utils/distance';
import { useSpatialAudio } from './hooks/useSpatialAudio';
import { useVLMEngine } from './hooks/useVLMEngine';
import { useVoiceRecognition } from './hooks/useVoiceRecognition';
import { useDetectionLoop } from './hooks/useDetectionLoop';
import { FULL_INTRO_MESSAGE, SHORT_INTRO_MESSAGE, BBOX_MIN_SCORE } from './config';
import './index.css';

const App: React.FC = () => {
  // First-launch detection
  const hasLaunchedBefore = useRef(localStorage.getItem('voiceeye_launched') === 'true');
  const defaultMessage = hasLaunchedBefore.current ? SHORT_INTRO_MESSAGE : FULL_INTRO_MESSAGE;

  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [latestMessage, setLatestMessage] = useState(defaultMessage);

  // Settings
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);

  // Mark first launch
  useEffect(() => { localStorage.setItem('voiceeye_launched', 'true'); }, []);

  // Audio primitives
  const { speak, speakQuick, playBeep } = useSpatialAudio();

  // VLM (Slow Lane)
  const vlm = useVLMEngine({
    videoElement,
    speak,
    ttsRate: settings.ttsRate,
    defaultMessage,
    setLatestMessage,
  });

  // Keep a stable ref for voice recognition callbacks
  const vlmTriggerRef = useRef(vlm.trigger);
  useEffect(() => { vlmTriggerRef.current = vlm.trigger; }, [vlm.trigger]);

  // Voice recognition
  const { isListening } = useVoiceRecognition({
    onDescribe: useCallback(() => {
      vlmTriggerRef.current('Describe', '');
    }, []),
    onRead: useCallback(() => {
      vlmTriggerRef.current('Read', '');
    }, []),
    onSearch: useCallback((query: string) => {
      vlmTriggerRef.current('Search', query);
    }, []),
    playBeep,
    speakQuick,
  });

  // Fast Lane — detection + tracking
  const { renderedTracks } = useDetectionLoop({
    videoElement,
    isProcessingSlowLane: vlm.isProcessing,
    settings,
    speak,
  });

  // Camera error handler
  const handleCameraError = useCallback((msg: string) => {
    setLatestMessage(msg);
    speak(msg, undefined, settings.ttsRate);
  }, [speak, settings.ttsRate]);

  // Render
  return (
    <div className="app-container" onClick={() => vlm.trigger()}>
      <CameraView
        onVideoReady={setVideoElement}
        onError={handleCameraError}
        isProcessing={vlm.isProcessing}
      />

      <div className="ui-layer">

        {/* Header */}
        <header className="app-header">
          <div className="logo">
            <Eye className="text-white" size={28} />
            VoiceEye
          </div>

          <div className="status-badge" style={{ display: 'flex', gap: '0.5rem' }}>
            {isListening && (
              <span style={{ fontSize: '0.7rem', color: '#10b981' }}>🎙️ Listening</span>
            )}
            <div className={`status-dot ${videoElement ? '' : 'hidden'}`}></div>
            {videoElement ? vlm.vlmMode : 'Loading...'}
          </div>

          <button
            className="glass-button"
            style={{ width: '44px', height: '44px' }}
            onClick={(e) => { e.stopPropagation(); setShowSettings(s => !s); }}
            aria-label="Settings"
          >
            <Settings size={20} />
          </button>
        </header>

        {/* Real-time Bounding Boxes */}
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
          {!vlm.isProcessing && videoElement &&
            renderedTracks
              .filter(t => t.age === 0 && t.score > BBOX_MIN_SCORE)
              .map(track => {
                const left   = (track.bbox[0] / videoElement.videoWidth)  * 100;
                const top    = (track.bbox[1] / videoElement.videoHeight) * 100;
                const width  = (track.bbox[2] / videoElement.videoWidth)  * 100;
                const height = (track.bbox[3] / videoElement.videoHeight) * 100;
                const dist   = estimateDistance(track.bbox[3], videoElement.videoHeight, track.class);

                const zone = track.proximityZone;
                const boxColor = zone === 'danger' ? '#ef4444'
                               : zone === 'near'   ? '#f59e0b'
                               : '#10b981';

                return (
                  <div
                    key={track.id}
                    className="bounding-box"
                    style={{
                      left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%`,
                      borderColor: boxColor,
                    }}
                  >
                    <div className="bounding-label" style={{ backgroundColor: boxColor }}>
                      {track.class} #{track.id} {dist}
                    </div>
                  </div>
                );
              })
          }
        </div>

        {/* Bottom Interaction Area */}
        <div className="interaction-area">
          <div className="glass-panel message-box">
            <h2 className="message-text" aria-live="polite">
              {latestMessage}
            </h2>

            <div className="lane-indicators">
              <div className="lane-badge active">
                <Zap size={14} />
                YOLO: Fast Lane
              </div>
              <div className={`lane-badge ${vlm.isProcessing ? 'active' : ''}`}>
                <ImageIcon size={14} />
                Qwen: Deep Context
              </div>
            </div>
          </div>

          <div className="trigger-button-container">
            <div className={`trigger-radar ${vlm.isProcessing ? 'scanning' : ''}`}></div>
            <button
              className={`trigger-button ${vlm.isProcessing ? 'listening' : ''}`}
              onClick={(e) => { e.stopPropagation(); vlm.trigger(); }}
              aria-label="Describe scene"
            >
              <Eye size={36} />
            </button>
          </div>
        </div>

      </div>

      {showSettings && (
        <SettingsPanel
          settings={settings}
          onChange={setSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
};

export default App;
