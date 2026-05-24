import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Settings, Eye, Zap, Image as ImageIcon } from 'lucide-react';
import CameraView from './components/CameraView';
import SettingsPanel, { type AppSettings, loadSettings } from './components/SettingsPanel';
import { estimateDistance } from './utils/distance';
import { useSpatialAudio } from './hooks/useSpatialAudio';
import { useVLMEngine } from './hooks/useVLMEngine';
import { useVoiceRecognition } from './hooks/useVoiceRecognition';
import { useDetectionLoop } from './hooks/useDetectionLoop';
import { useHoldToTalk } from './hooks/useHoldToTalk';
import {
  FULL_INTRO_MESSAGE,
  SHORT_INTRO_MESSAGE,
  BBOX_MIN_SCORE,
  VLM_DISPLAY_LABEL,
  COMMAND_WINDOW_MESSAGE,
  COMMAND_CLOSED_TONE_HZ,
  COMMAND_CLOSED_TONE_S,
} from './config';
import './index.css';

const HEARD_FLASH_MS = 2000;

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

  // Audio primitives — isSpeaking is driven by SpeechSynthesisUtterance lifecycle events
  const { unlockAudio, speak, speakQuick, playBeep } = useSpatialAudio();

  // VLM (Slow Lane)
  const vlm = useVLMEngine({
    videoElement,
    speak,
    ttsRate: settings.ttsRate,
    defaultMessage,
    setLatestMessage,
  });

  // Keep a stable ref for voice + gesture callbacks
  const vlmTriggerRef = useRef(vlm.trigger);
  useEffect(() => { vlmTriggerRef.current = vlm.trigger; }, [vlm.trigger]);

  // Flash the UI briefly with what the speech engine actually heard
  const heardFlashTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onHeard = useCallback((transcript: string) => {
    setLatestMessage(`Heard: "${transcript}"`);
    if (heardFlashTimer.current) clearTimeout(heardFlashTimer.current);
    heardFlashTimer.current = setTimeout(
      () => setLatestMessage(defaultMessage),
      HEARD_FLASH_MS,
    );
  }, [defaultMessage]);

  // ── Hold-to-talk gesture ↔ on-demand voice recognition bridge ──────────
  //
  // The two hooks are mutually recursive (the gesture activates the mic;
  // the mic, on result OR timeout, closes the gesture's visible window).
  // We resolve the cycle by declaring the bridge callbacks first, then
  // wiring them via refs that are filled in once each hook returns.

  const voiceActivateRef = useRef<() => void>(() => {});
  const voiceDeactivateRef = useRef<() => void>(() => {});
  const closeCommandWindowRef = useRef<() => void>(() => {});

  // Called by the gesture hook at 2 s lock-in.
  //
  // Three-stage cue sequence so voice activation is discoverable for
  // visually impaired users without re-introducing the mic-bleed problem
  // the original wake word had:
  //   1. Instant 880 Hz beep so the user hears "hold completed" the
  //      moment the threshold is crossed (no TTS startup delay).
  //   2. Quick spoken "Listening" prompt.
  //   3. Recognition.start() chained onto the utterance's onend so the
  //      device speaker is silent before the mic opens.
  const handleActivate = useCallback(() => {
    playBeep();
    speakQuick('Listening', () => voiceActivateRef.current());
  }, [playBeep, speakQuick]);

  // Called by the gesture hook when its 8 s window times out without a
  // result. Stop the mic so we don't leak the session — onResolved(false)
  // will then fire the "Cancelled" cue.
  const handleDeactivate = useCallback(() => {
    voiceDeactivateRef.current();
  }, []);

  // Called by SpeechRecognition when a result fires (matched=true) OR when
  // recognition ends without dispatching anything (matched=false). Either
  // way, close the visible "Listening…" window so the user doesn't wait
  // for the full 8 s. When unmatched, also play a distinct closed-tone
  // earcon + spoken "Cancelled" so the user knows the window has shut and
  // can press-and-hold again.
  const handleResolved = useCallback((matched: boolean) => {
    closeCommandWindowRef.current();
    if (!matched) {
      playBeep(COMMAND_CLOSED_TONE_HZ, COMMAND_CLOSED_TONE_S);
      speakQuick('Cancelled');
    }
  }, [playBeep, speakQuick]);

  // The hold-to-talk gesture itself. onTap fires the quick describe (same
  // as the previous global tap behaviour). onActivate opens the mic.
  const hold = useHoldToTalk({
    onTap: useCallback(() => {
      unlockAudio();
      vlmTriggerRef.current();
    }, [unlockAudio]),
    onActivate: handleActivate,
    onDeactivate: handleDeactivate,
    unlockAudio,
    disabled: showSettings,
  });

  // Mirror the close callback into the ref now that the hook has returned.
  useEffect(() => {
    closeCommandWindowRef.current = hold.closeCommandWindow;
  }, [hold.closeCommandWindow]);

  // Voice recognition (on-demand)
  const voice = useVoiceRecognition({
    enabled: !!videoElement,
    isVLMBusy: vlm.isProcessing || vlm.isSpeaking,
    onDescribe: useCallback(() => {
      vlmTriggerRef.current('Describe', '');
    }, []),
    onRead: useCallback(() => {
      vlmTriggerRef.current('Read', '');
    }, []),
    onSearch: useCallback((query: string) => {
      vlmTriggerRef.current('Search', query);
    }, []),
    onHeard,
    onResolved: handleResolved,
    playBeep,
    speakQuick,
  });

  // Mirror the voice activate/deactivate methods into refs the gesture
  // callbacks already reference.
  useEffect(() => {
    voiceActivateRef.current = voice.activate;
    voiceDeactivateRef.current = voice.deactivate;
  }, [voice.activate, voice.deactivate]);

  // ── Unsupported-browser fallback ────────────────────────────────────────
  // The press-and-hold gesture still works for tap (Describe) even when
  // SpeechRecognition is missing — we just tell the user once.
  const voiceFallbackSpokenRef = useRef(false);
  useEffect(() => {
    if (voice.isSupported) return;
    if (!videoElement) return;
    if (voiceFallbackSpokenRef.current) return;
    voiceFallbackSpokenRef.current = true;
    const msg = 'Voice commands are not supported in this browser. Tap the screen to describe what is around you.';
    setLatestMessage(msg);
    speak(msg, undefined, settings.ttsRate);
  }, [voice.isSupported, videoElement, speak, settings.ttsRate]);

  // ── Visible "Listening…" message while the command window is open ──────
  // Track the previous awake state so we can restore the default message
  // when the window closes (without clobbering an in-flight VLM message).
  const wasAwakeRef = useRef(false);
  useEffect(() => {
    if (hold.isAwaitingCommand && !wasAwakeRef.current) {
      setLatestMessage(COMMAND_WINDOW_MESSAGE);
    } else if (!hold.isAwaitingCommand && wasAwakeRef.current) {
      // Only revert if we're still showing the listening message — don't
      // clobber a VLM response that landed during the window.
      setLatestMessage(prev => prev === COMMAND_WINDOW_MESSAGE ? defaultMessage : prev);
    }
    wasAwakeRef.current = hold.isAwaitingCommand;
  }, [hold.isAwaitingCommand, defaultMessage]);

  // Fast Lane — detection + tracking. Pass isAwaitingCommand so the loop
  // can mute per-frame announcements while the user is mid-command.
  const { renderedTracks } = useDetectionLoop({
    videoElement,
    isProcessingSlowLane: vlm.isProcessing || vlm.isSpeaking,
    isCommandWindowOpen: hold.isAwaitingCommand,
    settings,
    speak,
    playBeep,
  });

  // Camera error handler
  const handleCameraError = useCallback((msg: string) => {
    setLatestMessage(msg);
    speak(msg, undefined, settings.ttsRate);
  }, [speak, settings.ttsRate]);

  // ── Hold gesture style — drives the conic-gradient progress ring ──────
  // Updated every animation frame inside useHoldToTalk via setHoldProgress.
  const holdStyle = {
    ['--hold-progress' as string]: hold.holdProgress.toFixed(3),
  } as React.CSSProperties;

  // Render
  return (
    <div
      className={`app-container ${hold.isAwaitingCommand ? 'awaiting-command' : ''}`}
      style={holdStyle}
      {...hold.bindHandlers}
    >
      <CameraView
        onVideoReady={setVideoElement}
        onError={handleCameraError}
        isProcessing={vlm.isProcessing}
      />

      {/* Press-and-hold progress ring overlay — visible only when the user
          is mid-hold, and only after a tap-grace period (handled in CSS). */}
      <div
        className="hold-progress-ring"
        aria-hidden="true"
        data-active={hold.holdProgress > 0 ? 'true' : 'false'}
      >
        <div className="hold-progress-ring-inner">
          <span className="hold-progress-ring-label">Hold to talk</span>
        </div>
      </div>

      <div className="ui-layer">

        {/* Header */}
        <header className="app-header">
          <h1 className="logo">
            <Eye className="text-white" size={28} aria-hidden="true" />
            VoiceEye
          </h1>

          <div
            className="status-badge"
            style={{ display: 'flex', gap: '0.5rem' }}
            role="status"
            aria-live="polite"
          >
            {hold.isAwaitingCommand && (
              <span style={{ fontSize: '0.7rem', color: '#10b981' }}>
                <span aria-hidden="true">🎙️ </span>Listening
              </span>
            )}
            <div className={`status-dot ${videoElement ? '' : 'hidden'}`} aria-hidden="true"></div>
            {videoElement ? vlm.vlmMode : 'Loading...'}
          </div>

          <button
            className="glass-button"
            style={{ width: '44px', height: '44px' }}
            // stopPropagation keeps the Settings tap from also firing the
            // hold gesture's tap handler.
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => { e.stopPropagation(); setShowSettings(s => !s); }}
            aria-label="Settings"
            aria-haspopup="dialog"
            aria-expanded={showSettings}
          >
            <Settings size={20} aria-hidden="true" />
          </button>
        </header>

        {/* Real-time Bounding Boxes */}
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
          {!vlm.isProcessing && videoElement &&
            renderedTracks
              .filter(t => t.ageMs === 0 && t.status === 'confirmed' && t.score > BBOX_MIN_SCORE)
              .map(track => {
                const left   = (track.bbox[0] / videoElement.videoWidth)  * 100;
                const top    = (track.bbox[1] / videoElement.videoHeight) * 100;
                const width  = (track.bbox[2] / videoElement.videoWidth)  * 100;
                const height = (track.bbox[3] / videoElement.videoHeight) * 100;
                const dist   = estimateDistance(track.bbox[3], videoElement.videoHeight, track.class, { verticalFovDeg: settings.cameraVfovDeg });

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
                <Zap size={14} aria-hidden="true" />
                YOLO: Fast Lane
              </div>
              <div className={`lane-badge ${vlm.isProcessing ? 'active' : ''}`}>
                <ImageIcon size={14} aria-hidden="true" />
                {VLM_DISPLAY_LABEL}
              </div>
            </div>

            {voice.isSupported ? (
              <p className="voice-hint" aria-hidden="true">
                <strong>Tap</strong> to describe &middot; <strong>Press and hold</strong> for: describe &middot; read &middot; find &lt;object&gt;
              </p>
            ) : (
              <p className="voice-hint" aria-hidden="true">
                Voice commands unavailable here — <strong>tap anywhere to describe the scene</strong>.
              </p>
            )}
          </div>

          <div className="trigger-button-container">
            <div className={`trigger-radar ${vlm.isProcessing ? 'scanning' : ''}`} aria-hidden="true"></div>
            <button
              className={`trigger-button ${vlm.isProcessing ? 'listening' : ''}`}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => { e.stopPropagation(); unlockAudio(); vlm.trigger(); }}
              aria-label={vlm.isProcessing ? 'Analyzing scene, please wait' : 'Describe scene'}
              aria-busy={vlm.isProcessing}
              disabled={vlm.isProcessing}
            >
              <Eye size={36} aria-hidden="true" />
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
