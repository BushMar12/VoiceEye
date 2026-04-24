import React, { useEffect, useState } from 'react';
import { X } from 'lucide-react';

export type Verbosity = 'quiet' | 'normal' | 'detailed';

export interface AppSettings {
  ttsRate: number;
  confThreshold: number;
  hapticEnabled: boolean;
  verbosity: Verbosity;
}

const DEFAULTS: AppSettings = {
  ttsRate: 1.0,
  confThreshold: 0.5,
  hapticEnabled: true,
  verbosity: 'normal',
};

const STORAGE_KEY = 'voiceeye_settings';

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch { /* corrupt data — fall back */ }
  return { ...DEFAULTS };
}

function saveSettings(s: AppSettings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

interface Props {
  settings: AppSettings;
  onChange: (s: AppSettings) => void;
  onClose: () => void;
}

const SettingsPanel: React.FC<Props> = ({ settings, onChange, onClose }) => {
  const [local, setLocal] = useState(settings);

  useEffect(() => { setLocal(settings); }, [settings]);

  const update = (patch: Partial<AppSettings>) => {
    const next = { ...local, ...patch };
    setLocal(next);
    saveSettings(next);
    onChange(next);
  };

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div
        className="glass-panel settings-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        onClick={e => e.stopPropagation()}
      >
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="glass-button" onClick={onClose} aria-label="Close settings" style={{ width: 36, height: 36 }}>
            <X size={18} />
          </button>
        </div>

        {/* TTS Speed */}
        <label className="settings-row">
          <span className="settings-label">Voice Speed</span>
          <div className="settings-control">
            <input
              type="range"
              className="settings-slider"
              min="0.5" max="2" step="0.1"
              value={local.ttsRate}
              onChange={e => update({ ttsRate: parseFloat(e.target.value) })}
              aria-label="Text-to-speech speed"
            />
            <span className="settings-value">{local.ttsRate.toFixed(1)}x</span>
          </div>
        </label>

        {/* Detection Sensitivity */}
        <label className="settings-row">
          <span className="settings-label">Detection Sensitivity</span>
          <div className="settings-control">
            <input
              type="range"
              className="settings-slider"
              min="0.3" max="0.8" step="0.05"
              value={local.confThreshold}
              onChange={e => update({ confThreshold: parseFloat(e.target.value) })}
              aria-label="Object detection confidence threshold"
            />
            <span className="settings-value">{(local.confThreshold * 100).toFixed(0)}%</span>
          </div>
        </label>

        {/* Haptic Toggle */}
        <label className="settings-row">
          <span className="settings-label">Haptic Feedback</span>
          <button
            className={`settings-toggle ${local.hapticEnabled ? 'on' : ''}`}
            onClick={() => update({ hapticEnabled: !local.hapticEnabled })}
            role="switch"
            aria-checked={local.hapticEnabled}
            aria-label="Toggle haptic feedback"
          >
            <span className="settings-toggle-thumb" />
          </button>
        </label>

        {/* Verbosity */}
        <div className="settings-row" role="radiogroup" aria-label="Verbosity level">
          <span className="settings-label">Verbosity</span>
          <div className="settings-control" style={{ display: 'flex', gap: 4 }}>
            {([['quiet', 'Quiet'], ['normal', 'Normal'], ['detailed', 'Detailed']] as const).map(([value, label]) => (
              <button
                key={value}
                role="radio"
                className={`glass-button${local.verbosity === value ? ' active' : ''}`}
                onClick={() => update({ verbosity: value })}
                aria-checked={local.verbosity === value}
                style={{ flex: 1, height: 36, fontSize: 13 }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;
