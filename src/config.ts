// ── VoiceEye Configuration ─────────────────────────────────────────────
// Single source of truth for all tunable constants.

// ── Fast Lane (Detection Loop) ─────────────────────────────────────────
export const INFERENCE_INTERVAL_MS = 100;   // ~10 fps target
export const MAX_REANNOUNCE = 3;            // max re-announcements per track
export const SUSTAINED_ZONE_MS = 3000;      // "still close" after 3s in danger zone

// Classes that trigger approaching-hazard alerts
export const HAZARDOUS_CLASSES = new Set([
  'car', 'motorcycle', 'bus', 'train', 'truck', 'bicycle',
  'fire hydrant', 'stop sign', 'bench',
]);

// ── Tracker ────────────────────────────────────────────────────────────
export const TRACKER_MAX_AGE = 10;          // frames without match before track is dropped
export const TRACKER_MIN_IOU = 0.3;         // minimum IoU for detection-to-track matching
export const TRACKER_EMA_ALPHA = 0.3;       // smoothing factor for velocity estimation

// ── Proximity Zones ────────────────────────────────────────────────────
export const PROXIMITY_DANGER_THRESHOLD = 0.5;   // screen area fraction
export const PROXIMITY_NEAR_THRESHOLD = 0.25;    // screen area fraction

// ── Haptic Thresholds (screen area fraction) ───────────────────────────
export const HAPTIC_TRIPLE_PULSE_THRESHOLD = 0.6;
export const HAPTIC_SINGLE_PULSE_THRESHOLD = 0.5;

// ── Approaching Detection ──────────────────────────────────────────────
export const APPROACHING_GROWTH_RATE = 0.05;     // area growth rate threshold
export const FIRST_ANNOUNCE_MIN_SCORE = 0.6;     // min score for first TTS announcement
export const FIRST_ANNOUNCE_AREA_THRESHOLD = 0.5; // min area fraction for non-hazardous announce

// ── YOLO Model ─────────────────────────────────────────────────────────
export const YOLO_INPUT_SIZE = 640;
export const YOLO_DEFAULT_CONF = 0.5;
export const YOLO_IOU_THRESHOLD = 0.45;
export const YOLO_MODEL_PATH = '/models/yolo26n.onnx';

// ── Distance Estimation ────────────────────────────────────────────────
export const DEFAULT_OBJECT_HEIGHT_M = 0.50;
export const CAMERA_VFOV_DEG = 70;

// ── VLM (Slow Lane) ───────────────────────────────────────────────────
export const VLM_TIMEOUT_MS = 15_000;       // abort VLM request after 15s
export const VLM_ENDPOINT = '/api/ollama/api/generate';
export const VLM_MODEL = 'qwen2.5vl';
export const VLM_IMAGE_QUALITY = 0.7;

// ── Camera ─────────────────────────────────────────────────────────────
export const CAMERA_MAX_RETRIES = 3;
export const CAMERA_RETRY_DELAY_MS = 2000;

// ── Audio ──────────────────────────────────────────────────────────────
export const BEEP_FREQUENCY_HZ = 880;
export const BEEP_DURATION_S = 0.1;
export const BEEP_GAIN = 0.3;
export const QUICK_TTS_RATE = 1.3;
export const QUICK_TTS_VOLUME = 0.8;

// ── Bounding Box Rendering ─────────────────────────────────────────────
export const BBOX_MIN_SCORE = 0.5;          // minimum score to render a bounding box

// ── Inference Metrics ──────────────────────────────────────────────────
export const METRICS_BUFFER_SIZE = 100;
export const METRICS_LOG_INTERVAL = 300;    // log every N frames

// ── UI Messages ────────────────────────────────────────────────────────
export const FULL_INTRO_MESSAGE = "Camera ready. Tap the screen or say 'Voice Eye, describe' to read the scene.";
export const SHORT_INTRO_MESSAGE = 'Ready';
