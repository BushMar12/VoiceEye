import { DEFAULT_OBJECT_HEIGHT_M, CAMERA_VFOV_DEG } from '../config';

// Average real-world heights (metres) for common YOLO/VoiceEye classes.
// Used with the pinhole camera model: distance = (real_h * focal_px) / bbox_h_px.
export const KNOWN_HEIGHTS_M: Record<string, number> = {
  person: 1.70,
  bicycle: 1.10,
  car: 1.50,
  motorcycle: 1.10,
  airplane: 4.00,
  bus: 3.20,
  train: 4.00,
  truck: 2.80,
  boat: 1.50,
  'traffic light': 0.80,
  'fire hydrant': 0.70,
  'stop sign': 0.90,
  bench: 0.90,
  bird: 0.20,
  cat: 0.30,
  dog: 0.50,
  horse: 1.60,
  cow: 1.40,
  elephant: 3.00,
  bear: 1.20,
  chair: 0.90,
  couch: 0.90,
  bed: 0.60,
  'dining table': 0.75,
  toilet: 0.80,
  tv: 0.60,
  laptop: 0.30,
  bottle: 0.25,
  cup: 0.12,

  // VoiceEye Fast Lane accessibility classes.
  stairs: 0.18,       // one riser; approximate, scene dependent
  curb: 0.15,
  crosswalk: 0.10,    // markings are flat; height estimate is intentionally conservative
  pole: 2.00,
  bollard: 0.90,
  door: 2.00,
};

export interface DistanceEstimateOptions {
  /**
   * Override vertical field of view when the test device is known/calibrated.
   * Falls back to CAMERA_VFOV_DEG.
   */
  verticalFovDeg?: number;

  /**
   * Override the real-world object height. Use this for real-world validation
   * when the measured test target height is known.
   */
  objectHeightM?: number;

  /**
   * Optional per-class height overrides, useful for test scenes with measured
   * objects or device-specific validation fixtures.
   */
  classHeightsM?: Record<string, number>;

  /**
   * Clamp display/numeric estimates. Defaults to 0.1m..10m because monocular
   * height estimates become very unstable outside that range for phone testing.
   */
  minDistanceM?: number;
  maxDistanceM?: number;
}

export interface DistanceEstimate {
  distanceM: number;
  rawDistanceM: number;
  objectHeightM: number;
  focalLengthPx: number;
  verticalFovDeg: number;
  clamped: boolean;
  label: string;
}

export interface CalibrationSample {
  bboxHeightPx: number;
  frameHeightPx: number;
  objectHeightM: number;
  measuredDistanceM: number;
}

function normaliseClassName(className: string): string {
  return className.trim().toLowerCase();
}

function getObjectHeightM(
  className: string,
  options: DistanceEstimateOptions = {},
): number {
  if (options.objectHeightM && options.objectHeightM > 0) return options.objectHeightM;

  const key = normaliseClassName(className);
  const override = options.classHeightsM?.[key];
  if (override && override > 0) return override;

  return KNOWN_HEIGHTS_M[key] ?? DEFAULT_OBJECT_HEIGHT_M;
}

function focalLengthPxFromVfov(frameHeightPx: number, verticalFovDeg: number): number {
  return frameHeightPx / (2 * Math.tan((verticalFovDeg * Math.PI) / 360));
}

function formatDistance(distanceM: number, maxDistanceM: number): string {
  if (distanceM >= maxDistanceM) return `>${Math.round(maxDistanceM)}m`;
  if (distanceM > 5) return '>5m';
  return `~${distanceM.toFixed(1)}m`;
}

/**
 * Estimate numeric distance in metres from a bounding-box height.
 *
 * This is the real-world validation API. Tests and benchmarks should use this
 * function because it returns raw metres, clamp metadata, focal length, and the
 * object height assumption used for the estimate.
 */
export function estimateDistanceMeters(
  bboxHeightPx: number,
  frameHeightPx: number,
  className: string,
  options: DistanceEstimateOptions = {},
): DistanceEstimate | null {
  if (bboxHeightPx <= 0 || frameHeightPx <= 0) return null;

  const verticalFovDeg = options.verticalFovDeg ?? CAMERA_VFOV_DEG;
  if (verticalFovDeg <= 0 || verticalFovDeg >= 180) return null;

  const objectHeightM = getObjectHeightM(className, options);
  const focalLengthPx = focalLengthPxFromVfov(frameHeightPx, verticalFovDeg);
  const rawDistanceM = (objectHeightM * focalLengthPx) / bboxHeightPx;

  const minDistanceM = options.minDistanceM ?? 0.1;
  const maxDistanceM = options.maxDistanceM ?? 10;
  const distanceM = Math.min(maxDistanceM, Math.max(minDistanceM, rawDistanceM));

  return {
    distanceM,
    rawDistanceM,
    objectHeightM,
    focalLengthPx,
    verticalFovDeg,
    clamped: distanceM !== rawDistanceM,
    label: formatDistance(distanceM, maxDistanceM),
  };
}

/**
 * Calibrate a device's vertical field of view from a real-world validation
 * sample. Use this in real-world tests when a known-height object is placed at
 * a measured distance.
 */
export function calibrateVerticalFovDeg({
  bboxHeightPx,
  frameHeightPx,
  objectHeightM,
  measuredDistanceM,
}: CalibrationSample): number | null {
  if (
    bboxHeightPx <= 0 ||
    frameHeightPx <= 0 ||
    objectHeightM <= 0 ||
    measuredDistanceM <= 0
  ) {
    return null;
  }

  const focalLengthPx = (measuredDistanceM * bboxHeightPx) / objectHeightM;
  return (2 * Math.atan(frameHeightPx / (2 * focalLengthPx)) * 180) / Math.PI;
}

export function distanceRelativeError(
  estimatedM: number,
  measuredM: number,
): number | null {
  if (estimatedM <= 0 || measuredM <= 0) return null;
  return Math.abs(estimatedM - measuredM) / measuredM;
}

/**
 * Estimates the distance (in metres) from the camera to an object.
 *
 * @param bboxHeightPx  Height of the bounding box in pixels
 * @param videoHeightPx Height of the video frame in pixels
 * @param className     YOLO class name
 * @returns Human-readable distance string, e.g. "~1.2m", ">5m", or "" if data is invalid
 */
export function estimateDistance(
  bboxHeightPx: number,
  videoHeightPx: number,
  className: string,
  options: DistanceEstimateOptions = {},
): string {
  return estimateDistanceMeters(bboxHeightPx, videoHeightPx, className, options)?.label ?? '';
}

/**
 * Same as estimateDistance, but formatted for text-to-speech. The visual
 * label (e.g. "~1.2m") reads literally as "tilde one point two m" through
 * SpeechSynthesis - this returns "about 1.2 metres" / "over 5 metres" instead.
 */
export function estimateDistanceForSpeech(
  bboxHeightPx: number,
  videoHeightPx: number,
  className: string,
  options: DistanceEstimateOptions = {},
): string {
  const estimate = estimateDistanceMeters(bboxHeightPx, videoHeightPx, className, options);
  if (!estimate) return '';
  const maxDistanceM = options.maxDistanceM ?? 10;
  if (estimate.distanceM >= maxDistanceM) return `over ${Math.round(maxDistanceM)} metres`;
  if (estimate.distanceM > 5) return 'over 5 metres';
  return `about ${estimate.distanceM.toFixed(1)} metres`;
}
