import type { Detection } from './yolo';
import { METRICS_BUFFER_SIZE, METRICS_LOG_INTERVAL } from '../config';

export interface MetricsSummary {
  totalFrames: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  fps: number;
  avgDetectionsPerFrame: number;
  confidenceDistribution: Record<string, number>;
  announcementsTotal: number;
  suppressedTotal: number;
  announcementsPerMin: number;
  activeClusters: number;
}

class InferenceMetrics {
  private latencies: number[] = [];
  private detectionCounts: number[] = [];
  private confidenceBuckets = { '0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0, '0.8-0.9': 0, '0.9-1.0': 0 };
  private totalFrames = 0;
  private firstTimestamp = 0;
  private lastTimestamp = 0;
  private framesSinceLog = 0;
  private announcementsTotal = 0;
  private suppressedTotal = 0;
  private lastActiveClusters = 0;

  recordInference(latencyMs: number, detections: Detection[]): void {
    const now = performance.now();
    if (this.totalFrames === 0) this.firstTimestamp = now;
    this.lastTimestamp = now;
    this.totalFrames++;

    // Circular buffer for latencies
    if (this.latencies.length >= METRICS_BUFFER_SIZE) this.latencies.shift();
    this.latencies.push(latencyMs);

    // Circular buffer for detection counts
    if (this.detectionCounts.length >= METRICS_BUFFER_SIZE) this.detectionCounts.shift();
    this.detectionCounts.push(detections.length);

    // Confidence distribution
    for (const det of detections) {
      if (det.score < 0.6) this.confidenceBuckets['0.5-0.6']++;
      else if (det.score < 0.7) this.confidenceBuckets['0.6-0.7']++;
      else if (det.score < 0.8) this.confidenceBuckets['0.7-0.8']++;
      else if (det.score < 0.9) this.confidenceBuckets['0.8-0.9']++;
      else this.confidenceBuckets['0.9-1.0']++;
    }

    // Periodic console log
    this.framesSinceLog++;
    if (this.framesSinceLog >= METRICS_LOG_INTERVAL) {
      this.logSummary();
      this.framesSinceLog = 0;
    }
  }

  recordAttention(announced: number, suppressed: number, clusters: number): void {
    this.announcementsTotal += announced;
    this.suppressedTotal    += suppressed;
    this.lastActiveClusters = clusters;
  }

  getSummary(): MetricsSummary {
    const sorted = [...this.latencies].sort((a, b) => a - b);
    const avg = sorted.length > 0
      ? sorted.reduce((s, v) => s + v, 0) / sorted.length
      : 0;
    const p95Idx = Math.floor(sorted.length * 0.95);
    const p95 = sorted[p95Idx] ?? 0;

    const elapsed = (this.lastTimestamp - this.firstTimestamp) / 1000;
    const fps = elapsed > 0 ? this.totalFrames / elapsed : 0;

    const avgDet = this.detectionCounts.length > 0
      ? this.detectionCounts.reduce((s, v) => s + v, 0) / this.detectionCounts.length
      : 0;

    return {
      totalFrames: this.totalFrames,
      avgLatencyMs: Math.round(avg * 10) / 10,
      p95LatencyMs: Math.round(p95 * 10) / 10,
      fps: Math.round(fps * 10) / 10,
      avgDetectionsPerFrame: Math.round(avgDet * 10) / 10,
      confidenceDistribution: { ...this.confidenceBuckets },
      announcementsTotal: this.announcementsTotal,
      suppressedTotal:    this.suppressedTotal,
      announcementsPerMin: elapsed > 0 ? (this.announcementsTotal / elapsed) * 60 : 0,
      activeClusters:     this.lastActiveClusters,
    };
  }

  reset(): void {
    this.latencies = [];
    this.detectionCounts = [];
    this.confidenceBuckets = { '0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0, '0.8-0.9': 0, '0.9-1.0': 0 };
    this.totalFrames = 0;
    this.firstTimestamp = 0;
    this.lastTimestamp = 0;
    this.framesSinceLog = 0;
    this.announcementsTotal = 0;
    this.suppressedTotal = 0;
    this.lastActiveClusters = 0;
  }

  private logSummary(): void {
    const s = this.getSummary();
    console.log(
      `[VoiceEye] fps=${s.fps} avgLatency=${s.avgLatencyMs}ms ` +
      `p95=${s.p95LatencyMs}ms detections/frame=${s.avgDetectionsPerFrame} ` +
      `announcements/min=${Math.round(s.announcementsPerMin * 10) / 10} ` +
      `suppressed=${s.suppressedTotal} clusters=${s.activeClusters}`
    );
  }
}

// Singleton instance
export const inferenceMetrics = new InferenceMetrics();
