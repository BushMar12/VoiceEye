import type { Track, ProximityZone } from './tracker';
import {
  HAZARD_TIER, DEFAULT_TIER, TIER_WEIGHT, ZONE_WEIGHT,
  APPROACHING_GROWTH_THRESHOLD_ATTN, APPROACHING_BOOST,
} from '../config';

export type AnnouncementReason = 'new' | 'zone-escalation' | 'approaching' | 'sustained';

export interface Announcement {
  kind: 'single' | 'group';
  trackIds: number[];
  class: string;
  memberCount: number;
  bbox: [number, number, number, number];
  zone: ProximityZone;
  priority: number;
  reason: AnnouncementReason;
}

export interface CooldownEntry {
  lastZone: ProximityZone;
  lastSpokeAt: number;
  sustainedCount: number;
  lastReason: AnnouncementReason;
  lastSeenAt: number;
  dangerEnteredAt: number;
}

export interface ClusterEntry {
  id: number;
  class: string;
  memberTrackIds: number[];
  centroid: [number, number];
  lastAnnouncedCount: number;
  lastSpokeAt: number;
}

export interface AttentionState {
  cooldowns: Map<number, CooldownEntry>;
  clusters: Map<number, ClusterEntry>;
  windowStart: number;
  windowCount: number;
  nextClusterId: number;
}

export interface AttentionConfig {
  verbosity: 'quiet' | 'normal' | 'detailed';
  frameWidth: number;
  frameHeight: number;
  screenArea: number;
}

export interface AttentionOutput {
  toAnnounce: Announcement[];
  renderTracks: Track[];
  deescalationTones: number[];
  suppressedCount: number;
}

export function createAttentionState(now: number): AttentionState {
  return {
    cooldowns: new Map(),
    clusters: new Map(),
    windowStart: now,
    windowCount: 0,
    nextClusterId: 1,
  };
}

export function computePriority(
  cls: string,
  zone: ProximityZone,
  areaGrowthRate: number,
): number {
  const tier = HAZARD_TIER[cls] ?? DEFAULT_TIER;
  const tierW = TIER_WEIGHT[tier];
  const zoneW = ZONE_WEIGHT[zone];
  const boost = areaGrowthRate > APPROACHING_GROWTH_THRESHOLD_ATTN ? APPROACHING_BOOST : 0;
  return tierW * zoneW * (1 + boost);
}
