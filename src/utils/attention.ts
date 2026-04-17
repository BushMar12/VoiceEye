import type { Track, ProximityZone } from './tracker';
import {
  HAZARD_TIER, DEFAULT_TIER, TIER_WEIGHT, ZONE_WEIGHT,
  APPROACHING_GROWTH_THRESHOLD_ATTN, APPROACHING_BOOST,
  COOLDOWN_APPROACHING_MS, COOLDOWN_SUSTAINED_MS,
  SUSTAINED_ENTRY_DELAY_MS, SUSTAINED_MAX_COUNT,
  CLUSTER_MIN_MEMBERS, CLUSTER_RADIUS_FRAC,
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

const ZONE_RANK: Record<ProximityZone, number> = { safe: 0, near: 1, danger: 2 };

export function detectReason(
  track: Track,
  zone: ProximityZone,
  cooldown: CooldownEntry | undefined,
  now: number,
): AnnouncementReason | null {
  if (!cooldown) return 'new';

  if (ZONE_RANK[zone] > ZONE_RANK[cooldown.lastZone]) return 'zone-escalation';

  if (track.areaGrowthRate > APPROACHING_GROWTH_THRESHOLD_ATTN
      && now - cooldown.lastSpokeAt > COOLDOWN_APPROACHING_MS) {
    return 'approaching';
  }

  if (zone === 'danger'
      && now - cooldown.dangerEnteredAt > SUSTAINED_ENTRY_DELAY_MS
      && now - cooldown.lastSpokeAt > COOLDOWN_SUSTAINED_MS
      && cooldown.sustainedCount < SUSTAINED_MAX_COUNT) {
    return 'sustained';
  }

  return null;
}

export interface RawCluster {
  class: string;
  members: Track[];
  centroid: [number, number];
}

function centroidOf(t: Track): [number, number] {
  return [t.bbox[0] + t.bbox[2] / 2, t.bbox[1] + t.bbox[3] / 2];
}

export function clusterTracks(tracks: Track[], frameDiagonal: number): RawCluster[] {
  const byClass = new Map<string, Track[]>();
  for (const t of tracks) {
    const arr = byClass.get(t.class) ?? [];
    arr.push(t);
    byClass.set(t.class, arr);
  }

  const clusters: RawCluster[] = [];
  const radius = frameDiagonal * CLUSTER_RADIUS_FRAC;

  for (const [cls, members] of byClass) {
    if (members.length < CLUSTER_MIN_MEMBERS) continue;

    const used = new Set<number>();
    for (let i = 0; i < members.length; i++) {
      if (used.has(i)) continue;
      const [sx, sy] = centroidOf(members[i]);
      const group: Track[] = [members[i]];
      used.add(i);

      for (let j = i + 1; j < members.length; j++) {
        if (used.has(j)) continue;
        const [cx, cy] = centroidOf(members[j]);
        if (Math.hypot(cx - sx, cy - sy) < radius) {
          group.push(members[j]);
          used.add(j);
        }
      }

      if (group.length >= CLUSTER_MIN_MEMBERS) {
        const mx = group.reduce((s, t) => s + centroidOf(t)[0], 0) / group.length;
        const my = group.reduce((s, t) => s + centroidOf(t)[1], 0) / group.length;
        clusters.push({ class: cls, members: group, centroid: [mx, my] });
      }
    }
  }
  return clusters;
}
