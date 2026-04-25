import type { Track, ProximityZone } from './tracker';
import {
  HAZARD_TIER, DEFAULT_TIER, TIER_WEIGHT, ZONE_WEIGHT,
  APPROACHING_GROWTH_THRESHOLD_ATTN, APPROACHING_BOOST,
  COOLDOWN_APPROACHING_MS, COOLDOWN_SUSTAINED_MS,
  SUSTAINED_ENTRY_DELAY_MS, SUSTAINED_MAX_COUNT, COOLDOWN_GC_MS,
  CLUSTER_MIN_MEMBERS, CLUSTER_RADIUS_FRAC, CLUSTER_MATCH_FRAC, CLUSTER_COUNT_DELTA,
  BUDGET_WINDOW_MS, MAX_RENDERED_BOXES, VERBOSITY_K,
  PROXIMITY_DANGER_THRESHOLD, PROXIMITY_NEAR_THRESHOLD,
  SPEAK_SAFE_TIER1_HAZARDS, SPEAK_NEAR_OR_DANGER_OBJECTS,
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

function zoneFromArea(areaPercent: number): ProximityZone {
  if (areaPercent > PROXIMITY_DANGER_THRESHOLD) return 'danger';
  if (areaPercent > PROXIMITY_NEAR_THRESHOLD)   return 'near';
  return 'safe';
}

function isHazardOverride(reason: AnnouncementReason, cls: string, lastZone: ProximityZone, zone: ProximityZone): boolean {
  // Only tier-1 safe→danger step-jumps bypass the budget
  const tier = HAZARD_TIER[cls] ?? DEFAULT_TIER;
  return tier === 1 && reason === 'zone-escalation' && lastZone === 'safe' && zone === 'danger';
}

export function isSpeechEligible(cls: string, zone: ProximityZone): boolean {
  const tier = HAZARD_TIER[cls] ?? DEFAULT_TIER;
  if (SPEAK_NEAR_OR_DANGER_OBJECTS && zone !== 'safe') return true;
  if (SPEAK_SAFE_TIER1_HAZARDS && tier === 1) return true;
  return false;
}

function matchCluster(state: AttentionState, cls: string, centroid: [number, number], frameDiagonal: number): ClusterEntry | undefined {
  const radius = frameDiagonal * CLUSTER_MATCH_FRAC;
  for (const entry of state.clusters.values()) {
    if (entry.class !== cls) continue;
    if (Math.hypot(entry.centroid[0] - centroid[0], entry.centroid[1] - centroid[1]) < radius) {
      return entry;
    }
  }
  return undefined;
}

export function runAttention(
  tracks: Track[],
  now: number,
  state: AttentionState,
  config: AttentionConfig,
): AttentionOutput {
  // Reset budget window if expired
  if (now - state.windowStart > BUDGET_WINDOW_MS) {
    state.windowStart = now;
    state.windowCount = 0;
  }

  const frameDiagonal = Math.hypot(config.frameWidth, config.frameHeight);
  const budgetK = VERBOSITY_K[config.verbosity];
  const deescalationTones: number[] = [];

  const activeTracks = tracks.filter(t => t.age === 0);
  const clusters = clusterTracks(activeTracks, frameDiagonal);
  const clusteredIds = new Set<number>();
  clusters.forEach(c => c.members.forEach(m => clusteredIds.add(m.id)));

  // Build candidates ----------------------------------------------------
  interface Candidate {
    announcement: Announcement;
    override: boolean;
    commit: () => void; // applies cooldown / cluster state updates
  }
  const candidates: Candidate[] = [];

  // Singles
  for (const t of activeTracks) {
    if (clusteredIds.has(t.id)) continue;

    const areaPercent = (t.bbox[2] * t.bbox[3]) / config.screenArea;
    const zone = zoneFromArea(areaPercent);
    const cd = state.cooldowns.get(t.id);

    // Touch lastSeenAt for GC
    if (cd) cd.lastSeenAt = now;

    // De-escalation tone (independent of announce)
    if (cd && ZONE_RANK[zone] < ZONE_RANK[cd.lastZone]) {
      deescalationTones.push(t.id);
    }

    const reason = detectReason(t, zone, cd, now);
    if (!reason) {
      // Still update the zone tracking in cooldown so transitions work next frame
      if (cd) {
        if (zone === 'danger' && cd.lastZone !== 'danger') cd.dangerEnteredAt = now;
        cd.lastZone = zone;
      }
      continue;
    }

    if (!isSpeechEligible(t.class, zone)) {
      if (cd) {
        if (zone === 'danger' && cd.lastZone !== 'danger') cd.dangerEnteredAt = now;
        cd.lastZone = zone;
      }
      continue;
    }

    const priority = computePriority(t.class, zone, t.areaGrowthRate);
    const override = cd ? isHazardOverride(reason, t.class, cd.lastZone, zone) : false;

    const ann: Announcement = {
      kind: 'single',
      trackIds: [t.id],
      class: t.class,
      memberCount: 1,
      bbox: t.bbox,
      zone,
      priority,
      reason,
    };

    candidates.push({
      announcement: ann,
      override,
      commit: () => {
        const prev = state.cooldowns.get(t.id);
        state.cooldowns.set(t.id, {
          lastZone: zone,
          lastSpokeAt: now,
          sustainedCount: reason === 'sustained' ? (prev?.sustainedCount ?? 0) + 1 : (reason === 'zone-escalation' ? 0 : (prev?.sustainedCount ?? 0)),
          lastReason: reason,
          lastSeenAt: now,
          dangerEnteredAt: zone === 'danger' && prev?.lastZone !== 'danger' ? now : (prev?.dangerEnteredAt ?? 0),
        });
      },
    });
  }

  // Clusters
  for (const rc of clusters) {
    const match = matchCluster(state, rc.class, rc.centroid, frameDiagonal);
    const memberCount = rc.members.length;
    let reason: AnnouncementReason | null = null;

    if (!match) reason = 'new';
    else if (Math.abs(memberCount - match.lastAnnouncedCount) >= CLUSTER_COUNT_DELTA) reason = 'zone-escalation';

    if (!reason) {
      // Update cluster position without announcing
      if (match) {
        match.centroid = rc.centroid;
        match.memberTrackIds = rc.members.map(m => m.id);
      }
      continue;
    }

    // Use max-priority member for the group's priority and highest-area bbox as display
    let priority = 0;
    let repZone: ProximityZone = 'safe';
    let repBbox = rc.members[0].bbox;
    let repArea = 0;
    for (const m of rc.members) {
      const ap = (m.bbox[2] * m.bbox[3]) / config.screenArea;
      const z = zoneFromArea(ap);
      const p = computePriority(m.class, z, m.areaGrowthRate);
      if (p > priority) { priority = p; repZone = z; }
      if (m.bbox[2] * m.bbox[3] > repArea) { repArea = m.bbox[2] * m.bbox[3]; repBbox = m.bbox; }
    }

    if (!isSpeechEligible(rc.class, repZone)) continue;

    const ann: Announcement = {
      kind: 'group',
      trackIds: rc.members.map(m => m.id),
      class: rc.class,
      memberCount,
      bbox: repBbox,
      zone: repZone,
      priority,
      reason,
    };

    candidates.push({
      announcement: ann,
      override: false,
      commit: () => {
        const clusterId = match?.id ?? state.nextClusterId++;
        state.clusters.set(clusterId, {
          id: clusterId,
          class: rc.class,
          memberTrackIds: rc.members.map(m => m.id),
          centroid: rc.centroid,
          lastAnnouncedCount: memberCount,
          lastSpokeAt: now,
        });
      },
    });
  }

  // Sort + top-K + override ---------------------------------------------
  candidates.sort((a, b) => b.announcement.priority - a.announcement.priority);

  const selected: Announcement[] = [];
  const selectedKeys = new Set<string>();

  // Overrides first
  for (const c of candidates) {
    if (c.override) {
      selected.push(c.announcement);
      c.commit();
      state.windowCount++;
      selectedKeys.add(`${c.announcement.kind}:${c.announcement.trackIds.join(',')}`);
    }
  }

  // Then fill budget
  for (const c of candidates) {
    const key = `${c.announcement.kind}:${c.announcement.trackIds.join(',')}`;
    if (selectedKeys.has(key)) continue;
    if (state.windowCount >= budgetK) break;
    selected.push(c.announcement);
    c.commit();
    state.windowCount++;
    selectedKeys.add(key);
  }

  const suppressedCount = candidates.length - selected.length;

  // Render cap ----------------------------------------------------------
  const renderTracks = [...activeTracks]
    .map(t => {
      const ap = (t.bbox[2] * t.bbox[3]) / config.screenArea;
      return { t, p: computePriority(t.class, zoneFromArea(ap), t.areaGrowthRate) };
    })
    .sort((a, b) => b.p - a.p)
    .slice(0, MAX_RENDERED_BOXES)
    .map(({ t }) => t);

  // Cooldown GC ---------------------------------------------------------
  for (const [id, cd] of state.cooldowns) {
    if (now - cd.lastSeenAt > COOLDOWN_GC_MS) state.cooldowns.delete(id);
  }

  return { toAnnounce: selected, renderTracks, deescalationTones, suppressedCount };
}
