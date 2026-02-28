from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .types import BandTracks, Track


@dataclass(slots=True)
class ClusterMember:
    band_id: str
    track: Track
    normalized_points: list[tuple[float, float]]


@dataclass(slots=True)
class TrackCluster:
    cluster_id: str
    members: list[ClusterMember]
    prototype: list[tuple[float, float]]


def find_cross_band_tracks(
    per_band_tracks: list[BandTracks],
    spatial_tolerance_px: float,
    min_shared_bands: int,
    max_shared_bands: int,
    sample_points: int = 32,
) -> list[dict[str, Any]]:
    flattened: list[ClusterMember] = []
    for band in per_band_tracks:
        for track in band.tracks:
            if len(track.points) < 2:
                continue
            normalized = resample_points(track.points, sample_points)
            flattened.append(
                ClusterMember(
                    band_id=band.band_id,
                    track=track,
                    normalized_points=normalized,
                )
            )

    flattened.sort(key=lambda m: m.track.confidence, reverse=True)

    clusters: list[TrackCluster] = []
    cluster_counter = 0

    for member in flattened:
        best_idx = -1
        best_dist = float("inf")

        for idx, cluster in enumerate(clusters):
            existing_bands = {m.band_id for m in cluster.members}
            if member.band_id in existing_bands:
                continue

            dist = trajectory_distance(member.normalized_points, cluster.prototype)
            if dist <= spatial_tolerance_px and dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx >= 0:
            target = clusters[best_idx]
            target.members.append(member)
            target.prototype = average_trajectories(
                [m.normalized_points for m in target.members]
            )
        else:
            cluster_counter += 1
            clusters.append(
                TrackCluster(
                    cluster_id=f"c{cluster_counter}",
                    members=[member],
                    prototype=member.normalized_points,
                )
            )

    results: list[dict[str, Any]] = []
    for cluster in clusters:
        band_ids = sorted({m.band_id for m in cluster.members})
        band_count = len(band_ids)
        if band_count < min_shared_bands or band_count > max_shared_bands:
            continue

        conf_values = [m.track.confidence for m in cluster.members]
        members_payload = [
            {
                "band_id": m.band_id,
                "track_id": m.track.track_id,
                "confidence": m.track.confidence,
                "summary": m.track.summary,
                "points": [[x, y] for x, y in m.track.points],
            }
            for m in cluster.members
        ]

        results.append(
            {
                "cluster_id": cluster.cluster_id,
                "band_ids": band_ids,
                "band_count": band_count,
                "confidence_avg": sum(conf_values) / len(conf_values),
                "confidence_min": min(conf_values),
                "prototype_points": [[x, y] for x, y in cluster.prototype],
                "members": members_payload,
            }
        )

    return results


def average_trajectories(
    trajectories: list[list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    if not trajectories:
        return []

    point_count = len(trajectories[0])
    avg: list[tuple[float, float]] = []
    for idx in range(point_count):
        xs = [traj[idx][0] for traj in trajectories]
        ys = [traj[idx][1] for traj in trajectories]
        avg.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return avg


def trajectory_distance(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> float:
    if len(a) != len(b) or not a:
        return float("inf")

    total = 0.0
    for (ax, ay), (bx, by) in zip(a, b):
        total += math.dist((ax, ay), (bx, by))
    return total / len(a)


def resample_points(
    points: list[tuple[float, float]],
    target_count: int,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points[:]

    cumulative = [0.0]
    for i in range(1, len(points)):
        cumulative.append(cumulative[-1] + math.dist(points[i - 1], points[i]))

    total_len = cumulative[-1]
    if total_len == 0:
        return [points[0]] * target_count

    desired = [total_len * i / (target_count - 1) for i in range(target_count)]
    result: list[tuple[float, float]] = []

    seg = 0
    for d in desired:
        while seg < len(cumulative) - 2 and cumulative[seg + 1] < d:
            seg += 1

        left_d = cumulative[seg]
        right_d = cumulative[seg + 1]
        left_p = points[seg]
        right_p = points[seg + 1]

        if right_d == left_d:
            result.append(left_p)
            continue

        ratio = (d - left_d) / (right_d - left_d)
        x = left_p[0] + ratio * (right_p[0] - left_p[0])
        y = left_p[1] + ratio * (right_p[1] - left_p[1])
        result.append((x, y))

    return result


def clusters_to_csv_rows(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        for member in cluster["members"]:
            rows.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "band_count": cluster["band_count"],
                    "band_ids": ",".join(cluster["band_ids"]),
                    "track_id": member["track_id"],
                    "band_id": member["band_id"],
                    "confidence": member["confidence"],
                    "summary": member["summary"],
                }
            )
    return rows
