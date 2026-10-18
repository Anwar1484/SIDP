"""Gated Hungarian IoU/centroid matching with constant-velocity prediction.

IDs are local to a camera session. This small POC tracker cannot guarantee identity
through complete occlusion or crossing; stale tracks never reuse IDs.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.config import TrackerConfig
from app.schemas import BoundingBox, PersonPose


def iou(a: BoundingBox, b: BoundingBox) -> float:
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0, min(a.y2, b.y2) - max(a.y1, b.y1)
    )
    return intersection / (a.width * a.height + b.width * b.height - intersection)


@dataclass
class Track:
    id: int
    pose: PersonPose
    last_seen: float
    velocity: tuple[float, float] = (0.0, 0.0)


def create_tracker(config: TrackerConfig):
    """Use the configured association backend for live and extracted sequences."""
    if config.backend == "bytetrack":
        from app.tracking.byte_tracker import ByteTrackTracker
        return ByteTrackTracker(config)
    return PersonTracker(config)


class PersonTracker:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.last_timestamp: float | None = None
        self.expired_ids: list[int] = []

    def update(self, poses: list[PersonPose], timestamp: float, frame_size: tuple[int, int]) -> list[Track]:
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("tracking timestamps must increase")
        self.last_timestamp = timestamp
        self.expired_ids = [i for i, t in self.tracks.items()
                            if timestamp - t.last_seen > self.config.stale_seconds]
        for i in self.expired_ids:
            del self.tracks[i]
        existing = list(self.tracks.values())
        matched: dict[int, Track] = {}
        if existing and poses:
            costs = np.full((len(existing), len(poses)), 1e6)
            diagonal = float(np.hypot(*frame_size))
            for i, track in enumerate(existing):
                dx, dy = np.array(track.velocity) * (timestamp - track.last_seen)
                b = track.pose.bbox
                predicted = BoundingBox(x1=b.x1+dx, y1=b.y1+dy, x2=b.x2+dx, y2=b.y2+dy)
                for j, pose in enumerate(poses):
                    overlap = iou(predicted, pose.bbox)
                    distance = float(np.linalg.norm(np.array(predicted.center) - pose.bbox.center) / diagonal)
                    if overlap >= self.config.min_iou or distance <= self.config.max_centroid_distance:
                        w = self.config.iou_weight
                        costs[i, j] = w * (1 - overlap) + (1 - w) * distance
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] >= 1e6:
                    continue
                track = existing[i]
                dt = timestamp - track.last_seen
                track.velocity = tuple((np.array(poses[j].bbox.center) - track.pose.bbox.center) / dt)
                track.pose, track.last_seen = poses[j], timestamp
                matched[j] = track
        for j, pose in enumerate(poses):
            if j not in matched:
                track = Track(self.next_id, pose, timestamp)
                self.tracks[track.id] = track
                matched[j] = track
                self.next_id += 1
        return [matched[j] for j in range(len(poses))]
