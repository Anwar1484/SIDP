"""Time-aware derivatives. Image y increases DOWN; positive velocity means descent."""
import numpy as np

from app.config import TemporalConfig
from app.schemas import MotionFeatures
from app.temporal.buffer import PoseSample


def motion_features(current: PoseSample, previous: PoseSample | None, config: TemporalConfig) -> MotionFeatures:
    g = current.geometry
    output = MotionFeatures(torso_angle=g.torso_angle, aspect_ratio=g.aspect_ratio)
    if previous is None or current.frame_size != previous.frame_size:
        return output
    dt = current.timestamp - previous.timestamp
    if dt <= 0 or dt > config.max_gap_seconds:
        return output
    if min(g.confidence_coverage, previous.geometry.confidence_coverage) < config.min_coverage:
        return output
    p, width, height = previous.geometry, *current.frame_size

    def derivative(now, before, divisor=1.0):
        return None if now is None or before is None else (now - before) / (dt * divisor)

    hip = derivative(g.hip_midpoint[1] if g.hip_midpoint else None,
                     p.hip_midpoint[1] if p.hip_midpoint else None, height)
    shoulder = derivative(g.shoulder_midpoint[1] if g.shoulder_midpoint else None,
                          p.shoulder_midpoint[1] if p.shoulder_midpoint else None, height)
    mask = np.array([a.valid and b.valid for a, b in zip(current.pose.keypoints, previous.pose.keypoints)])
    now = np.array([(k.x / width, k.y / height) for k in current.pose.keypoints])
    before = np.array([(k.x / width, k.y / height) for k in previous.pose.keypoints])
    speeds = np.linalg.norm((now[mask] - before[mask]) / dt, axis=1)
    # Require both torso endpoints in both frames before trusting fall dynamics.
    reliable = all(x is not None for x in (hip, shoulder, g.torso_angle, p.torso_angle))
    centroid_delta = (np.array(g.centroid) - p.centroid) / [width, height] / dt
    return MotionFeatures(
        vertical_velocity=derivative(g.centroid[1], p.centroid[1], height),
        centroid_velocity=float(np.linalg.norm(centroid_delta)), hip_vertical_velocity=hip,
        shoulder_vertical_velocity=shoulder, torso_angle=g.torso_angle,
        torso_angular_velocity=derivative(g.torso_angle, p.torso_angle),
        bbox_height_velocity=derivative(g.bbox_height, p.bbox_height, height),
        aspect_ratio=g.aspect_ratio, aspect_ratio_velocity=derivative(g.aspect_ratio, p.aspect_ratio),
        joint_velocity=float(speeds.mean()) if len(speeds) else None,
        motion_energy=float(np.mean(speeds ** 2)) if len(speeds) else None,
        reliable=reliable, delta_seconds=dt,
    )
