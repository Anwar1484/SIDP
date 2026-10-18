"""Causal pose-window features shared by training and inference; never probabilities."""
import numpy as np

from app.schemas import AppearanceEvidence
from app.temporal.buffer import PoseSample, TemporalBuffer

STATIC_NAMES = ["torso_angle", "aspect_ratio", "box_height", "box_width", "hip_y", "shoulder_y",
                "knee_angle", "hip_angle", "coverage"]
TEMPORAL_NAMES = ["hip_drop", "shoulder_drop", "torso_change", "height_retention", "aspect_change",
                  "max_down_speed", "min_down_speed", "hip_speed_std", "max_rotation", "box_drop"]
RGB_NAMES = ["fallen_box_score", "fallen_box_present"]
FEATURE_NAMES = STATIC_NAMES + TEMPORAL_NAMES + RGB_NAMES


def static_features(sample: PoseSample) -> np.ndarray:
    g = sample.geometry
    width, height = sample.frame_size
    knees = [v for v in (g.left_knee_angle, g.right_knee_angle) if v is not None]
    hips = [v for v in (g.left_hip_angle, g.right_hip_angle) if v is not None]
    return np.array([
        g.torso_angle / 90 if g.torso_angle is not None else np.nan,
        min(g.aspect_ratio, 6), g.bbox_height / height, g.bbox_width / width,
        g.hip_midpoint[1] / height if g.hip_midpoint else np.nan,
        g.shoulder_midpoint[1] / height if g.shoulder_midpoint else np.nan,
        min(knees) / 180 if knees else np.nan, min(hips) / 180 if hips else np.nan,
        g.confidence_coverage,
    ], dtype=np.float64)


def event_features(buffer: TemporalBuffer, appearance: AppearanceEvidence,
                   window_seconds: float = 1.2, min_coverage: float = .5):
    samples = list(buffer.samples)
    if not samples:
        return None, "waiting_for_person"
    current = samples[-1]
    visible = current.pose.keypoints
    body_ready = (all(visible[i].valid for i in (5, 6, 11, 12))
                  and sum(visible[i].valid for i in (13, 14, 15, 16)) >= 2
                  and current.geometry.confidence_coverage >= min_coverage)
    if not body_ready:
        return None, "insufficient_body_coverage"
    start, end = current.timestamp - window_seconds, current.timestamp
    if samples[0].timestamp > start + 1e-6:
        return None, "building_temporal_window"
    times = np.array([s.timestamp for s in samples])
    values = np.stack([static_features(s) for s in samples])
    targets = np.linspace(start, end, 7)
    history = np.stack([np.interp(targets, times, values[:, col]) for col in range(values.shape[1])], axis=1)
    if not np.isfinite(history[:, [0, 4, 5]]).all():
        return None, "incomplete_temporal_window"
    dt = window_seconds / 6
    hip_velocity = np.diff(history[:, 4]) / dt
    rotation = np.diff(history[:, 0]) / dt
    box_centers = np.array([s.geometry.centroid[1]/s.frame_size[1] for s in samples])
    center_history = np.interp(targets, times, box_centers)
    temporal = [history[-1, 4]-history[0, 4], history[-1, 5]-history[0, 5],
                history[-1, 0]-history[0, 0], history[-1, 2]/max(history[:, 2].max(), .001),
                history[-1, 1]-history[0, 1], hip_velocity.max(), hip_velocity.min(),
                hip_velocity.std(), np.abs(rotation).max(), center_history[-1]-center_history[0]]
    rgb = [appearance.fallen_confidence or 0.0, float(appearance.fallen_confidence is not None)]
    return np.concatenate((static_features(current), temporal, rgb)), "ready"
