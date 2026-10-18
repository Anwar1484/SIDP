"""Confidence-aware geometry with nullable missing measurements."""
import math

import numpy as np

from app.schemas import BoundingBox, GeometryFeatures, PersonPose


def clip_bbox(xyxy, width: int, height: int) -> BoundingBox | None:
    values = np.asarray(xyxy, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    x1, y1, x2, y2 = values
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def angle(a, b, c) -> float | None:
    """Angle ABC in degrees; missing or coincident joints have no angle."""
    if any(p is None for p in (a, b, c)):
        return None
    u, v = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-8:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / norm, -1, 1))))


def extract_geometry(pose: PersonPose, frame_height: int) -> GeometryFeatures:
    points = [(k.x, k.y) if k.valid else None for k in pose.keypoints]

    def midpoint(i: int, j: int):
        if points[i] is None or points[j] is None:
            return None
        return tuple((np.asarray(points[i]) + np.asarray(points[j])) / 2)

    shoulders, hips, ankles = midpoint(5, 6), midpoint(11, 12), midpoint(15, 16)
    torso = None if shoulders is None or hips is None else tuple(np.asarray(shoulders) - hips)
    torso_angle = None
    if torso is not None and np.linalg.norm(torso) > 1e-8:
        # Unsigned angle to the vertical body axis, in [0, 90].
        torso_angle = math.degrees(math.atan2(abs(torso[0]), abs(torso[1])))
    box = pose.bbox
    return GeometryFeatures(
        shoulder_midpoint=shoulders, hip_midpoint=hips, ankle_midpoint=ankles,
        left_knee_angle=angle(points[11], points[13], points[15]),
        right_knee_angle=angle(points[12], points[14], points[16]),
        left_hip_angle=angle(points[5], points[11], points[13]),
        right_hip_angle=angle(points[6], points[12], points[14]),
        torso_vector=torso, torso_angle=torso_angle, bbox_width=box.width,
        bbox_height=box.height, aspect_ratio=box.width / box.height, centroid=box.center,
        head_to_hip_vertical_displacement=(hips[1] - points[0][1])
        if hips is not None and points[0] is not None else None,
        hip_height=(frame_height - hips[1]) / frame_height if hips is not None else None,
        shoulder_height=(frame_height - shoulders[1]) / frame_height if shoulders is not None else None,
        confidence_coverage=sum(k.valid for k in pose.keypoints) / 17,
    )
