"""Versioned, dimensionless YOLO26 features. No scaler is fitted during inference."""
from dataclasses import dataclass

import numpy as np

from app.schemas import PersonPose

NORMALIZATION_VERSION = "coco17_midhip_torso_mask_v1"
POSTURE_INDICES = (5, 6, 7, 8, 11, 12, 13, 14, 15, 16)
REFERENCE_FALL_INDICES = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


@dataclass(frozen=True)
class NormalizedPose:
    xy: np.ndarray
    mask: np.ndarray
    valid: bool
    scale_pixels: float

    def vector(self) -> np.ndarray:
        """34 interleaved x/y values followed by 17 visibility masks."""
        return np.concatenate((self.xy.flatten(), self.mask.astype(np.float32))).astype(np.float32)


def normalize_pose(pose: PersonPose) -> NormalizedPose:
    xy = np.array([(k.x, k.y) for k in pose.keypoints], dtype=np.float32)
    mask = np.array([k.valid for k in pose.keypoints], dtype=bool)
    if not mask[[5, 6, 11, 12]].all():
        return NormalizedPose(np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=bool), False, 0)
    origin = xy[[11, 12]].mean(axis=0)
    scale = float(np.linalg.norm(xy[[5, 6]].mean(axis=0) - origin))
    if scale < 1e-6:
        return NormalizedPose(np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=bool), False, 0)
    output = (xy - origin) / scale
    output[~mask] = 0
    return NormalizedPose(output, mask, True, scale)


def reference_vector(pose: PersonPose, indices: tuple[int, ...]) -> np.ndarray:
    """Audit adapter: reproduce reference inference's raw mid-hip pixel offsets.

    This is NOT the app's normalized representation and is not sent to shipped H5
    weights. Shape agreement alone cannot establish training compatibility.
    """
    xy = np.array([(k.x, k.y) for k in pose.keypoints], dtype=np.float32)
    if not all(pose.keypoints[i].valid for i in (11, 12)):
        raise ValueError("reference adapter requires both hips")
    return (xy[list(indices)] - xy[[11, 12]].mean(axis=0)).flatten()
