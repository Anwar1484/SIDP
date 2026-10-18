import numpy as np
import pytest

from app.config import load_config
from app.schemas import BoundingBox, Keypoint, PersonPose


@pytest.fixture
def config():
    cfg = load_config()
    cfg.tracker.backend = "centroid"
    cfg.api.camera_enabled = False
    cfg.secondary.enabled = cfg.risk.enabled = cfg.scene.enabled = False
    return cfg


def skeleton(kind="standing", offset=(0, 0), missing=()) -> PersonPose:
    xy = np.array([[270, 105], [260, 100], [280, 100], [250, 105], [290, 105],
                   [250, 150], [290, 150], [235, 215], [305, 215], [225, 270], [315, 270],
                   [255, 250], [285, 250], [255, 340], [285, 340], [255, 430], [285, 430]], dtype=float)
    box = np.array([210, 90, 325, 440], dtype=float)
    if kind == "sitting":
        xy[[13, 14, 15, 16]] = [[340, 260], [365, 260], [340, 350], [365, 350]]
        box = np.array([210, 90, 380, 365], dtype=float)
    elif kind == "bending":
        xy[[0, 1, 2, 3, 4, 5, 6]] = [[360, 195], [355, 190], [365, 190], [350, 195], [370, 195],
                                     [345, 210], [375, 210]]
        box = np.array([210, 175, 395, 440], dtype=float)
    elif kind == "lying":
        xy = np.column_stack((xy[:, 1] + 50, 500 - xy[:, 0]))
        box = np.array([130, 175, 500, 290], dtype=float)
    xy += offset
    box += [*offset, *offset]
    return PersonPose(bbox=BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), box))),
                      detection_confidence=0.95,
                      keypoints=[Keypoint(index=i, x=x, y=y, confidence=0 if i in missing else 0.95,
                                          valid=i not in missing) for i, (x, y) in enumerate(xy)])


@pytest.fixture
def make_pose():
    return skeleton
