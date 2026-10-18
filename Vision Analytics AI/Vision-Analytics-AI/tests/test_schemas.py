import pytest
from pydantic import ValidationError

from app.schemas import BoundingBox, Keypoint, PersonPose


def test_reject_nonfinite_and_invalid_probability():
    with pytest.raises(ValidationError):
        BoundingBox(x1=0, y1=0, x2=float("inf"), y2=5)
    with pytest.raises(ValidationError):
        Keypoint(index=0, x=0, y=0, confidence=1.1, valid=True)


def test_reject_reordered_and_short_keypoints(make_pose):
    pose = make_pose()
    values = pose.model_dump()
    values["keypoints"] = values["keypoints"][::-1]
    with pytest.raises(ValidationError):
        PersonPose.model_validate(values)
    values["keypoints"] = values["keypoints"][:16]
    with pytest.raises(ValidationError):
        PersonPose.model_validate(values)
