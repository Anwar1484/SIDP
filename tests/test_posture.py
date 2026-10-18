import pytest

from app.posture.classifier import GeometricPostureClassifier, PostureSmoother
from app.posture.geometry import extract_geometry
from app.schemas import Posture


@pytest.mark.parametrize("kind,label", [("standing", Posture.STANDING), ("sitting", Posture.SITTING),
                                       ("bending", Posture.BENDING), ("lying", Posture.LYING)])
def test_selected_postures(config, make_pose, kind, label):
    result = GeometricPostureClassifier(config.posture).classify(extract_geometry(make_pose(kind), 480))
    assert result.label == label and result.confidence is None
    assert result.source == "geometric"
    assert result.reason


def test_occluded_pose_is_unknown(config, make_pose):
    result = GeometricPostureClassifier(config.posture).classify(
        extract_geometry(make_pose(missing=(5, 6)), 480))
    assert result.label == Posture.UNKNOWN


def test_smoothing_sitting_transition_and_missing_evidence(config, make_pose):
    classifier = GeometricPostureClassifier(config.posture)
    smooth = PostureSmoother(config.posture)
    standing = classifier.classify(extract_geometry(make_pose(), 480))
    sitting = classifier.classify(extract_geometry(make_pose("sitting"), 480))
    assert smooth.update(0, standing).label == Posture.STANDING
    assert smooth.update(0.1, sitting).label == Posture.STANDING
    assert smooth.update(0.4, sitting).label == Posture.SITTING
    unknown = classifier.classify(extract_geometry(make_pose(missing=(5, 6)), 480))
    assert smooth.update(0.5, unknown).label == Posture.UNKNOWN
