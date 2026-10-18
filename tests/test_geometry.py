import pytest

from app.posture.geometry import angle, clip_bbox, extract_geometry


def test_angles_and_degenerate_points():
    assert angle((0, 0), (1, 0), (1, 1)) == pytest.approx(90)
    assert angle((0, 0), (1, 0), (2, 0)) == pytest.approx(180)
    assert angle((0, 0), (0, 0), (1, 1)) is None
    assert angle(None, (0, 0), (1, 1)) is None


def test_bbox_clipping_and_invalid_boxes():
    assert clip_bbox([-10, -20, 900, 600], 640, 480).as_list() == [0, 0, 640, 480]
    assert clip_bbox([900, 1, 1000, 20], 640, 480) is None
    assert clip_bbox([1, 1, 1, 2], 640, 480) is None
    assert clip_bbox([1, 1, float("nan"), 2], 640, 480) is None


def test_missing_joints_are_nullable(make_pose):
    g = extract_geometry(make_pose(missing=(5, 11, 13)), 480)
    assert g.shoulder_midpoint is None and g.hip_midpoint is None
    assert g.left_knee_angle is None and g.torso_angle is None
    assert g.confidence_coverage == pytest.approx(14/17)


@pytest.mark.parametrize("kind,angle_expected", [("standing", 0), ("sitting", 0), ("lying", 90)])
def test_torso_orientation(make_pose, kind, angle_expected):
    assert extract_geometry(make_pose(kind), 480).torso_angle == pytest.approx(angle_expected)
