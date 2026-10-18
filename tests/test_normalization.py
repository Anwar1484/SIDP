import numpy as np
import pytest

from app.posture.normalization import (
    POSTURE_INDICES,
    REFERENCE_FALL_INDICES,
    normalize_pose,
    reference_vector,
)


def test_translation_and_scale_invariance(make_pose):
    pose = make_pose()
    shifted = make_pose(offset=(100, -30))
    doubled = pose.model_copy(deep=True)
    for k in doubled.keypoints:
        k.x *= 2
        k.y *= 2
    np.testing.assert_allclose(normalize_pose(pose).xy, normalize_pose(shifted).xy)
    np.testing.assert_allclose(normalize_pose(pose).xy, normalize_pose(doubled).xy)
    assert normalize_pose(pose).vector().shape == (51,)


def test_missing_hip_disables_normalization(make_pose):
    n = normalize_pose(make_pose(missing=(11,)))
    assert not n.valid and not n.mask.any() and not n.xy.any()


def test_missing_joint_mask_and_coco_order(make_pose):
    n = normalize_pose(make_pose(missing=(9,)))
    assert n.valid and not n.mask[9]
    np.testing.assert_array_equal(n.xy[9], [0, 0])
    assert n.xy[5, 0] < n.xy[6, 0]


def test_reference_adapter_exact_order_and_dimensions(make_pose):
    pose = make_pose()
    posture = reference_vector(pose, POSTURE_INDICES)
    fall = reference_vector(pose, REFERENCE_FALL_INDICES)
    assert posture.shape == (20,) and fall.shape == (24,)
    np.testing.assert_array_equal(posture[:4], [-20, -100, 20, -100])
    np.testing.assert_array_equal(fall[8:12], [-45, 20, 45, 20])
    with pytest.raises(ValueError):
        reference_vector(make_pose(missing=(12,)), POSTURE_INDICES)
