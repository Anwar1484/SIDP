import numpy as np
import pytest

from app.posture.classifier import GeometricPostureClassifier
from app.posture.geometry import extract_geometry
from app.posture.normalization import normalize_pose
from app.temporal.buffer import PoseSample, TemporalBuffer
from app.temporal.features import motion_features


def sample(config, pose, t):
    g = extract_geometry(pose, 480)
    posture = GeometricPostureClassifier(config.posture).classify(g)
    return PoseSample(t, pose, normalize_pose(pose), posture, g, (640, 480))


def test_bounded_buffer_resamples_and_never_extrapolates(config, make_pose):
    buffer = TemporalBuffer(config.temporal)
    assert buffer.window(60)[0] is None
    for i in range(150):
        buffer.append(sample(config, make_pose(), i/20))
    assert len(buffer.samples) == config.temporal.capacity
    window, ratio = buffer.window(60)
    assert window.shape == (60, 51) and ratio == 1
    np.testing.assert_allclose(window[:, :34], np.tile(normalize_pose(make_pose()).vector()[:34], (60, 1)))


def test_gap_resets_and_repeated_time_rejected(config, make_pose):
    buffer = TemporalBuffer(config.temporal)
    buffer.append(sample(config, make_pose(), 0))
    with pytest.raises(ValueError):
        buffer.append(sample(config, make_pose(), 0))
    assert buffer.append(sample(config, make_pose(), 1))
    assert len(buffer.samples) == 1 and buffer.window(60)[0] is None


def test_invalid_windows_are_not_confident(config, make_pose):
    buffer = TemporalBuffer(config.temporal)
    for i in range(80):
        buffer.append(sample(config, make_pose(missing=(11,)), i/30))
    window, ratio = buffer.window(60)
    assert window is None and ratio == 0


def test_time_aware_motion_and_positive_down_convention(config, make_pose):
    a = sample(config, make_pose(), 0)
    b = sample(config, make_pose(offset=(0, 48)), 0.5)
    motion = motion_features(b, a, config.temporal)
    assert motion.hip_vertical_velocity == pytest.approx(0.2)
    assert motion.vertical_velocity == pytest.approx(0.2)
    assert motion.torso_angular_velocity == pytest.approx(0)
    assert motion.reliable
    c = sample(config, make_pose(offset=(0, 48)), 0.25)
    assert motion_features(c, a, config.temporal).vertical_velocity == pytest.approx(0.4)
    assert not motion_features(sample(config, make_pose(), 2), a, config.temporal).reliable
