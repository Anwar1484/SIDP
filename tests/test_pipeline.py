import json

import numpy as np
import pytest

from app.fusion.pipeline import VisionPipeline
from app.schemas import FallState, Posture, VisionObservation


class FakePoseBackend:
    def __init__(self, poses):
        self.poses = poses

    def infer(self, frame):
        return self.poses


def test_track_buffers_separate_and_expire(config, make_pose):
    backend = FakePoseBackend([make_pose(offset=(-150, 0)), make_pose("sitting", offset=(180, 0))])
    pipeline = VisionPipeline(config, backend)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(10):
        output = pipeline.process(frame, i/10, i)
    assert output.persons[0].posture == Posture.STANDING
    assert output.persons[1].posture == Posture.SITTING
    assert pipeline.contexts[1].buffer is not pipeline.contexts[2].buffer
    assert pipeline.contexts[1].buffer.samples[-1].pose.bbox.center[0] < 200
    assert pipeline.contexts[2].buffer.samples[-1].pose.bbox.center[0] > 400
    backend.poses = []
    assert pipeline.process(frame, 2.1, 10).persons == []
    assert not pipeline.contexts


def test_json_roundtrip_finite_and_missing_temporal_evidence(config, make_pose):
    pipeline = VisionPipeline(config, FakePoseBackend([make_pose()]))
    output = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8), 0, 0)
    encoded = output.model_dump_json()
    assert VisionObservation.model_validate_json(encoded) == output
    assert json.loads(encoded)["persons"][0]["fall_probability"] is None
    assert output.persons[0].state == FallState.NORMAL
    assert output.persons[0].fall_inference.calibrated is False
    assert "alert" not in json.loads(encoded)
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8), 0, 1)


def test_stationary_pose_does_not_fabricate_model_confidence(config, make_pose):
    pipeline = VisionPipeline(config, FakePoseBackend([make_pose()]))
    for i in range(70):
        output = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8), i/30, i)
    assert output.persons[0].fall_probability is None
    assert output.persons[0].movement_state == "STATIONARY"
    assert output.persons[0].state == FallState.NORMAL
    assert len(pipeline.contexts[1].buffer.samples) == 70


def test_geometry_alone_does_not_generate_a_model_fall_alert(config, make_pose):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    standing, lying = make_pose(), make_pose("lying", offset=(0, 155))
    backend = FakePoseBackend([standing])
    pipeline = VisionPipeline(config, backend)
    states = set()
    for i in range(35):
        fraction = 0 if i < 5 else min(1, (i-4)/8)
        pose = standing.model_copy(deep=True)
        for k, a, b in zip(pose.keypoints, standing.keypoints, lying.keypoints):
            k.x, k.y = a.x*(1-fraction)+b.x*fraction, a.y*(1-fraction)+b.y*fraction
        box = np.array(standing.bbox.as_list())*(1-fraction)+np.array(lying.bbox.as_list())*fraction
        pose.bbox = type(pose.bbox)(**dict(zip(("x1", "y1", "x2", "y2"), box)))
        backend.poses = [pose]
        result = pipeline.process(frame, i/10, i)
        assert result.persons[0].track_id == 1
        states.add(result.persons[0].state)
    assert states == {FallState.NORMAL}
    backend.poses = [standing]
    for i in range(35, 50):
        result = pipeline.process(frame, i/10, i)
    assert result.persons[0].state == FallState.NORMAL
