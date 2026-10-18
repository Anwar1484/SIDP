import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.inference.yolo26_pose import Yolo26PoseBackend


class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class Boxes:
    def __init__(self):
        self.xyxy, self.conf = Tensor([[-10, -10, 200, 300]]), Tensor([0.9])

    def __len__(self):
        return 1


def test_backend_filters_missing_confidence_and_retains_original_pixels(config):
    joints = np.tile([100.0, 120.0, 0.95], (1, 17, 1))
    joints[0, 3] = [150, 150, 0.1]
    joints[0, 4] = [float("nan"), 0, 0.9]
    result = SimpleNamespace(boxes=Boxes(), keypoints=SimpleNamespace(data=Tensor(joints)))
    backend = Yolo26PoseBackend.__new__(Yolo26PoseBackend)
    backend.config, backend.device = config.pose, "cpu"
    backend.model = SimpleNamespace(predict=lambda **kwargs: [result])
    pose = backend.infer(np.zeros((480, 640, 3), dtype=np.uint8))[0]
    assert pose.bbox.as_list() == [0, 0, 200, 300]
    assert pose.keypoints[0].x == 100 and pose.keypoints[0].y == 120
    assert not pose.keypoints[3].valid and not pose.keypoints[4].valid
    assert len(pose.keypoints) == 17


def test_reject_renamed_older_checkpoint(config, monkeypatch):
    fake = SimpleNamespace(task="pose", model=SimpleNamespace(yaml={"yaml_file": "yolov8n-pose.yaml"}))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: fake))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    monkeypatch.setattr("app.inference.yolo26_pose.select_device", lambda _: "cpu")
    with pytest.raises(ValueError, match="architecture is not YOLO26"):
        Yolo26PoseBackend(config.pose)
