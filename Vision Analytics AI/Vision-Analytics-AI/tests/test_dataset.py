import json

import numpy as np
import pytest

from app.posture.normalization import NORMALIZATION_VERSION
from app.temporal.dataset import prepare_windows


def sequence(tmp_path, make_pose, gaps=False):
    file = tmp_path / "track_1.jsonl"
    lines = []
    for i in range(80):
        pose = make_pose()
        row = {"video_id": "subject-a.mp4", "track_id": 1, "frame": i,
               "timestamp": i/30 + (2 if gaps and i >= 40 else 0), "pose_family": "YOLO26",
               "normalization": NORMALIZATION_VERSION, "frame_size": [640, 480],
               "bbox": pose.bbox.as_list(), "keypoints": [k.model_dump() for k in pose.keypoints]}
        lines.append(json.dumps(row))
    file.write_text("\n".join(lines), encoding="utf-8")


def test_no_fabricated_labels_and_exact_window_contract(tmp_path, config, make_pose):
    sequence(tmp_path, make_pose)
    output = prepare_windows(tmp_path, None, config.temporal, 60, 15, "fall")
    assert output["X"].shape == (2, 60, 51)
    assert (output["y"] == -1).all()
    assert set(output["groups"]) == {"subject-a.mp4"}
    assert np.isfinite(output["X"]).all()


def test_explicit_interval_labels_and_gap_rejection(tmp_path, config, make_pose):
    sequence(tmp_path, make_pose)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([{"video_id": "subject-a.mp4", "track_id": 1,
                                   "start": 0, "end": 3, "fall": 0}]), encoding="utf-8")
    output = prepare_windows(tmp_path, labels, config.temporal, 60, 15, "fall")
    assert (output["y"] == 0).all()
    sequence(tmp_path, make_pose, gaps=True)
    assert len(prepare_windows(tmp_path, labels, config.temporal, 60, 15, "fall")["X"]) == 0


def test_overlapping_annotations_rejected(tmp_path, config, make_pose):
    sequence(tmp_path, make_pose)
    labels = tmp_path / "labels.json"
    entry = {"video_id": "subject-a.mp4", "track_id": 1, "start": 0, "end": 3, "fall": 1}
    labels.write_text(json.dumps([entry, entry]), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        prepare_windows(tmp_path, labels, config.temporal, 60, 15, "fall")
