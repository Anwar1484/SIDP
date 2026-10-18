import json

import cv2
import numpy as np
import pytest

from app.capture.video import VideoCapture
from app.fusion.pipeline import VisionPipeline
from scripts.evaluate_video import evaluate
from tests.test_pipeline import FakePoseBackend


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fallback_fps_rejected(fps):
    with pytest.raises(ValueError, match="FPS"):
        VideoCapture("unused.avi", fps)


def test_video_evaluation_outputs_decodable_media_csv_and_statistics(tmp_path, config, make_pose):
    video_path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
    writer.release()
    with VideoCapture(video_path) as source:
        frames = list(source)
    assert len(frames) == 12
    assert frames[-1].timestamp > frames[0].timestamp
    output = tmp_path / "evaluation"
    summary = evaluate(video_path, output, VisionPipeline(config, FakePoseBackend([make_pose()])))
    assert summary["frames"] == 12 and summary["unique_tracks"] == 1
    assert summary["accuracy_metrics"] is None
    assert len((output / "results.csv").read_text().splitlines()) == 13
    assert json.loads((output / "summary.json").read_text())["frames"] == 12
    with VideoCapture(output / "annotated.mp4") as annotated:
        assert len(list(annotated)) == 12
