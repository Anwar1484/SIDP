import cv2
import numpy as np
import pytest

from app.capture.webcam import WebcamCapture
from app.fusion.pipeline import VisionPipeline
from app.main import main
from tests.test_pipeline import FakePoseBackend


def test_camera_release_on_read_failure(config, monkeypatch):
    class Camera:
        released = False

        def isOpened(self):
            return True

        def set(self, *_):
            pass

        def read(self):
            return False, None

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: camera)
    with pytest.raises(RuntimeError, match="stopped returning"), WebcamCapture(config.camera) as cap:
        cap.read()
    assert camera.released


def test_q_exits_and_releases_camera_and_windows(config, make_pose, monkeypatch):
    class Camera:
        released = False
        resolution = (640, 480)

        def __enter__(self):
            return self

        def read(self):
            return np.zeros((480, 640, 3), dtype=np.uint8), 0

        def __exit__(self, *_):
            self.released = True

    camera = Camera()
    backend = FakePoseBackend([make_pose()])
    backend.warmup = lambda *_: None
    monkeypatch.setattr("app.main.WebcamCapture", lambda _: camera)
    monkeypatch.setattr("app.main.load_config", lambda *_: config)
    monkeypatch.setattr("app.main.VisionPipeline", lambda cfg: VisionPipeline(cfg, backend))
    monkeypatch.setattr("app.main.startup_info", lambda *_: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _: ord("q"))
    windows_closed = []
    models_loaded = []
    pipeline_closed = []
    monkeypatch.setattr(VisionPipeline, "load_optional_models", lambda _: models_loaded.append(True))
    monkeypatch.setattr(VisionPipeline, "close", lambda _: pipeline_closed.append(True))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: windows_closed.append(True))
    main(["--max-frames", "10"])
    assert camera.released and windows_closed and models_loaded and pipeline_closed
