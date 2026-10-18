import threading
from types import SimpleNamespace

import numpy as np
import pytest

from app.api.server import ObservationStore
from app.capture.monitor import CameraMonitor
from app.fusion.pipeline import VisionPipeline
from tests.test_pipeline import FakePoseBackend


def test_running_camera_rejects_source_changes(config):
    monitor = CameraMonitor(config, None, ObservationStore(), threading.Lock())
    monitor.capture_thread = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(RuntimeError, match="Stop"):
        monitor.start(index=config.camera.index+1)


def test_stopping_discards_inflight_observation_and_releases_preview(config, make_pose):
    entered, release = threading.Event(), threading.Event()
    frame = np.zeros((480, 640, 3), np.uint8)
    observation = VisionPipeline(config, FakePoseBackend([make_pose()])).process(frame, 0, 0)
    closed = []

    def process(*_):
        entered.set()
        assert release.wait(3)
        return observation

    store = ObservationStore()
    monitor = CameraMonitor(config, None, store, threading.Lock())
    monitor.pipeline = SimpleNamespace(process=process, close=lambda: closed.append(True))
    monitor.latest = 0, frame, 0
    monitor.inference_thread = threading.Thread(target=monitor._infer)
    monitor.inference_thread.start()
    assert entered.wait(3)
    monitor.stop_event.set()
    store.clear("stopping")
    release.set()
    monitor.inference_thread.join(3)
    assert not monitor.inference_thread.is_alive()
    assert store.snapshot() is None and store.camera_status == "stopped"
    assert monitor.latest is None and closed


def test_browser_launcher_honors_yaml_and_keeps_camera_id_consistent(config, monkeypatch):
    import sys

    from app.monitor import main

    config.camera.index = 2
    config.pose.image_size = 640
    calls = []
    monkeypatch.setenv("SIDP_CONFIG", "custom.yaml")
    monkeypatch.setattr("app.monitor.load_config", lambda path: calls.append(path) or config)
    monkeypatch.setattr("app.api.server.create_app", lambda cfg: cfg)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda app, **_: calls.append(app)))
    main(["--no-camera"])
    assert calls[0] == "custom.yaml"
    assert calls[1].camera.id == "webcam_2" and calls[1].pose.image_size == 640
    assert not calls[1].api.camera_enabled
