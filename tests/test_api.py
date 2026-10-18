import json
import struct

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.server import ObservationStore, create_app
from app.fusion.pipeline import VisionPipeline
from tests.test_pipeline import FakePoseBackend


def test_image_endpoint_isolated_requests_and_limits(config, make_pose):
    config.api.camera_enabled = False
    config.api.max_image_bytes = 20000
    backend = FakePoseBackend([make_pose()])
    app = create_app(config, backend)
    _, encoded = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/status").json()["camera"] == "disabled"
        responses = [client.post("/infer/image", content=encoded.tobytes(), headers={"content-type": "image/jpeg"})
                     for _ in range(2)]
        assert all(r.status_code == 200 for r in responses)
        a, b = [r.json() for r in responses]
        assert a["session_id"] != b["session_id"]
        assert a["persons"][0]["fall_probability"] is None
        assert client.post("/infer/image", content=b"bad", headers={"content-type": "image/png"}).status_code == 400
        assert client.post("/infer/image", content=b"x"*20001,
                           headers={"content-type": "image/jpeg"}).status_code == 413
        assert client.post("/infer/image", content=b"x", headers={"content-type": "text/plain"}).status_code == 415


def test_websocket_only_sends_observation(config, make_pose):
    config.api.camera_enabled = False
    backend = FakePoseBackend([make_pose()])
    store = ObservationStore()
    observation = VisionPipeline(config, backend).process(np.zeros((480, 640, 3), dtype=np.uint8), 0, 0)
    store.publish(observation)
    with TestClient(create_app(config, backend, store)) as client, client.websocket_connect("/ws/vision") as ws:
        payload = ws.receive_json()
        assert payload["schema_version"] == "2.0" and len(payload["persons"]) == 1
        assert "image" not in payload and "raw_frame" not in payload


def test_monitor_packet_keeps_image_and_observation_together(config, make_pose):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    backend = FakePoseBackend([make_pose()])
    store = ObservationStore()
    observation = VisionPipeline(config, backend).process(frame, 0, 7)
    store.publish(observation, frame)
    with TestClient(create_app(config, backend, store)) as client, client.websocket_connect("/ws/monitor") as ws:
        packet = ws.receive_bytes()
        length = struct.unpack("<I", packet[:4])[0]
        payload = json.loads(packet[4:4+length])
        image = cv2.imdecode(np.frombuffer(packet[4+length:], np.uint8), cv2.IMREAD_COLOR)
        assert payload["frame_id"] == 7 and image.shape == frame.shape
        assert payload == observation.model_dump(mode="json")


def test_local_controls_reject_foreign_origins_and_bad_camera_options(config):
    with TestClient(create_app(config, FakePoseBackend([]))) as client:
        assert client.post("/api/session/reset", headers={"origin": "https://unrelated.test"}).status_code == 403
        assert client.post("/api/camera/start", json={"index": -1}).status_code == 422
        assert client.post("/api/camera/start", json={"image_size": 417}).status_code == 422
        assert client.post("/api/session/reset").json() == {"status": "baseline_reset"}
        assert client.post("/api/camera/stop").json()["camera"] == "stopped"


def test_image_request_keeps_rgb_confidence_out_of_fall_probability(config, make_pose, monkeypatch):
    from types import SimpleNamespace

    from app.inference.appearance import FallenBox

    pose = make_pose("lying")

    def load_models(pipeline):
        pipeline.appearance_backend = SimpleNamespace(infer=lambda _: [FallenBox(pose.bbox, .92)])

    monkeypatch.setattr(VisionPipeline, "load_optional_models", load_models)
    _, encoded = cv2.imencode(".png", np.zeros((480, 640, 3), np.uint8))
    with TestClient(create_app(config, FakePoseBackend([pose]))) as client:
        person = client.post("/infer/image", content=encoded.tobytes(),
                             headers={"content-type": "image/png"}).json()["persons"][0]
        assert person["appearance"]["fallen_confidence"] == .92
        assert person["fall_inference"]["raw_score"] == .92
        assert person["fall_probability"] is None and person["state"] == "NORMAL"


def test_explicit_camera_configuration_overrides_environment(config, monkeypatch):
    monkeypatch.setenv("SIDP_CAMERA_ENABLED", "1")
    config.api.camera_enabled = False
    with TestClient(create_app(config, FakePoseBackend([]))) as client:
        assert client.get("/status").json()["camera"] == "disabled"
        models = client.get("/status").json()["models"]
        assert all(model.get("enabled") is False for model in models[1:])
