from types import SimpleNamespace

import numpy as np
import pytest

from app.fusion.pipeline import VisionPipeline
from app.inference.appearance import FallenBox
from app.inference.scene import SceneWorker
from app.inference.temporal_fall import KerasTemporalBackend
from app.schemas import FallInference, RiskAssessment, SceneEvidence
from app.visualization.overlay import draw_overlay
from tests.test_pipeline import FakePoseBackend


def test_rgb_fallen_score_is_not_a_temporal_probability(config, make_pose):
    pose = make_pose("lying")
    rgb = SimpleNamespace(infer=lambda _: [FallenBox(pose.bbox, .9)])
    pipeline = VisionPipeline(config, FakePoseBackend([pose]), appearance_backend=rgb)
    person = pipeline.process(np.zeros((480, 640, 3), np.uint8), 0, 0).persons[0]
    assert person.appearance.fallen_confidence == .9
    assert person.fall_inference.raw_score == .9
    assert person.fall_probability is None
    assert not person.fall_inference.calibrated
    assert "RGB" in person.posture_reason


def test_explicit_temporal_backend_is_called_and_isolated_per_person(config, make_pose):
    buffers = []

    def infer(buffer, motion, posture):
        buffers.append(buffer)
        return FallInference(probability=None, raw_score=.8, model_name="retrained",
                             window_frames=1, valid_keypoint_ratio=1, status="uncalibrated")

    pipeline = VisionPipeline(config, FakePoseBackend([
        make_pose(offset=(-150, 0)), make_pose(offset=(180, 0)),
    ]), fall_backend=SimpleNamespace(infer=infer))
    result = pipeline.process(np.zeros((480, 640, 3), np.uint8), 0, 0)
    assert len(buffers) == 2 and buffers[0] is not buffers[1]
    assert all(p.fall_inference.model_name == "retrained" for p in result.persons)


def test_temporal_cadence_never_reuses_a_score_for_missing_torso(config, make_pose):
    calls = []

    def infer(*_):
        calls.append(True)
        return FallInference(probability=None, raw_score=.8, model_name="retrained",
                             window_frames=60, valid_keypoint_ratio=1, status="uncalibrated")

    pose = FakePoseBackend([make_pose()])
    pipeline = VisionPipeline(config, pose, fall_backend=SimpleNamespace(infer=infer))
    frame = np.zeros((480, 640, 3), np.uint8)
    pipeline.process(frame, 0, 0)
    pipeline.process(frame, .05, 1)
    assert len(calls) == 1
    pose.poses = [make_pose(missing=(5, 6))]
    person = pipeline.process(frame, .1, 2).persons[0]
    assert person.fall_inference.raw_score is None
    pose.poses = [make_pose()]
    pipeline.process(frame, .11, 3)
    assert len(calls) == 2


def test_uncalibrated_injected_backend_cannot_export_a_probability(config, make_pose):
    backend = SimpleNamespace(infer=lambda *_: FallInference(
        probability=.8, model_name="legacy raw score", window_frames=2,
        valid_keypoint_ratio=1, status="uncalibrated", calibrated=False,
    ))
    pipeline = VisionPipeline(config, FakePoseBackend([make_pose()]), fall_backend=backend)
    person = pipeline.process(np.zeros((480, 640, 3), np.uint8), 0, 0).persons[0]
    assert person.fall_probability is None and person.fall_inference.raw_score == .8


def test_scene_shutdown_reports_an_unfinished_worker():
    import threading

    worker = SceneWorker.__new__(SceneWorker)
    worker.condition = threading.Condition()
    worker.closed, worker.pending = False, None
    worker.thread = SimpleNamespace(join=lambda **_: None, is_alive=lambda: True)
    with pytest.raises(RuntimeError, match="did not stop"):
        worker.close()


def test_temporal_manifest_is_loaded_only_when_explicitly_selected(config, make_pose, monkeypatch):
    calls = []
    backend = SimpleNamespace(infer=lambda *_: None)
    monkeypatch.setattr("app.inference.temporal_fall.KerasTemporalBackend",
                        lambda path, cfg: calls.append(path) or backend)
    assert VisionPipeline(config, FakePoseBackend([])).fall_backend is None
    config.temporal.model_manifest = "explicit_manifest.json"
    assert VisionPipeline(config, FakePoseBackend([])).fall_backend is backend
    assert calls == ["explicit_manifest.json"]


def test_keras_output_remains_an_uncalibrated_raw_score(config, make_pose):
    backend = KerasTemporalBackend.__new__(KerasTemporalBackend)
    backend.config = config.temporal
    backend.manifest = {"model": "fall.keras"}
    backend.mean, backend.std = np.zeros(51), np.ones(51)
    backend.model = lambda *_, **__: np.array([[.85]])
    buffer = SimpleNamespace(window=lambda _: (np.zeros((60, 51)), 1.0))
    result = backend.infer(buffer, None, None)
    assert result.probability is None and result.raw_score == .85
    assert not result.calibrated


def test_risk_error_discards_mutated_person_state(config, make_pose):
    instances = []

    def new_track():
        value = object()
        instances.append(value)
        return value

    def infer(*_):
        raise RuntimeError("invalid model output")

    risk = SimpleNamespace(new_track=new_track, infer=infer)
    pipeline = VisionPipeline(config, FakePoseBackend([make_pose()]), risk_backend=risk)
    person = pipeline.process(np.zeros((480, 640, 3), np.uint8), 0, 0).persons[0]
    assert person.risk.state == "UNASSESSED" and person.risk.confidence is None
    assert len(instances) == 2
    assert pipeline.contexts[1].risk_track is instances[-1]


def test_scene_result_from_before_identity_reset_is_rejected():
    import threading

    worker = SceneWorker.__new__(SceneWorker)
    worker.condition = threading.Condition()
    worker.results = {1: SceneEvidence(status="matched", label="Sitting on chair", sampled_at=1)}
    worker.session, worker.error = "session", None
    worker.backend = SimpleNamespace(config=SimpleNamespace(max_age_seconds=2.5))
    assert worker.snapshot(1, 1.2, "session", min_timestamp=1.1).label is None
    assert worker.snapshot(1, 1.2, "session", min_timestamp=1).label == "Sitting on chair"
    assert worker.snapshot(1, 4, "session").status == "stale"
    assert worker.snapshot(1, 1.2, "different-session").label is None


def test_overlay_accepts_unavailable_branch_scores(config, make_pose):
    frame = np.zeros((480, 640, 3), np.uint8)
    result = VisionPipeline(config, FakePoseBackend([make_pose()])).process(frame, 0, 0)
    result.persons[0].risk = RiskAssessment(status="ready", state="UNASSESSED")
    assert draw_overlay(frame, result).shape == frame.shape


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -1])
def test_invalid_timestamp_does_not_create_person_state(config, timestamp):
    pipeline = VisionPipeline(config, FakePoseBackend([]))
    with pytest.raises(ValueError, match="timestamp"):
        pipeline.process(np.zeros((480, 640, 3), np.uint8), timestamp, 0)
    assert not pipeline.contexts
