"""Adapt pinned pretrained tree/ST-GCN/TCN inference to independent YOLO tracks.

The upstream code stays in the local model cache; it is not redistributed here.
Fixed-rate causal resampling preserves the 30 Hz, 15-frame training convention.
Warm-start heuristics from that code never become model confidence in this API.
"""
import copy
import sys
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from app.config import RiskModelConfig
from app.model_assets import verify_assets
from app.runtime import ROOT
from app.schemas import PersonPose, RiskAssessment


@dataclass
class RiskTrack:
    predictor: object
    previous: np.ndarray | None = None
    previous_time: float | None = None
    sampled_time: float | None = None
    sample_index: int = 0
    latest: RiskAssessment = field(default_factory=RiskAssessment)


class PrefallBackend:
    def __init__(self, config: RiskModelConfig):
        verify_assets("models/fall-risk/", ROOT / config.directory)
        source = ROOT / config.directory / "source"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from fall_prediction.ensemble_predictor import DualModelFallPredictor
        from fall_prediction.pose import coco17_to_mediapipe_landmarks

        # HistGradientBoosting uses OpenMP. Bound it so webcam inference and
        # the background relation model retain CPU time on a laptop.
        from threadpoolctl import threadpool_limits
        self.thread_limit = threadpool_limits(limits=2, user_api="openmp")
        directory = ROOT / config.directory
        self.template = DualModelFallPredictor(
            directory / "yolo_tail60_prefall_accel_robust_classifier.joblib",
            directory / "skeleton_feature_fusion_tuned.pt",
        )
        self.adapter = coco17_to_mediapipe_landmarks
        self.config = config
        tree, fusion = self.template.tree_predictor, self.template.fusion_predictor
        tree.model.predict_proba(np.zeros((1, tree.window_size*len(tree.feature_columns)), np.float32))
        fusion.model.set_skeleton_window(np.zeros((5, fusion.window_size, 17), np.float32))
        fusion.model.predict_proba(np.zeros((1, fusion.window_size*len(fusion.feature_columns)), np.float32))

    def new_track(self) -> RiskTrack:
        # Weights are immutable; each person's windows, calibrator, smoothing
        # and wrapper input remain independent without copying network tensors.
        tree = self.template.tree_predictor.model
        network = self.template.fusion_predictor.model.model
        return RiskTrack(copy.deepcopy(self.template, {id(tree): tree, id(network): network}))

    def infer(self, track: RiskTrack, pose: PersonPose, timestamp: float,
              frame_size: tuple[int, int], identity_uncertain: bool = False) -> RiskAssessment:
        start = perf_counter()
        joints = np.array([(k.x, k.y, k.confidence if k.valid else 0.0) for k in pose.keypoints],
                          dtype=np.float32)
        coverage = float(np.count_nonzero(joints[:, 2]) / 17)
        torso_ok = all(pose.keypoints[k].valid for k in (5, 6, 11, 12))
        if identity_uncertain or coverage < self.config.min_coverage or not torso_ok:
            track.predictor.reset()
            track.previous = None
            track.previous_time = track.sampled_time = None
            track.sample_index = 0
            track.latest = RiskAssessment(status="identity_uncertain" if identity_uncertain else "pose_incomplete")
            return track.latest
        if track.previous_time is not None and timestamp - track.previous_time > self.config.max_gap_seconds:
            track.predictor.reset()
            track.previous = None
            track.previous_time = track.sampled_time = None
            track.sample_index = 0
            track.latest = RiskAssessment(status="collecting")
        step = 1 / self.config.sample_hz
        next_time = timestamp if track.sampled_time is None else track.sampled_time + step
        prediction = None
        while next_time <= timestamp + 1e-7:
            sample = joints.copy()
            if track.previous is not None and track.previous_time is not None:
                fraction = np.clip((next_time-track.previous_time) / max(timestamp-track.previous_time, 1e-8), 0, 1)
                sample[:, :2] = track.previous[:, :2]*(1-fraction) + joints[:, :2]*fraction
                sample[:, 2] = np.minimum(track.previous[:, 2], joints[:, 2])
            landmarks = self.adapter(sample[:, :2], sample[:, 2], *frame_size)
            prediction = track.predictor.predict(landmarks, track.sample_index, next_time)
            track.sample_index += 1
            track.sampled_time = next_time
            next_time += step
        track.previous, track.previous_time = joints, timestamp
        tree = track.predictor.tree_predictor
        fusion = track.predictor.fusion_predictor
        calibrator = tree._standing_calibrator
        progress = (min(1.0, calibrator.collected_frames / calibrator.baseline_frames)
                    if calibrator is not None else 1.0)
        frames = min(len(tree._window), len(fusion._window))
        if calibrator is not None and not calibrator.ready:
            track.latest = RiskAssessment(status="calibrating", calibration_progress=progress, window_frames=frames)
        elif frames < tree.window_size:
            track.latest = RiskAssessment(status="collecting", calibration_progress=progress, window_frames=frames)
        elif prediction is not None:
            state_map = {"Normal": "NORMAL", "Pre-fall": "PRE_FALL", "Fall": "FALL", "Unknown": "UNASSESSED"}
            # Keep each branch's actual posterior. The decision policy can use
            # agreement/persistence without inventing a blended probability.
            ts = {k: float(np.clip(v, 0, 1)) for k, v in tree._last_state_probabilities.items()}
            fs = {k: float(np.clip(v, 0, 1)) for k, v in fusion._last_state_probabilities.items()}
            state = state_map.get(prediction.alert_state or prediction.state, "UNASSESSED")
            advisory = state_map.get(prediction.advisory_state)
            key = {"NORMAL": "Normal", "PRE_FALL": "Pre-fall", "FALL": "Fall"}.get(state)
            # The authoritative branch is tree; confirmed fusion-only alerts
            # carry the fusion confidence and preserve both original scores.
            scores = fs if prediction.decision_tier == "critical-fusion-alert" else ts
            track.latest = RiskAssessment(
                status="ready", state=state,
                advisory=advisory if advisory in ("PRE_FALL", "FALL") else None,
                confidence=scores.get(key) if key else None,
                prefall_confidence=fs.get("Pre-fall"), fall_confidence=ts.get("Fall"),
                tree_scores=ts, fusion_scores=fs, window_frames=frames,
                calibration_progress=progress,
            )
        return track.latest.model_copy(update={"latency_ms": (perf_counter()-start)*1000})
