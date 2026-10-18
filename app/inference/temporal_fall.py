"""Temporal backend contract, honest geometric fallback, optional retrained Keras model."""
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from app.config import StateConfig, TemporalConfig
from app.posture.normalization import NORMALIZATION_VERSION
from app.schemas import FallInference, MotionFeatures, Posture
from app.temporal.buffer import TemporalBuffer

UNAVAILABLE = "TEMPORAL MODEL UNAVAILABLE / RETRAINING REQUIRED"


class TemporalFallBackend(Protocol):
    def infer(self, buffer: TemporalBuffer, motion: MotionFeatures, posture: Posture) -> FallInference: ...


class GeometricTemporalBackend:
    def __init__(self, config: StateConfig):
        self.config = config

    def infer(self, buffer: TemporalBuffer, motion: MotionFeatures, posture: Posture) -> FallInference:
        start = perf_counter()
        cfg, weights = self.config, self.config.score_weights
        ratio = buffer.samples[-1].geometry.confidence_coverage if buffer.samples else 0

        def component(value: float | None, scale: float) -> float:
            return float(np.clip((value or 0) / scale, 0, 1))

        score = None
        if motion.reliable:
            score = (weights.downward * component(motion.hip_vertical_velocity, cfg.downward_velocity)
                     + weights.rotation * component(abs(motion.torso_angular_velocity or 0), cfg.rotation_velocity)
                     + weights.shrinking * component(-(motion.bbox_height_velocity or 0),
                                                       -cfg.height_reduction_velocity)
                     + weights.widening * component(motion.aspect_ratio_velocity, cfg.aspect_ratio_velocity)
                     + weights.horizontal * float(posture == Posture.LYING))
        return FallInference(probability=None, raw_score=score, model_name="geometric_temporal_v1",
                             window_frames=min(2, len(buffer.samples)), valid_keypoint_ratio=ratio,
                             status=UNAVAILABLE + "; uncalibrated geometric score" if score is not None
                             else UNAVAILABLE + "; insufficient continuous pose evidence",
                             latency_ms=(perf_counter() - start) * 1000)


class KerasTemporalBackend:
    """Only accepts the versioned 51-feature YOLO26 retraining artifact.

    Shipped reference H5 files lack this contract and cannot be enabled accidentally.
    Training mean/std are plain arrays, avoiding pickle and never fitted here.
    """
    def __init__(self, manifest_path: str, config: TemporalConfig):
        import tensorflow as tf
        source = Path(manifest_path)
        manifest = json.loads(source.read_text(encoding="utf-8"))
        required = {"normalization": NORMALIZATION_VERSION, "pose_family": "YOLO26",
                    "features": 51, "frames": config.fall_window, "sample_hz": config.sample_hz,
                    "task": "fall"}
        for name, expected in required.items():
            if manifest.get(name) != expected:
                raise ValueError(f"temporal manifest {name}: expected {expected!r}")
        self.config, self.manifest = config, manifest
        self.model = tf.keras.models.load_model(source.parent / manifest["model"], compile=False)
        if tuple(self.model.input_shape[1:]) != (config.fall_window, 51) or self.model.output_shape[-1] != 1:
            raise ValueError("temporal model shape disagrees with manifest")
        stats = np.load(source.parent / manifest["scaler"], allow_pickle=False)
        self.mean, self.std = stats["mean"], stats["std"]
        if (self.mean.shape != (51,) or self.std.shape != (51,) or not np.isfinite(self.mean).all()
                or not np.isfinite(self.std).all() or np.any(self.std <= 0)):
            raise ValueError("invalid saved normalization statistics")

    def infer(self, buffer: TemporalBuffer, motion: MotionFeatures, posture: Posture) -> FallInference:
        start = perf_counter()
        window, ratio = buffer.window(self.config.fall_window)
        raw_score = None
        if window is not None:
            value = float(np.asarray(self.model(((window-self.mean)/self.std)[None], training=False))[0, 0])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError("temporal model returned an invalid score")
            raw_score = value
        return FallInference(probability=None, raw_score=raw_score, model_name=self.manifest["model"],
                             window_frames=self.config.fall_window if window is not None else len(buffer.samples),
                             valid_keypoint_ratio=ratio, calibrated=False,
                             status="trained model; calibration not established" if raw_score is not None
                             else "warming up / insufficient valid continuous window",
                             latency_ms=(perf_counter()-start)*1000)
