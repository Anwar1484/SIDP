"""Portable learned coefficients and Platt calibration; no pickle or arbitrary fusion."""
import json
from pathlib import Path

import numpy as np

from app.schemas import FallInference
from app.temporal.event_features import FEATURE_NAMES


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


class EventClassifier:
    def __init__(self, path: str | Path):
        self.artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if self.artifact.get("feature_names") != FEATURE_NAMES or self.artifact.get("format") != 1:
            raise ValueError("Event classifier feature contract mismatch")
        self.selected = self.artifact["selected"]
        for model in self.artifact["models"].values():
            n = len(model["indices"])
            if any(len(model[key]) != n for key in ("medians", "mean", "scale", "coef")):
                raise ValueError("Invalid model coefficient dimensions")
            parameters = np.r_[model["medians"], model["mean"], model["scale"], model["coef"],
                               model["intercept"], model["platt_coef"], model["platt_intercept"], model["threshold"]]
            if not np.isfinite(parameters).all():
                raise ValueError("Non-finite model parameters")
            if min(model["scale"]) <= 0 or not 0 <= model["threshold"] <= 1:
                raise ValueError("Invalid model scale/threshold")

    def infer(self, features: np.ndarray, rgb_available: bool, coverage: float, frames: int) -> FallInference:
        name = self.selected
        if "rgb" in name and not rgb_available:
            name = "temporal_pose"
        model = self.artifact["models"][name]
        x = np.asarray(features)[model["indices"]]
        x = np.where(np.isfinite(x), x, model["medians"])
        z = (x - model["mean"]) / model["scale"]
        logit = float(z @ np.asarray(model["coef"]) + model["intercept"])
        return FallInference(
            probability=float(sigmoid(model["platt_coef"] * logit + model["platt_intercept"])),
            raw_score=float(sigmoid(logit)), model_name=f"urfd_{name}_logistic_v1",
            window_frames=frames, valid_keypoint_ratio=coverage, calibrated=True,
            status="ready", threshold=model["threshold"], calibration_dataset="URFD camera 0 / calibration split",
        )
