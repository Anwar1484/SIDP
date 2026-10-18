"""Explainable classification plus time-based smoothing per person."""
from collections import deque

from app.config import PostureConfig
from app.schemas import GeometryFeatures, Posture, PostureResult


class GeometricPostureClassifier:
    def __init__(self, config: PostureConfig):
        self.config = config

    def classify(self, features: GeometryFeatures) -> PostureResult:
        cfg, f = self.config, features
        label, reason = Posture.UNKNOWN, "insufficient confident geometry"
        knees = [v for v in (f.left_knee_angle, f.right_knee_angle) if v is not None]
        hips = [v for v in (f.left_hip_angle, f.right_hip_angle) if v is not None]
        if f.confidence_coverage >= cfg.min_coverage and f.torso_angle is not None:
            if f.torso_angle >= cfg.lying_min_torso_angle and f.aspect_ratio >= cfg.lying_min_aspect_ratio:
                label, reason = Posture.LYING, "torso approximately horizontal and bbox width/height increased"
            elif f.torso_angle >= cfg.bending_min_torso_angle:
                label, reason = Posture.BENDING, f"torso inclined {f.torso_angle:.0f} deg from vertical"
            elif (f.torso_angle <= cfg.sitting_max_torso_angle and knees
                  and min(knees) <= cfg.sitting_max_knee_angle and hips
                  and min(hips) <= cfg.sitting_max_hip_angle):
                label, reason = Posture.SITTING, "knees and hips flexed with torso approximately vertical"
            elif (f.torso_angle <= cfg.standing_max_torso_angle and knees
                  and min(knees) >= cfg.standing_min_knee_angle):
                label, reason = Posture.STANDING, "torso vertical with extended knees"
            else:
                reason = "geometry does not satisfy a selected posture"
        # Geometry is an interpretable label, not a trained probabilistic classifier.
        if label == Posture.UNKNOWN and (f.hip_midpoint is None or not knees or f.torso_angle is None):
            reason = "Step back for full-body posture detection"
        return PostureResult(label=label, confidence=None,
                             reason=reason, features=f)


class PostureSmoother:
    def __init__(self, config: PostureConfig):
        self.config = config
        self.history: deque[tuple[float, PostureResult]] = deque()
        self.current: PostureResult | None = None
        self.candidate: Posture | None = None
        self.since = 0.0

    def update(self, timestamp: float, result: PostureResult) -> PostureResult:
        self.history.append((timestamp, result))
        while self.history and timestamp - self.history[0][0] > self.config.history_seconds:
            self.history.popleft()
        if result.label == Posture.UNKNOWN:
            self.current, self.candidate = result, None
            return result
        if self.current is None or self.current.label == Posture.UNKNOWN:
            self.current = result
        elif result.label != self.current.label:
            if self.candidate != result.label:
                self.candidate, self.since = result.label, timestamp
            if timestamp - self.since >= self.config.smoothing_seconds:
                self.current, self.candidate = result, None
        else:
            self.current, self.candidate = result, None
        # Retain the selected label but attach current-frame confidence and geometry.
        return result.model_copy(update={"label": self.current.label,
                                         "reason": result.reason if result.label == self.current.label
                                         else "temporal smoothing: awaiting persistent posture change"})
