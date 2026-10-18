"""Validated YAML configuration; all experimental thresholds live in configs/."""
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Unit = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[float, Field(gt=0)]


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class CameraConfig(SettingsModel):
    index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    id: str


class PoseConfig(SettingsModel):
    model: str
    device: str
    confidence: Unit
    keypoint_confidence: Unit
    image_size: int = Field(gt=0)
    cpu_threads: int = Field(default=4, ge=1, le=32)
    half: bool = False


class TrackerConfig(SettingsModel):
    backend: Literal["bytetrack", "centroid"] = "centroid"
    stale_seconds: Positive
    min_iou: Unit
    max_centroid_distance: Positive
    iou_weight: Unit
    high_threshold: Unit = 0.35
    low_threshold: Unit = 0.1
    match_threshold: Unit = 0.8
    ambiguity_iou: Unit = 0.55


class SecondaryConfig(SettingsModel):
    enabled: bool = True
    model: str = "models/human-fall-detection.pt"
    image_size: int = 640
    confidence_floor: Unit = 0.01
    interval_seconds: Positive = 0.2
    max_age_seconds: Positive = 0.4
    min_iou: Unit = 0.2


class EvidenceConfig(SettingsModel):
    model: str | None = "models/event_classifier.json"
    window_seconds: Positive = 1.2
    min_coverage: Unit = 0.5
    persistence_seconds: Positive = 0.4
    recovery_seconds: Positive = 1.0


class RiskModelConfig(SettingsModel):
    enabled: bool = True
    directory: str = "models/fall-risk"
    sample_hz: Positive = 30.0
    max_gap_seconds: Positive = 0.5
    min_coverage: Unit = 0.45


class SceneConfig(SettingsModel):
    enabled: bool = True
    directory: str = "models/relateanything"
    interval_seconds: Positive = 1.0
    max_age_seconds: Positive = 2.5
    confidence: Unit = 0.56
    cpu_threads: int = Field(default=2, ge=1, le=8)


class TemporalConfig(SettingsModel):
    capacity: int = Field(ge=60)
    posture_window: int = Field(ge=2)
    fall_window: int = Field(ge=2)
    sample_hz: Positive
    max_gap_seconds: Positive
    min_coverage: Unit
    min_valid_ratio: Unit
    inference_interval_seconds: Positive
    stationary_velocity: Positive
    model_manifest: str | None = None

    @model_validator(mode="after")
    def enough_capacity(self):
        if self.capacity < max(self.posture_window, self.fall_window):
            raise ValueError("buffer capacity must cover both model windows")
        return self


class ScoreWeights(SettingsModel):
    downward: Unit
    rotation: Unit
    shrinking: Unit
    widening: Unit
    horizontal: Unit

    @model_validator(mode="after")
    def normalized(self):
        if abs(sum(self.model_dump().values()) - 1) > 1e-6:
            raise ValueError("score weights must sum to 1")
        return self


class StateConfig(SettingsModel):
    downward_velocity: Positive
    rotation_velocity: Positive
    height_reduction_velocity: float = Field(lt=0)
    aspect_ratio_velocity: Positive
    horizontal_angle: float = Field(ge=0, le=90)
    horizontal_ratio: Positive
    possible_probability: Unit
    falling_probability: Unit
    possible_persistence_seconds: Positive
    falling_persistence_seconds: Positive
    lying_persistence_seconds: Positive
    recovery_seconds: Positive
    episode_timeout_seconds: Positive
    score_weights: ScoreWeights


class PostureConfig(SettingsModel):
    min_coverage: Unit
    standing_max_torso_angle: float
    standing_min_knee_angle: float
    sitting_max_torso_angle: float
    sitting_max_knee_angle: float
    sitting_max_hip_angle: float
    lying_min_torso_angle: float
    lying_min_aspect_ratio: Positive
    smoothing_seconds: Positive
    history_seconds: Positive


class ApiConfig(SettingsModel):
    camera_enabled: bool
    websocket_hz: Positive
    max_image_bytes: int = Field(gt=0)
    max_image_pixels: int = Field(gt=0)
    preview_hz: Positive = 20.0
    inference_hz: Positive = 24.0


class AppConfig(SettingsModel):
    camera: CameraConfig
    pose: PoseConfig
    tracker: TrackerConfig
    temporal: TemporalConfig
    state: StateConfig
    api: ApiConfig
    posture: PostureConfig
    secondary: SecondaryConfig = Field(default_factory=SecondaryConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    risk: RiskModelConfig = Field(default_factory=RiskModelConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)


def load_config(path: str | Path | None = None, posture_path: str | Path | None = None) -> AppConfig:
    root = Path(__file__).resolve().parent.parent / "configs"
    source = Path(path) if path else root / "default.yaml"
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    psource = Path(posture_path) if posture_path else root / "posture.yaml"
    values["posture"] = yaml.safe_load(psource.read_text(encoding="utf-8"))
    if values["temporal"].get("model_manifest"):
        values["temporal"]["model_manifest"] = str(
            (source.parent / values["temporal"]["model_manifest"]).resolve()
        )
    return AppConfig.model_validate(values)
