"""Validated YAML configuration; all experimental thresholds live in configs/."""
import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Unit = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[float, Field(gt=0)]


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class CameraConfig(SettingsModel):
    """Subsystem 1 video source: the Tapo C210 RTSP stream or a local webcam."""
    source: Literal["rtsp", "webcam"] = "webcam"
    index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    id: str
    host: str = "172.20.10.4"
    username: str = ""
    password: str = ""
    rtsp_port: int = Field(default=554, gt=0, le=65535)
    stream: str = "stream2"
    rtsp_transport: Literal["tcp", "udp"] | None = None
    reconnect_seconds: Positive = 2.0
    frame_timeout_seconds: Positive = 10.0

    @property
    def rtsp_url(self) -> str:
        credentials = f"{self.username}:{self.password}@" if self.username else ""
        return f"rtsp://{credentials}{self.host}:{self.rtsp_port}/{self.stream}"

    @property
    def display_address(self) -> str:
        """Source description that never exposes credentials to the dashboard."""
        if self.source == "rtsp":
            return f"rtsp://{self.host}:{self.rtsp_port}/{self.stream}"
        return f"webcam {self.index}"


class PTZConfig(SettingsModel):
    """Local ONVIF pan/tilt service of the Tapo C210 (credentials shared with the camera)."""
    enabled: bool = True
    port: int = Field(default=2020, gt=0, le=65535)
    pan_speed: Unit = 0.4
    tilt_speed: Unit = 0.3
    step_seconds: Positive = 0.4
    patrol_sweep_seconds: Positive = 1.0
    patrol_dwell_seconds: float = Field(default=0.4, ge=0)
    settle_seconds: float = Field(default=0.6, ge=0)


class DatasetConfig(SettingsModel):
    """Supervisor posture-dataset collector: clean frames saved without any overlay."""
    directory: str = "dataset"
    classes: list[str] = Field(default_factory=lambda: ["standing", "sitting", "bending", "falling_lying"],
                               min_length=1, max_length=9)
    record_hz: Positive = 2.0
    jpeg_quality: int = Field(default=95, ge=50, le=100)


class ActivityConfig(SettingsModel):
    """Activity layer on top of the geometric posture decision (walking, inactivity, fall)."""
    walking_min_velocity: Positive = 0.12
    inactivity_seconds: Positive = 20.0
    fall_transition_seconds: Positive = 2.0
    recovery_seconds: Positive = 1.0


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
    # Bending: torso inclined past this angle while the legs stay under the hips.
    bending_min_torso_angle: float = 35.0
    # Vertical hip-to-knee drop below this fraction of torso length means thighs are horizontal (seated).
    sitting_max_thigh_ratio: Positive = 0.65
    # Vertical hip-to-ankle drop above this fraction of torso length means the legs are upright.
    upright_min_leg_ratio: Positive = 0.8


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
    ptz: PTZConfig = Field(default_factory=PTZConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    secondary: SecondaryConfig = Field(default_factory=SecondaryConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    risk: RiskModelConfig = Field(default_factory=RiskModelConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)


ENVIRONMENT_OVERRIDES = {"SIDP_CAMERA_HOST": "host", "SIDP_CAMERA_USER": "username",
                         "SIDP_CAMERA_PASSWORD": "password", "SIDP_CAMERA_STREAM": "stream"}


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
    for variable, field in ENVIRONMENT_OVERRIDES.items():
        if os.getenv(variable):
            values["camera"][field] = os.environ[variable]
    return AppConfig.model_validate(values)

