"""Typed public contract. Coordinates are original-frame pixels, never model-input pixels."""
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1)]
COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Posture(str, Enum):
    STANDING = "STANDING"
    SITTING = "SITTING"
    BENDING = "BENDING"
    LYING = "LYING"
    UNKNOWN = "UNKNOWN"


class Activity(str, Enum):
    """Monitored activities (SIDP brief): posture plus motion/time context."""
    STANDING = "STANDING"
    SITTING = "SITTING"
    WALKING = "WALKING"
    BENDING = "BENDING"
    LYING = "LYING"
    FALL = "FALL"
    INACTIVE = "INACTIVE"
    UNKNOWN = "UNKNOWN"


class FallState(str, Enum):
    NORMAL = "NORMAL"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    FALLING = "FALLING"
    FALLEN = "FALLEN"


class Keypoint(Schema):
    index: int = Field(ge=0, le=16)
    x: float
    y: float
    confidence: Probability
    valid: bool


class BoundingBox(Schema):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def ordered(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must have positive width and height")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class PersonPose(Schema):
    bbox: BoundingBox
    detection_confidence: Probability
    keypoints: list[Keypoint] = Field(min_length=17, max_length=17)

    @model_validator(mode="after")
    def coco_order(self):
        if [k.index for k in self.keypoints] != list(range(17)):
            raise ValueError("keypoints must follow the 17 COCO indices exactly")
        return self


class GeometryFeatures(Schema):
    shoulder_midpoint: tuple[float, float] | None = None
    hip_midpoint: tuple[float, float] | None = None
    ankle_midpoint: tuple[float, float] | None = None
    left_knee_angle: float | None = None
    right_knee_angle: float | None = None
    left_hip_angle: float | None = None
    right_hip_angle: float | None = None
    torso_vector: tuple[float, float] | None = None
    torso_angle: float | None = None
    bbox_width: float
    bbox_height: float
    aspect_ratio: float
    centroid: tuple[float, float]
    head_to_hip_vertical_displacement: float | None = None
    hip_height: float | None = None
    shoulder_height: float | None = None
    confidence_coverage: Probability


class PostureResult(Schema):
    label: Posture
    confidence: Probability | None
    reason: str
    source: str = "geometric"
    features: GeometryFeatures


class MotionFeatures(Schema):
    vertical_velocity: float | None = None
    centroid_velocity: float | None = None
    hip_vertical_velocity: float | None = None
    shoulder_vertical_velocity: float | None = None
    torso_angle: float | None = None
    torso_angular_velocity: float | None = None
    bbox_height_velocity: float | None = None
    aspect_ratio: float
    aspect_ratio_velocity: float | None = None
    joint_velocity: float | None = None
    motion_energy: float | None = None
    reliable: bool = False
    delta_seconds: float | None = None


class FallInference(Schema):
    probability: Probability | None
    model_name: str
    window_frames: int = Field(ge=0)
    valid_keypoint_ratio: Probability
    calibrated: bool = False
    status: str
    latency_ms: float = Field(default=0, ge=0)
    raw_score: Probability | None = None
    threshold: Probability | None = None
    target: str = "current fall event or aftermath, not future risk"
    calibration_dataset: str | None = None


class AppearanceEvidence(Schema):
    model_name: str = "human-fall-detection"
    status: str = "unavailable"
    fallen_confidence: Probability | None = None
    sampled_at: float | None = None
    age_seconds: float | None = None
    latency_ms: float = Field(default=0, ge=0)
    matched_iou: Probability | None = None
    calibrated: Literal[False] = False
    posture: Posture | None = None
    posture_confidence: Probability | None = None


class SceneEvidence(Schema):
    model_name: str = "RelateAnything ViT-S+"
    status: str = "warming_up"
    label: str | None = None
    predicate: str | None = None
    object_label: str | None = None
    confidence: Probability | None = None
    sampled_at: float | None = None
    age_seconds: float | None = None
    latency_ms: float = Field(default=0, ge=0)


class RiskAssessment(Schema):
    model_name: str = "Tree + ST-GCN / TCN"
    status: str = "unavailable"
    state: Literal["UNASSESSED", "NORMAL", "PRE_FALL", "FALL"] = "UNASSESSED"
    advisory: Literal["PRE_FALL", "FALL"] | None = None
    confidence: Probability | None = None
    prefall_confidence: Probability | None = None
    fall_confidence: Probability | None = None
    tree_scores: dict[str, Probability] = Field(default_factory=dict)
    fusion_scores: dict[str, Probability] = Field(default_factory=dict)
    window_frames: int = Field(default=0, ge=0)
    calibration_progress: Probability = 0
    latency_ms: float = Field(default=0, ge=0)
    future_horizon_seconds: float | None = None
    target: str = "Pre-fall pattern and current fall; no guaranteed future horizon"


class TrackedPerson(Schema):
    track_id: int = Field(gt=0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    detection_confidence: Probability
    keypoints: list[Keypoint] = Field(min_length=17, max_length=17)
    posture: Posture
    posture_confidence: Probability | None
    posture_reason: str
    posture_source: str
    fall_probability: Probability | None
    fall_inference: FallInference
    state: FallState
    motion: MotionFeatures
    geometry: GeometryFeatures
    movement_state: Literal["STATIONARY", "MOVING", "UNKNOWN"]
    pose_coverage: Probability = 0
    appearance: AppearanceEvidence = Field(default_factory=AppearanceEvidence)
    history_seconds: float = Field(default=0, ge=0)
    identity_uncertain: bool = False
    scene: SceneEvidence = Field(default_factory=SceneEvidence)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    activity_label: str = "Person detected"
    activity_confidence: Probability | None = None
    activity_source: str = "pose"
    activity: Activity = Activity.UNKNOWN
    activity_reason: str = ""
    inactive_seconds: float = Field(default=0, ge=0)


class VisionObservation(Schema):
    schema_version: Literal["2.0"] = "2.0"
    timestamp: datetime
    elapsed_seconds: float = Field(ge=0)
    frame_id: int = Field(ge=0)
    camera_id: str
    session_id: str
    fps: float = Field(ge=0)
    processing_latency_ms: float = Field(ge=0)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    timings_ms: dict[str, float]
    persons: list[TrackedPerson]
    detected_people: int = Field(default=0, ge=0)
    capture_to_publish_ms: float = Field(default=0, ge=0)
