"""Pydantic models for MQTT input and output messages."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PostureEventIn(BaseModel):
    """A posture event published by S2 on posture/events."""

    event_id: str
    timestamp: datetime
    camera_id: str
    person_id: str
    posture: str
    confidence: float = Field(ge=0.0, le=1.0)
    bend_angle_deg: float | None = None
    zone: str | None = None


class PostureRejectedError(ValueError):
    """Raised when a posture is not in the configured allowed list."""


def validate_posture(event: PostureEventIn, allowed_postures: list[str]) -> None:
    """Check the event's posture against the configured allowed list.

    Kept separate from PostureEventIn because the allowed posture list is
    still being agreed with the S2 team and must come from config, not a
    hard-coded enum on the model.
    """
    if event.posture not in allowed_postures:
        raise PostureRejectedError(
            f"posture '{event.posture}' is not in allowed postures {allowed_postures}"
        )


class ClassifiedEvent(BaseModel):
    """A scored, tiered event published on alert/events and written to the sink."""

    alert_id: str
    event_id: str
    person_id: str
    camera_id: str
    zone: str | None = None
    posture: str
    tier: Literal["safe", "warning", "unsafe", "alert"]
    risk_score: int = Field(ge=0, le=100)
    duration_s: float
    reasons: list[str]
    detected_at: datetime
    dispatched_at: datetime


class AudioDeviceSelect(BaseModel):
    """Payload for alert/audio/select: dashboard picks a speaker."""

    device_id: int


class CameraStatus(BaseModel):
    """Camera monitored/unmonitored status published on alert/status/cameras."""

    camera_id: str
    status: Literal["monitored", "unmonitored"]
    last_seen: datetime | None = None
