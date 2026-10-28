"""Loads and validates config/config.yaml into typed models."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class MqttConfig(BaseModel):
    host: str
    port: int
    client_id: str


class DebounceConfig(BaseModel):
    window_s: float
    min_confirmations: int
    min_confidence: float
    resolve_after_s: float
    safe_postures: list[str] = Field(default_factory=lambda: ["sitting", "standing", "walking"])


class ScoringConfig(BaseModel):
    posture_severity: dict[str, float]
    duration_weight: float
    duration_cap_s: float
    force_alert_postures: list[str]


class TiersConfig(BaseModel):
    safe: float
    warning: float
    unsafe: float
    alert: float

    def classify(self, score: float) -> str:
        """Map a risk score to the highest tier whose lower bound it meets."""
        ordered = sorted(
            [("alert", self.alert), ("unsafe", self.unsafe), ("warning", self.warning), ("safe", self.safe)],
            key=lambda pair: pair[1],
            reverse=True,
        )
        for tier, lower_bound in ordered:
            if score >= lower_bound:
                return tier
        return "safe"


class DispatchConfig(BaseModel):
    cooldown_s: float
    status_min_interval_s: float


class WatchdogConfig(BaseModel):
    camera_timeout_s: float = 5.0


class AudioConfig(BaseModel):
    default_device: int | None = None
    messages: dict[str, str] = Field(default_factory=dict)


class SinkConfig(BaseModel):
    type: str = "sqlite"
    sqlite_path: str = "data/events.db"


class MlConfig(BaseModel):
    enabled: bool = True
    model_path: str = "models/risk_model.npz"


class NotifyConfig(BaseModel):
    enabled: bool = True
    tiers: list[str] = Field(default_factory=lambda: ["alert"])
    postures: list[str] = Field(default_factory=lambda: ["fall", "lying"])  # empty = any posture


class Config(BaseModel):
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    ml: MlConfig = Field(default_factory=MlConfig)
    mqtt: MqttConfig
    postures: list[str]
    debounce: DebounceConfig
    scoring: ScoringConfig
    tiers: TiersConfig
    resident_profiles: dict[str, float] = Field(default_factory=dict)
    dispatch: DispatchConfig
    watchdog: WatchdogConfig
    audio: AudioConfig
    sink: SinkConfig

    def resident_multiplier(self, person_id: str) -> float:
        return self.resident_profiles.get(person_id, 1.0)


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Read the YAML config file and validate it into a Config model."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config.model_validate(raw)
