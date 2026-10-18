"""Persistence applied only to a trained model's validation-selected threshold."""
from app.config import EvidenceConfig
from app.schemas import FallInference, FallState


class EvidenceStateMachine:
    def __init__(self, config: EvidenceConfig):
        self.config = config
        self.state = FallState.NORMAL
        self.high_since: float | None = None
        self.low_since: float | None = None

    def update(self, timestamp: float, evidence: FallInference) -> FallState:
        if evidence.probability is None or evidence.threshold is None or not evidence.calibrated:
            self.high_since = self.low_since = None
            self.state = FallState.NORMAL
            return self.state
        if evidence.probability >= evidence.threshold:
            self.low_since = None
            self.high_since = timestamp if self.high_since is None else self.high_since
            self.state = (FallState.FALLEN if timestamp-self.high_since >= self.config.persistence_seconds
                          else FallState.POSSIBLE_FALL)
        else:
            self.high_since = None
            self.low_since = timestamp if self.low_since is None else self.low_since
            if self.state != FallState.FALLEN or timestamp-self.low_since >= self.config.recovery_seconds:
                self.state = FallState.NORMAL
        return self.state
