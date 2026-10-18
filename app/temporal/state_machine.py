"""Experimental evidence states; no caregiver alert decision or medical prediction."""
from app.config import StateConfig
from app.schemas import FallInference, FallState, MotionFeatures, Posture


class FallStateMachine:
    def __init__(self, config: StateConfig):
        self.config = config
        self.state = FallState.NORMAL
        self.timers: dict[str, float] = {}
        self.episode_start: float | None = None
        self.last_timestamp: float | None = None

    def reset_continuity(self):
        self.timers.clear()
        if self.state != FallState.FALLEN:
            self.state, self.episode_start = FallState.NORMAL, None

    def _persistent(self, name: str, evidence: bool, timestamp: float, duration: float) -> bool:
        if not evidence:
            self.timers.pop(name, None)
            return False
        self.timers.setdefault(name, timestamp)
        return timestamp - self.timers[name] >= duration

    def update(self, timestamp: float, posture: Posture, motion: MotionFeatures,
               fall: FallInference) -> FallState:
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("state timestamps must increase")
        self.last_timestamp = timestamp
        if not motion.reliable:
            self.reset_continuity()
            return self.state
        cfg = self.config
        downward = (motion.hip_vertical_velocity or 0) >= cfg.downward_velocity
        rotation = abs(motion.torso_angular_velocity or 0) >= cfg.rotation_velocity
        shrinking = (motion.bbox_height_velocity or 0) <= cfg.height_reduction_velocity
        horizontal = ((motion.torso_angle or 0) >= cfg.horizontal_angle
                      and motion.aspect_ratio >= cfg.horizontal_ratio)
        probability = fall.probability or 0
        suspicious = (downward and (rotation or shrinking)) or probability >= cfg.possible_probability
        strong = (downward and (rotation or shrinking)) or (
            probability >= cfg.falling_probability and (fall.calibrated or downward or rotation)
        )
        upright = posture in (Posture.STANDING, Posture.SITTING) and not suspicious
        if self.state == FallState.NORMAL:
            if self._persistent("possible", suspicious, timestamp, cfg.possible_persistence_seconds):
                self.state = FallState.POSSIBLE_FALL
                self.episode_start = timestamp
                self.timers.clear()
        elif self.state == FallState.POSSIBLE_FALL:
            if self._persistent("falling", strong, timestamp, cfg.falling_persistence_seconds):
                self.state = FallState.FALLING
                self.timers.clear()
            elif self._persistent("recovery", not suspicious, timestamp, cfg.recovery_seconds):
                self.state = FallState.NORMAL
                self.timers.clear()
        elif self.state == FallState.FALLING:
            if self._persistent("lying", horizontal and posture == Posture.LYING,
                                timestamp, cfg.lying_persistence_seconds):
                self.state = FallState.FALLEN
                self.timers.clear()
            elif self._persistent("recovery", upright, timestamp, cfg.recovery_seconds):
                self.state = FallState.NORMAL
                self.timers.clear()
        elif self.state == FallState.FALLEN:
            if self._persistent("recovery", upright, timestamp, cfg.recovery_seconds):
                self.state = FallState.NORMAL
                self.timers.clear()
        if (self.state in (FallState.POSSIBLE_FALL, FallState.FALLING) and self.episode_start is not None
                and timestamp - self.episode_start > cfg.episode_timeout_seconds):
            self.state = FallState.NORMAL
            self.timers.clear()
        return self.state
