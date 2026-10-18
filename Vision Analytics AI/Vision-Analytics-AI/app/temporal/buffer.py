"""Bounded observations with explicit gaps and fixed-rate resampling."""
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.config import TemporalConfig
from app.posture.normalization import NormalizedPose
from app.schemas import GeometryFeatures, PersonPose, PostureResult


@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    pose: PersonPose
    normalized: NormalizedPose
    posture: PostureResult
    geometry: GeometryFeatures
    frame_size: tuple[int, int]


class TemporalBuffer:
    def __init__(self, config: TemporalConfig):
        self.config = config
        self.samples: deque[PoseSample] = deque(maxlen=config.capacity)

    def append(self, sample: PoseSample) -> bool:
        """Return True if a long gap required a fresh history."""
        reset = False
        if self.samples:
            dt = sample.timestamp - self.samples[-1].timestamp
            if dt <= 0:
                raise ValueError("buffer timestamps must increase")
            if dt > self.config.max_gap_seconds or sample.frame_size != self.samples[-1].frame_size:
                self.samples.clear()
                reset = True
        self.samples.append(sample)
        return reset

    def window(self, frames: int) -> tuple[np.ndarray | None, float]:
        """Resample xy linearly and masks conservatively at sample_hz.

        Both bracketing endpoints must be valid; no extrapolation or interpolation
        across large gaps. 60 samples at 30 Hz span 59/30 seconds.
        """
        if len(self.samples) < 2:
            return None, 0.0
        values = list(self.samples)
        times = np.array([s.timestamp for s in values])
        targets = times[-1] - np.arange(frames - 1, -1, -1) / self.config.sample_hz
        if targets[0] < times[0] - 1e-6:
            return None, 0.0
        out = np.zeros((frames, 51), dtype=np.float32)
        for i, t in enumerate(targets):
            right = int(np.searchsorted(times, t))
            right = min(right, len(times) - 1)
            if abs(times[right] - t) < 1e-6:
                a = b = right
            else:
                a, b = max(0, right - 1), right
            left, right_sample = values[a], values[b]
            gap = times[b] - times[a]
            if gap > self.config.max_gap_seconds or not (left.normalized.valid and right_sample.normalized.valid):
                continue
            fraction = (t - times[a]) / gap if gap else 0.0
            mask = left.normalized.mask & right_sample.normalized.mask
            xy = left.normalized.xy * (1 - fraction) + right_sample.normalized.xy * fraction
            xy[~mask] = 0
            out[i] = np.concatenate((xy.flatten(), mask.astype(np.float32)))
        ratio = float(out[:, 34:].mean())
        if ratio < self.config.min_valid_ratio:
            return None, ratio
        return out, ratio
