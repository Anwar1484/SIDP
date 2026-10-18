"""Offline videos use media time, never inference runtime, for temporal features."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    frame_id: int
    timestamp: float
    wall_timestamp: datetime
    image: np.ndarray


class VideoCapture:
    def __init__(self, path: str | Path, fallback_fps: float | None = None):
        if fallback_fps is not None and (not np.isfinite(fallback_fps) or fallback_fps <= 0):
            raise ValueError("fallback FPS must be finite and positive")
        self.path, self.fallback_fps = Path(path), fallback_fps
        self.cap: cv2.VideoCapture | None = None
        self.fps = 0.0
        self.origin = datetime.now(timezone.utc)
        self.timestamp_fallback_used = False

    def __enter__(self):
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"cannot open video {self.path}")
        reported = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported if np.isfinite(reported) and reported > 0 else self.fallback_fps or 0
        if self.fps <= 0:
            self.cap.release()
            raise ValueError("video has no valid FPS; provide --fps")
        return self

    def __iter__(self):
        if self.cap is None:
            raise RuntimeError("video is not open")
        previous = -1.0
        frame_id = 0
        while True:
            ok, image = self.cap.read()
            if not ok:
                break
            reported = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if np.isfinite(reported) and reported >= 0 and reported > previous:
                timestamp = reported
            else:
                self.timestamp_fallback_used = True
                timestamp = max(frame_id / self.fps, previous + 1 / self.fps)
            yield VideoFrame(frame_id, timestamp, self.origin + timedelta(seconds=timestamp), image)
            previous, frame_id = timestamp, frame_id + 1

    def __exit__(self, *_):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
