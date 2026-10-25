"""Subsystem 1 clean-frame posture dataset collector (no HUD overlay)."""
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.config import DatasetConfig
from app.runtime import ROOT


class DatasetCollector:
    """Saves clean BGR frames into dataset/<class>/ manually or at a fixed 2 Hz cadence."""

    def __init__(self, config: DatasetConfig, root: Path | None = None):
        self.config = config
        base = Path(config.directory)
        self.root = base if base.is_absolute() else (root or ROOT) / base
        self.classes = list(config.classes)
        self.current_class = self.classes[0]
        self.recording = False
        self.last_saved_at = 0.0
        self.last_saved_path: str | None = None
        self._lock = threading.Lock()
        for name in self.classes:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def select_index(self, index: int) -> str:
        with self._lock:
            if 0 <= index < len(self.classes):
                self.current_class = self.classes[index]
            return self.current_class

    def select_class(self, name: str) -> str:
        with self._lock:
            if name in self.classes:
                self.current_class = name
            return self.current_class

    def toggle_recording(self) -> bool:
        with self._lock:
            self.recording = not self.recording
            return self.recording

    def count(self, class_name: str | None = None) -> int:
        target = self.root / (class_name or self.current_class)
        if not target.is_dir():
            return 0
        return sum(1 for p in target.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    def snap(self, clean_frame: np.ndarray) -> Path:
        with self._lock:
            cls = self.current_class
            folder = self.root / cls
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = folder / f"{cls}_{stamp}.jpg"
            cv2.imwrite(str(path), clean_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
            self.last_saved_at = time.monotonic()
            self.last_saved_path = str(path.relative_to(self.root.parent) if path.is_relative_to(self.root.parent) else path)
            return path

    def maybe_record(self, clean_frame: np.ndarray) -> Path | None:
        if not self.recording:
            return None
        interval = 1.0 / self.config.record_hz
        if (time.monotonic() - self.last_saved_at) >= interval:
            return self.snap(clean_frame)
        return None

    def status(self) -> dict:
        with self._lock:
            cls = self.current_class
            rec = self.recording
            last = self.last_saved_path
        return {
            "current_class": cls,
            "classes": self.classes,
            "recording": rec,
            "record_hz": self.config.record_hz,
            "count": self.count(cls),
            "counts": {name: self.count(name) for name in self.classes},
            "last_saved": last,
        }

