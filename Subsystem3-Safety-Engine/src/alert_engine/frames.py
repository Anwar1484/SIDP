"""Finds the camera frame to attach to a phone alert.

The notifier only talks to the FrameSource interface, so the real camera hook can
replace FolderFrameSource later without changing the notifier or the dispatcher.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# camera_id arrives over MQTT, so it must never be able to point outside frame_dir.
SAFE_CAMERA_ID = re.compile(r"[A-Za-z0-9_-]+")
# Windows briefly locks a file while the camera feed replaces it, so retry a few times.
READ_ATTEMPTS = 3
READ_RETRY_S = 0.05


class FrameSource(Protocol):
    """Anything that can supply a JPEG for a camera around a given time."""

    def get_frame(self, camera_id: str, at: datetime) -> bytes | None: ...


class FolderFrameSource:
    """Reads the latest frame for each camera from <frame_dir>/<camera_id>.jpg.

    Returns None if the file is missing or was not written within max_age_s of the
    event, because an old picture could show staff the wrong scene.
    """

    def __init__(self, frame_dir: str | Path, max_age_s: float) -> None:
        self._frame_dir = Path(frame_dir)
        self._max_age_s = max_age_s

    def get_frame(self, camera_id: str, at: datetime) -> bytes | None:
        if not SAFE_CAMERA_ID.fullmatch(camera_id):
            logger.warning("Ignoring frame lookup for unsafe camera_id %r", camera_id)
            return None
        path = self._frame_dir / f"{camera_id}.jpg"
        for _ in range(READ_ATTEMPTS):
            try:
                if abs(at.timestamp() - path.stat().st_mtime) > self._max_age_s:
                    logger.info("Frame for %s is stale; sending text only", camera_id)
                    return None
                return path.read_bytes()
            except PermissionError:
                time.sleep(READ_RETRY_S)
            except OSError:
                break
        logger.info("No frame available for %s; sending text only", camera_id)
        return None
