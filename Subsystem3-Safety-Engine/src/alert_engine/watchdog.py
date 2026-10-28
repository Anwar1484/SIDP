"""Camera dropout detection (CLAUDE.md section 8.7).

Tracks the last event time per camera_id. A camera is flagged unmonitored
once no events have arrived for camera_timeout_s, and recovers to monitored
as soon as a new event is seen. Detecting an absence of events needs a real
clock tick (nothing arrives to compare against), so `check()` is meant to be
called periodically, e.g. from a background poll thread via `start_polling`.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

MONITORED = "monitored"
UNMONITORED = "unmonitored"

CameraSnapshot = dict[str, dict[str, str | None]]
StatusChangeHandler = Callable[[CameraSnapshot], None]


@dataclass
class _CameraState:
    status: str = MONITORED
    last_seen: datetime | None = None


class CameraWatchdog:
    def __init__(
        self,
        camera_timeout_s: float,
        on_status_change: StatusChangeHandler,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._timeout_s = camera_timeout_s
        self._on_status_change = on_status_change
        self._clock = clock
        self._cameras: dict[str, _CameraState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

    def touch(self, camera_id: str, at: datetime | None = None) -> None:
        """Record that camera_id just sent an event; recovers it if it had dropped out."""
        now = at if at is not None else self._clock()
        with self._lock:
            state = self._cameras.setdefault(camera_id, _CameraState())
            became_monitored = state.status != MONITORED
            state.last_seen = now
            state.status = MONITORED
        if became_monitored:
            self._notify()

    def check(self, now: datetime | None = None) -> None:
        """Flag any camera with no events for camera_timeout_s as unmonitored. Call periodically."""
        now = now if now is not None else self._clock()
        changed = False
        with self._lock:
            for state in self._cameras.values():
                if (
                    state.status == MONITORED
                    and state.last_seen is not None
                    and (now - state.last_seen).total_seconds() >= self._timeout_s
                ):
                    state.status = UNMONITORED
                    changed = True
        if changed:
            self._notify()

    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            return {
                camera_id: {"status": s.status, "last_seen": s.last_seen.isoformat() if s.last_seen else None}
                for camera_id, s in self._cameras.items()
            }

    def _notify(self) -> None:
        try:
            self._on_status_change(self.snapshot())
        except Exception:
            logger.exception("Camera status change handler raised")

    def start_polling(self, interval_s: float = 1.0) -> None:
        def _run() -> None:
            while not self._stop_event.wait(interval_s):
                self.check()

        self._poll_thread = threading.Thread(target=_run, daemon=True, name="camera-watchdog")
        self._poll_thread.start()

    def stop_polling(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=timeout)
