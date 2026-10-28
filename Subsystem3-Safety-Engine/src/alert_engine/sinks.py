"""EventSink interface (section 8.6) and the local SQLite implementation for this phase.

A FirebaseSink can be added later behind the same interface, selected in config,
with no changes to the dispatcher.
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Protocol

from alert_engine.schemas import ClassifiedEvent

logger = logging.getLogger(__name__)


class EventSink(Protocol):
    """Storage target for statuses, alerts and camera status. Dispatcher only talks to this."""

    def write_status(self, person_id: str, posture: str, tier: str, risk_score: int, updated_at: datetime) -> None: ...

    def write_alert(self, event: ClassifiedEvent) -> None: ...

    def write_camera_status(self, camera_id: str, status: str, last_seen: datetime | None) -> None: ...


class NullSink:
    """No-op sink used in tests that don't care about storage."""

    def write_status(self, person_id: str, posture: str, tier: str, risk_score: int, updated_at: datetime) -> None:
        pass

    def write_alert(self, event: ClassifiedEvent) -> None:
        pass

    def write_camera_status(self, camera_id: str, status: str, last_seen: datetime | None) -> None:
        pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS resident_status (
    person_id TEXT PRIMARY KEY,
    posture TEXT NOT NULL,
    tier TEXT NOT NULL,
    risk_score INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    person_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    zone TEXT,
    posture TEXT NOT NULL,
    tier TEXT NOT NULL,
    risk_score INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    reasons TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    dispatched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS camera_status (
    camera_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    last_seen TEXT
);
"""

_Job = tuple  # (kind: str, args: tuple)


class LocalSqliteSink:
    """EventSink backed by SQLite at `sqlite_path`.

    All writes are enqueued and applied on a single background thread (one
    sqlite3 connection, one writer), so callers such as the dispatcher never
    block on disk I/O.
    """

    def __init__(self, sqlite_path: str | Path = "data/events.db") -> None:
        self._path = Path(sqlite_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="sqlite-sink")

    def start(self) -> None:
        self._worker.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._worker.join(timeout=timeout)

    def write_status(self, person_id: str, posture: str, tier: str, risk_score: int, updated_at: datetime) -> None:
        self._queue.put(("status", (person_id, posture, tier, risk_score, updated_at.isoformat())))

    def write_alert(self, event: ClassifiedEvent) -> None:
        self._queue.put(("alert", (event,)))

    def write_camera_status(self, camera_id: str, status: str, last_seen: datetime | None) -> None:
        self._queue.put(("camera_status", (camera_id, status, last_seen.isoformat() if last_seen else None)))

    def _run(self) -> None:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
            while not self._stop_event.is_set():
                job = self._queue.get()
                if job is None:
                    self._queue.task_done()
                    break
                try:
                    self._apply(conn, job)
                except Exception:
                    logger.exception("Failed to apply sink write %s", job[0])
                finally:
                    self._queue.task_done()
        finally:
            conn.close()

    def _apply(self, conn: sqlite3.Connection, job: _Job) -> None:
        kind, args = job
        if kind == "status":
            person_id, posture, tier, risk_score, updated_at = args
            conn.execute(
                """
                INSERT INTO resident_status (person_id, posture, tier, risk_score, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    posture=excluded.posture, tier=excluded.tier,
                    risk_score=excluded.risk_score, updated_at=excluded.updated_at
                """,
                (person_id, posture, tier, risk_score, updated_at),
            )
        elif kind == "alert":
            (event,) = args
            conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, event_id, person_id, camera_id, zone, posture, tier,
                    risk_score, duration_s, reasons, detected_at, dispatched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.alert_id, event.event_id, event.person_id, event.camera_id, event.zone,
                    event.posture, event.tier, event.risk_score, event.duration_s,
                    json.dumps(event.reasons), event.detected_at.isoformat(), event.dispatched_at.isoformat(),
                ),
            )
        elif kind == "camera_status":
            camera_id, status, last_seen = args
            conn.execute(
                """
                INSERT INTO camera_status (camera_id, status, last_seen)
                VALUES (?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET status=excluded.status, last_seen=excluded.last_seen
                """,
                (camera_id, status, last_seen),
            )
        else:
            raise ValueError(f"Unknown sink job kind: {kind}")
        conn.commit()
