"""Alert latency metrics (section 8.8): how R1 (alert within 2 s) gets measured."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Appends one JSON record per dispatched alert to logs/metrics.jsonl."""

    def __init__(self, path: str | Path = "logs/metrics.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_alert(
        self,
        alert_id: str,
        detected_at: datetime,
        received_at: datetime,
        confirmed_at: datetime,
        dispatched_at: datetime,
    ) -> None:
        record = {
            "alert_id": alert_id,
            "detected_at": detected_at.isoformat(),
            "received_at": received_at.isoformat(),
            "confirmed_at": confirmed_at.isoformat(),
            "dispatched_at": dispatched_at.isoformat(),
            "latency_s": (dispatched_at - detected_at).total_seconds(),
        }
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            logger.exception("Failed to write metrics record for alert %s", alert_id)
