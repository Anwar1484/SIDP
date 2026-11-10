"""Reads the per-person activity log and works out durations and totals.

Each row marks the start of a posture or tier. A period ends when the person's next
row begins, so durations are calculated here rather than stored.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

UNMONITORED = "unmonitored"
RISK_TIERS = {"unsafe", "alert"}


@dataclass
class ActivityPeriod:
    person_id: str
    camera_id: str
    zone: str | None
    posture: str
    tier: str | None
    risk_score: int | None
    started_at: datetime
    duration_s: float
    ongoing: bool


@dataclass
class ActivitySummary:
    person_id: str
    posture_totals_s: dict[str, float]
    unmonitored_s: float
    risk_periods: int
    alert_periods: int


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def load_rows(db_path: str | Path, person_id: str | None = None) -> list[dict]:
    """Read activity rows in the order they were logged, without modifying the database."""
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM activity_log"
        params: tuple[str, ...] = ()
        if person_id is not None:
            query += " WHERE person_id = ?"
            params = (person_id,)
        return [dict(row) for row in conn.execute(query + " ORDER BY id", params)]
    finally:
        conn.close()


def build_periods(rows: list[dict], now: datetime) -> list[ActivityPeriod]:
    """Turn logged rows into periods, using each person's next row as the end time."""
    next_start: dict[str, datetime] = {}
    periods: list[ActivityPeriod] = []
    for row in reversed(rows):
        started_at = _parse_time(row["started_at"])
        ended_at = next_start.get(row["person_id"])
        duration = ((ended_at or now) - started_at).total_seconds()
        periods.append(
            ActivityPeriod(
                person_id=row["person_id"],
                camera_id=row["camera_id"],
                zone=row["zone"],
                posture=row["posture"],
                tier=row["tier"],
                risk_score=row["risk_score"],
                started_at=started_at,
                duration_s=max(duration, 0.0),
                ongoing=ended_at is None and row["posture"] != UNMONITORED,
            )
        )
        next_start[row["person_id"]] = started_at
    periods.reverse()
    return periods


def summarise(periods: list[ActivityPeriod], person_id: str) -> ActivitySummary:
    """Total time per posture for one person, and how many risky periods they had."""
    totals: dict[str, float] = {}
    unmonitored_s = 0.0
    risk_periods = alert_periods = 0
    previous_tier: str | None = None
    for period in (p for p in periods if p.person_id == person_id):
        if period.posture == UNMONITORED:
            unmonitored_s += period.duration_s
        else:
            totals[period.posture] = totals.get(period.posture, 0.0) + period.duration_s
        if period.tier in RISK_TIERS and previous_tier not in RISK_TIERS:
            risk_periods += 1
        if period.tier == "alert" and previous_tier != "alert":
            alert_periods += 1
        previous_tier = period.tier
    return ActivitySummary(person_id, totals, unmonitored_s, risk_periods, alert_periods)
