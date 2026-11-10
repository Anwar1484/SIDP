"""Prints each resident's activity timeline and totals from the local event store.

Usage:
    python tools/activity_report.py
    python tools/activity_report.py --person p-003 --hours 24
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alert_engine.activity import ActivityPeriod, ActivitySummary, build_periods, load_rows, summarise  # noqa: E402


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_period(period: ActivityPeriod) -> str:
    started = period.started_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    duration = "ongoing" if period.ongoing else format_duration(period.duration_s)
    score = "-" if period.risk_score is None else str(period.risk_score)
    where = period.zone or period.camera_id
    return f"  {started}  {duration:>9}  {period.posture:<12} {period.tier or '-':<8} {score:>5}  {where}"


def format_summary(summary: ActivitySummary) -> str:
    totals = ", ".join(f"{name} {format_duration(s)}" for name, s in sorted(summary.posture_totals_s.items()))
    return (
        f"  Totals: {totals or 'none'}\n"
        f"  Unmonitored: {format_duration(summary.unmonitored_s)}, "
        f"unsafe periods: {summary.risk_periods}, alert periods: {summary.alert_periods}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", help="only show this person_id")
    parser.add_argument("--hours", type=float, default=24, help="how far back to show (default 24)")
    parser.add_argument("--db", default="data/events.db")
    args = parser.parse_args()

    try:
        rows = load_rows(args.db, args.person)
    except sqlite3.OperationalError as exc:
        sys.exit(f"Could not read activity from {args.db}: {exc}. Start S3 once to create it.")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=args.hours)
    periods = [p for p in build_periods(rows, now) if p.started_at >= cutoff]
    if not periods:
        print("No activity logged for that selection.")
        return
    for person_id in sorted({p.person_id for p in periods}):
        print(f"\n{person_id}")
        print(f"  {'Started':<19}  {'Duration':>9}  {'Posture':<12} {'Tier':<8} {'Score':>5}  Where")
        for period in (p for p in periods if p.person_id == person_id):
            print(format_period(period))
        print(format_summary(summarise(periods, person_id)))


if __name__ == "__main__":
    main()
