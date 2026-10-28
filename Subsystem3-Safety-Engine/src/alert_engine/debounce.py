"""Per-person posture debounce state machine (CLAUDE.md section 8.2).

A new posture must accumulate `min_confirmations` events at or above
`min_confidence` within a `window_s` time window (measured from event
timestamps, not frame counts) before it replaces the current confirmed
posture. This absorbs single-frame misreads from the vision model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from alert_engine.config import DebounceConfig
from alert_engine.schemas import PostureEventIn

DebounceState = Literal["monitoring", "suspected", "confirmed", "resolved"]


@dataclass
class DebounceResult:
    """Outcome of feeding one event into a person's state machine."""

    person_id: str
    state: DebounceState
    posture: str | None
    duration_s: float
    changed: bool
    confirmed_at: datetime | None = None
    previous_posture: str | None = None


@dataclass
class _PersonState:
    person_id: str
    state: DebounceState = "monitoring"
    confirmed_posture: str | None = None
    confirmed_since: datetime | None = None
    suspect_posture: str | None = None
    suspect_event_times: list[datetime] = field(default_factory=list)


class DebounceTracker:
    """Owns one debounce state machine per person_id."""

    def __init__(self, config: DebounceConfig) -> None:
        self._config = config
        self._people: dict[str, _PersonState] = {}

    def reset_person(self, person_id: str) -> None:
        """Clears a person's state, e.g. after their camera drops out and recovers.

        Duration/posture confirmed before a camera outage isn't evidence of
        what happened during the outage, so on recovery the person must
        re-earn confirmation from scratch instead of resuming a stale
        duration_s that silently spans the blackout.
        """
        self._people.pop(person_id, None)

    def process(self, event: PostureEventIn) -> DebounceResult:
        """Feed one validated event into that person's state machine."""
        person = self._people.setdefault(event.person_id, _PersonState(person_id=event.person_id))
        if event.posture == person.confirmed_posture:
            return self._handle_matching(person, event)
        return self._handle_differing(person, event)

    def _handle_matching(self, person: _PersonState, event: PostureEventIn) -> DebounceResult:
        # A frame confirming the current posture ends any in-flight suspicion.
        person.suspect_posture = None
        person.suspect_event_times = []

        assert person.confirmed_since is not None  # confirmed_posture set implies this is set too
        duration = (event.timestamp - person.confirmed_since).total_seconds()

        is_safe = person.confirmed_posture in self._config.safe_postures
        person.state = "resolved" if (is_safe and duration >= self._config.resolve_after_s) else "confirmed"

        return DebounceResult(
            person_id=person.person_id,
            state=person.state,
            posture=person.confirmed_posture,
            duration_s=duration,
            changed=False,
            confirmed_at=person.confirmed_since,
        )

    def _handle_differing(self, person: _PersonState, event: PostureEventIn) -> DebounceResult:
        if person.suspect_posture != event.posture:
            person.suspect_posture = event.posture
            person.suspect_event_times = []

        if event.confidence >= self._config.min_confidence:
            person.suspect_event_times.append(event.timestamp)

        window_s = self._config.window_s
        person.suspect_event_times = [
            t for t in person.suspect_event_times if (event.timestamp - t).total_seconds() <= window_s
        ]

        if len(person.suspect_event_times) >= self._config.min_confirmations:
            return self._confirm(person, event)

        person.state = "suspected"
        duration = (
            (event.timestamp - person.confirmed_since).total_seconds()
            if person.confirmed_since is not None
            else 0.0
        )
        return DebounceResult(
            person_id=person.person_id,
            state="suspected",
            posture=person.confirmed_posture,
            duration_s=duration,
            changed=False,
            confirmed_at=person.confirmed_since,
        )

    def _confirm(self, person: _PersonState, event: PostureEventIn) -> DebounceResult:
        previous_posture = person.confirmed_posture
        person.confirmed_posture = event.posture
        person.confirmed_since = person.suspect_event_times[0]
        person.suspect_posture = None
        person.suspect_event_times = []
        person.state = "confirmed"
        duration = (event.timestamp - person.confirmed_since).total_seconds()
        return DebounceResult(
            person_id=person.person_id,
            state="confirmed",
            posture=person.confirmed_posture,
            duration_s=duration,
            changed=True,
            confirmed_at=person.confirmed_since,
            previous_posture=previous_posture,
        )
