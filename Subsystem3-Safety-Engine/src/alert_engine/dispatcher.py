"""Routes classified events by tier (CLAUDE.md section 8.4).

    Tier     Sink status       Sink alert record   alert/events   Audio
    safe     Yes (on change)   No                  Yes            No
    warning  Yes               Yes                 Yes            No
    unsafe   Yes               Yes                 Yes            Yes
    alert    Yes               Yes                 Yes            Yes (repeat once)

A per-person cooldown_s stops the same condition from re-alerting repeatedly.
This module never touches MQTT, the sink's storage, or audio hardware directly:
it calls injected ports, so it can be unit tested with plain Python objects and
never blocks on I/O.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol

from alert_engine.config import DispatchConfig
from alert_engine.schemas import ClassifiedEvent
from alert_engine.sinks import EventSink

SAFE_TIER = "safe"
AUDIBLE_TIERS = {"unsafe", "alert"}


class AudioPort(Protocol):
    """What the dispatcher needs from the audio module (implemented in audio.py)."""

    def enqueue(self, tier: str, zone: str | None, repeat: bool) -> None: ...


class NotifierPort(Protocol):
    """Optional phone notifier (implemented in notifier.py)."""

    def notify(self, event: ClassifiedEvent) -> bool: ...


PublishFn = Callable[[str], None]


@dataclass
class DispatchOutcome:
    event: ClassifiedEvent
    published: bool = False
    sink_status_written: bool = False
    sink_alert_written: bool = False
    audio_triggered: bool = False
    audio_repeat: bool = False
    notified: bool = False


@dataclass
class _PersonDispatchState:
    last_status_key: tuple[str, str] | None = None
    last_status_write_at: datetime | None = None
    last_alert_key: tuple[str, str] | None = None
    last_alert_at: datetime | None = None


class Dispatcher:
    def __init__(
        self,
        config: DispatchConfig,
        sink: EventSink,
        audio: AudioPort,
        publish: PublishFn,
        notifier: NotifierPort | None = None,
    ) -> None:
        self._notifier = notifier
        self._config = config
        self._sink = sink
        self._audio = audio
        self._publish = publish
        self._people: dict[str, _PersonDispatchState] = {}

    def handle(
        self,
        *,
        event_id: str,
        person_id: str,
        camera_id: str,
        zone: str | None,
        posture: str,
        tier: str,
        risk_score: int,
        duration_s: float,
        reasons: list[str],
        detected_at: datetime,
        now: datetime | None = None,
    ) -> DispatchOutcome:
        dispatched_at = now if now is not None else datetime.now(timezone.utc)
        event = ClassifiedEvent(
            alert_id=str(uuid.uuid4()),
            event_id=event_id,
            person_id=person_id,
            camera_id=camera_id,
            zone=zone,
            posture=posture,
            tier=tier,
            risk_score=risk_score,
            duration_s=duration_s,
            reasons=reasons,
            detected_at=detected_at,
            dispatched_at=dispatched_at,
        )
        outcome = DispatchOutcome(event=event)
        state = self._people.setdefault(person_id, _PersonDispatchState())
        status_key = (posture, tier)
        status_changed = state.last_status_key != status_key

        self._publish(event.model_dump_json())
        outcome.published = True

        if self._should_write_status(state, status_changed, tier, dispatched_at):
            self._sink.write_status(person_id, posture, tier, risk_score, dispatched_at)
            state.last_status_key = status_key
            state.last_status_write_at = dispatched_at
            outcome.sink_status_written = True

        if tier != SAFE_TIER:
            alert_changed = state.last_alert_key != status_key
            cooldown_elapsed = (
                state.last_alert_at is None
                or (dispatched_at - state.last_alert_at).total_seconds() >= self._config.cooldown_s
            )
            if alert_changed or cooldown_elapsed:
                self._sink.write_alert(event)
                state.last_alert_key = status_key
                state.last_alert_at = dispatched_at
                outcome.sink_alert_written = True

                if self._notifier is not None:
                    outcome.notified = self._notifier.notify(event)

                if tier in AUDIBLE_TIERS:
                    repeat = tier == "alert"
                    self._audio.enqueue(tier, zone, repeat)
                    outcome.audio_triggered = True
                    outcome.audio_repeat = repeat

        return outcome

    def _should_write_status(
        self, state: _PersonDispatchState, status_changed: bool, tier: str, dispatched_at: datetime
    ) -> bool:
        if tier == SAFE_TIER:
            return status_changed
        interval_elapsed = (
            state.last_status_write_at is None
            or (dispatched_at - state.last_status_write_at).total_seconds() >= self._config.status_min_interval_s
        )
        return status_changed or interval_elapsed
