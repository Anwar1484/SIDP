from datetime import datetime, timedelta, timezone

import pytest

from alert_engine.config import DispatchConfig
from alert_engine.dispatcher import Dispatcher

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


class RecordingSink:
    def __init__(self):
        self.statuses = []
        self.alerts = []
        self.camera_statuses = []

    def write_status(self, person_id, posture, tier, risk_score, updated_at):
        self.statuses.append((person_id, posture, tier, risk_score, updated_at))

    def write_alert(self, event):
        self.alerts.append(event)

    def write_camera_status(self, camera_id, status, last_seen):
        self.camera_statuses.append((camera_id, status, last_seen))


class RecordingAudio:
    def __init__(self):
        self.calls = []

    def enqueue(self, tier, zone, repeat):
        self.calls.append((tier, zone, repeat))


class RecordingPublisher:
    def __init__(self):
        self.published = []

    def __call__(self, payload):
        self.published.append(payload)


@pytest.fixture
def config() -> DispatchConfig:
    return DispatchConfig(cooldown_s=30, status_min_interval_s=2)


@pytest.fixture
def harness(config):
    sink = RecordingSink()
    audio = RecordingAudio()
    publisher = RecordingPublisher()
    dispatcher = Dispatcher(config, sink, audio, publisher)
    return dispatcher, sink, audio, publisher


def dispatch(dispatcher, offset_s, tier, posture="bending", person_id="p-1", risk_score=50):
    return dispatcher.handle(
        event_id=f"evt-{offset_s}",
        person_id=person_id,
        camera_id="cam-1",
        zone="lounge",
        posture=posture,
        tier=tier,
        risk_score=risk_score,
        duration_s=1.0,
        reasons=["test"],
        detected_at=BASE_TIME + timedelta(seconds=offset_s),
        now=BASE_TIME + timedelta(seconds=offset_s),
    )


def test_safe_tier_writes_status_only_on_change(harness):
    dispatcher, sink, audio, publisher = harness

    dispatch(dispatcher, 0, "safe", posture="sitting")
    assert len(sink.statuses) == 1
    assert len(sink.alerts) == 0
    assert len(audio.calls) == 0
    assert len(publisher.published) == 1

    dispatch(dispatcher, 1, "safe", posture="sitting")  # unchanged
    assert len(sink.statuses) == 1  # not written again
    assert len(publisher.published) == 2  # still published every time

    dispatch(dispatcher, 2, "safe", posture="standing")  # posture changed
    assert len(sink.statuses) == 2


def test_warning_tier_writes_status_and_alert_but_no_audio(harness):
    dispatcher, sink, audio, publisher = harness

    outcome = dispatch(dispatcher, 0, "warning")
    assert outcome.sink_status_written is True
    assert outcome.sink_alert_written is True
    assert outcome.audio_triggered is False
    assert len(audio.calls) == 0


def test_unsafe_tier_triggers_audio_without_repeat(harness):
    dispatcher, sink, audio, publisher = harness

    outcome = dispatch(dispatcher, 0, "unsafe")
    assert outcome.sink_alert_written is True
    assert outcome.audio_triggered is True
    assert outcome.audio_repeat is False
    assert audio.calls == [("unsafe", "lounge", False)]


def test_alert_tier_triggers_audio_with_repeat(harness):
    dispatcher, sink, audio, publisher = harness

    outcome = dispatch(dispatcher, 0, "alert")
    assert outcome.audio_triggered is True
    assert outcome.audio_repeat is True
    assert audio.calls == [("alert", "lounge", True)]


def test_cooldown_suppresses_repeated_alert_and_audio(harness):
    dispatcher, sink, audio, publisher = harness

    dispatch(dispatcher, 0, "unsafe")
    assert len(sink.alerts) == 1
    assert len(audio.calls) == 1

    # same condition, well within cooldown_s=30
    outcome = dispatch(dispatcher, 5, "unsafe")
    assert outcome.sink_alert_written is False
    assert outcome.audio_triggered is False
    assert len(sink.alerts) == 1
    assert len(audio.calls) == 1

    # status write can still refresh once status_min_interval_s (2s) has passed
    assert outcome.sink_status_written is True

    # cooldown has now elapsed -> re-alerts
    outcome = dispatch(dispatcher, 31, "unsafe")
    assert outcome.sink_alert_written is True
    assert outcome.audio_triggered is True
    assert len(sink.alerts) == 2
    assert len(audio.calls) == 2


def test_tier_change_bypasses_cooldown(harness):
    dispatcher, sink, audio, publisher = harness

    dispatch(dispatcher, 0, "warning")
    assert len(sink.alerts) == 1
    assert len(audio.calls) == 0  # warning: no audio

    # escalates to unsafe well within cooldown_s -> must alert immediately, not wait
    outcome = dispatch(dispatcher, 1, "unsafe")
    assert outcome.sink_alert_written is True
    assert outcome.audio_triggered is True
    assert len(sink.alerts) == 2
    assert len(audio.calls) == 1


def test_alert_events_always_published_regardless_of_tier(harness):
    dispatcher, sink, audio, publisher = harness

    for i, tier in enumerate(["safe", "warning", "unsafe", "alert"]):
        dispatch(dispatcher, i, tier)
    assert len(publisher.published) == 4


def test_people_tracked_independently_for_cooldown(harness):
    dispatcher, sink, audio, publisher = harness

    dispatch(dispatcher, 0, "unsafe", person_id="p-1")
    dispatch(dispatcher, 0.5, "unsafe", person_id="p-2")
    assert len(sink.alerts) == 2
    assert len(audio.calls) == 2
