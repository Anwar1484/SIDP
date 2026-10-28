from datetime import datetime, timedelta, timezone

import pytest

from alert_engine.config import DebounceConfig
from alert_engine.debounce import DebounceTracker
from alert_engine.schemas import PostureEventIn

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_event(offset_s: float, posture: str, confidence: float = 0.9, person_id: str = "p-1") -> PostureEventIn:
    return PostureEventIn(
        event_id=f"evt-{offset_s}-{posture}",
        timestamp=BASE_TIME + timedelta(seconds=offset_s),
        camera_id="cam-1",
        person_id=person_id,
        posture=posture,
        confidence=confidence,
        zone="lounge",
    )


@pytest.fixture
def config() -> DebounceConfig:
    return DebounceConfig(
        window_s=1.5,
        min_confirmations=3,
        min_confidence=0.6,
        resolve_after_s=3.0,
        safe_postures=["sitting", "standing", "walking"],
    )


def confirm(tracker: DebounceTracker, posture: str, start_offset: float, person_id: str = "p-1", gap: float = 0.1):
    """Feed 3 confirming events for `posture` starting at start_offset. Returns the final result."""
    result = None
    for i in range(3):
        result = tracker.process(make_event(start_offset + i * gap, posture, person_id=person_id))
    return result


def test_sustained_posture_confirmed(config):
    tracker = DebounceTracker(config)
    result = confirm(tracker, "bending", start_offset=0.0)
    assert result.changed is True
    assert result.state == "confirmed"
    assert result.posture == "bending"
    assert result.duration_s == pytest.approx(0.2)


def test_single_frame_glitch_ignored(config):
    tracker = DebounceTracker(config)
    confirm(tracker, "walking", start_offset=0.0)

    glitch_result = tracker.process(make_event(1.0, "fall", confidence=0.75))
    assert glitch_result.changed is False
    assert glitch_result.state == "suspected"
    assert glitch_result.posture == "walking"  # still reports the old confirmed posture

    back_to_normal = tracker.process(make_event(1.1, "walking"))
    assert back_to_normal.changed is False
    assert back_to_normal.posture == "walking"
    assert back_to_normal.state == "confirmed"  # dropped back, not confirmed as "fall"


def test_low_confidence_events_do_not_count_towards_confirmation(config):
    tracker = DebounceTracker(config)
    confirm(tracker, "walking", start_offset=0.0)

    for i in range(5):
        result = tracker.process(make_event(1.0 + i * 0.1, "fall", confidence=0.3))
    assert result.state == "suspected"
    assert result.posture == "walking"
    assert result.changed is False


def test_resolve_after_safe_posture_held_long_enough(config):
    tracker = DebounceTracker(config)
    confirm(tracker, "bending", start_offset=0.0)
    confirm(tracker, "sitting", start_offset=1.0)

    still_confirmed = tracker.process(make_event(2.0, "sitting"))
    assert still_confirmed.state == "confirmed"

    resolved = tracker.process(make_event(1.0 + 3.0, "sitting"))
    assert resolved.state == "resolved"
    assert resolved.posture == "sitting"


def test_unsafe_posture_never_resolves_on_its_own(config):
    tracker = DebounceTracker(config)
    confirm(tracker, "bending", start_offset=0.0)

    still_bending = tracker.process(make_event(10.0, "bending"))
    assert still_bending.state == "confirmed"  # never "resolved": bending is not a safe posture


def test_time_window_expiry_with_uneven_gaps_prevents_confirmation(config):
    tracker = DebounceTracker(config)
    confirm(tracker, "standing", start_offset=0.0)

    tracker.process(make_event(1.0, "lying", confidence=0.9))
    tracker.process(make_event(2.0, "lying", confidence=0.9))  # 1.0 now outside the 1.5s window
    result = tracker.process(make_event(2.6, "lying", confidence=0.9))  # both prior events now outside window

    assert result.state == "suspected"
    assert result.posture == "standing"
    assert result.changed is False


def test_time_window_with_uneven_gaps_still_confirms_within_window(config):
    tracker = DebounceTracker(config)
    confirm(tracker, "standing", start_offset=0.0)

    tracker.process(make_event(1.0, "lying", confidence=0.9))
    tracker.process(make_event(1.8, "lying", confidence=0.9))
    result = tracker.process(make_event(2.4, "lying", confidence=0.9))  # all three within 1.5s of each other

    assert result.state == "confirmed"
    assert result.posture == "lying"
    assert result.changed is True
    assert result.duration_s == pytest.approx(1.4)


def test_reset_person_clears_state_so_next_event_must_reconfirm(config):
    tracker = DebounceTracker(config)
    confirm(tracker, "walking", start_offset=0.0)

    tracker.reset_person("p-1")

    # duration_s does not resume from the stale confirmed_since; posture must reconfirm
    result = tracker.process(make_event(100.0, "walking"))
    assert result.state == "suspected"
    assert result.posture is None
    assert result.duration_s == 0.0


def test_reset_person_is_safe_for_unknown_person(config):
    tracker = DebounceTracker(config)
    tracker.reset_person("nobody")  # must not raise


def test_separate_people_tracked_independently(config):
    tracker = DebounceTracker(config)
    confirm(tracker, "sitting", start_offset=0.0, person_id="p-1")
    result_p2 = confirm(tracker, "bending", start_offset=0.0, person_id="p-2")

    assert result_p2.posture == "bending"
    result_p1 = tracker.process(make_event(0.5, "sitting", person_id="p-1"))
    assert result_p1.posture == "sitting"
