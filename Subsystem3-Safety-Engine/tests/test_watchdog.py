from datetime import datetime, timedelta, timezone

from alert_engine.watchdog import CameraWatchdog

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def t(offset_s: float) -> datetime:
    return BASE_TIME + timedelta(seconds=offset_s)


def test_camera_starts_monitored_on_first_touch():
    changes = []
    watchdog = CameraWatchdog(camera_timeout_s=5, on_status_change=changes.append)
    watchdog.touch("cam-01", at=t(0))
    assert watchdog.snapshot()["cam-01"]["status"] == "monitored"
    assert changes == []  # first touch isn't a "change", camera was never known before


def test_no_dropout_before_timeout():
    changes = []
    watchdog = CameraWatchdog(camera_timeout_s=5, on_status_change=changes.append)
    watchdog.touch("cam-01", at=t(0))
    watchdog.check(now=t(4))
    assert watchdog.snapshot()["cam-01"]["status"] == "monitored"
    assert changes == []


def test_dropout_flagged_after_timeout():
    changes = []
    watchdog = CameraWatchdog(camera_timeout_s=5, on_status_change=changes.append)
    watchdog.touch("cam-01", at=t(0))
    watchdog.check(now=t(6))
    assert watchdog.snapshot()["cam-01"]["status"] == "unmonitored"
    assert len(changes) == 1
    assert changes[0]["cam-01"]["status"] == "unmonitored"


def test_recovers_to_monitored_on_next_event():
    changes = []
    watchdog = CameraWatchdog(camera_timeout_s=5, on_status_change=changes.append)
    watchdog.touch("cam-01", at=t(0))
    watchdog.check(now=t(6))  # drops out
    assert watchdog.snapshot()["cam-01"]["status"] == "unmonitored"

    watchdog.touch("cam-01", at=t(8))  # recovers
    assert watchdog.snapshot()["cam-01"]["status"] == "monitored"
    assert len(changes) == 2
    assert changes[1]["cam-01"]["status"] == "monitored"


def test_repeated_checks_without_change_do_not_notify_again():
    changes = []
    watchdog = CameraWatchdog(camera_timeout_s=5, on_status_change=changes.append)
    watchdog.touch("cam-01", at=t(0))
    watchdog.check(now=t(6))
    watchdog.check(now=t(7))
    watchdog.check(now=t(8))
    assert len(changes) == 1  # only the one transition, not re-fired every check


def test_cameras_tracked_independently():
    changes = []
    watchdog = CameraWatchdog(camera_timeout_s=5, on_status_change=changes.append)
    watchdog.touch("cam-01", at=t(0))
    watchdog.touch("cam-02", at=t(0))
    watchdog.touch("cam-02", at=t(4))  # cam-02 keeps sending events

    watchdog.check(now=t(6))
    snapshot = watchdog.snapshot()
    assert snapshot["cam-01"]["status"] == "unmonitored"
    assert snapshot["cam-02"]["status"] == "monitored"


def test_snapshot_includes_last_seen_timestamp():
    watchdog = CameraWatchdog(camera_timeout_s=5, on_status_change=lambda snap: None)
    watchdog.touch("cam-01", at=t(0))
    assert watchdog.snapshot()["cam-01"]["last_seen"] == t(0).isoformat()
