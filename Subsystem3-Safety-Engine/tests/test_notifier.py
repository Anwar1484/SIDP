from datetime import datetime, timezone

from alert_engine.notifier import Notifier, format_message
from alert_engine.schemas import ClassifiedEvent

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_event(tier="alert", posture="fall") -> ClassifiedEvent:
    return ClassifiedEvent(
        alert_id="a1", event_id="e1", person_id="p-003", camera_id="cam-01", zone="lounge",
        posture=posture, tier=tier, risk_score=90, duration_s=2.5,
        reasons=["fall base severity 95"], detected_at=NOW, dispatched_at=NOW,
    )


def test_message_has_tier_zone_and_resident():
    text = format_message(make_event())
    assert "ALERT" in text and "lounge" in text and "p-003" in text


def test_only_configured_tiers_are_queued():
    n = Notifier({"unsafe", "alert"}, send=lambda t: None)
    assert n.notify(make_event("alert")) is True
    assert n.notify(make_event("warning", "bending")) is False
    assert n.notify(make_event("safe", "walking")) is False


def test_posture_filter_limits_messages():
    n = Notifier({"alert"}, send=lambda t: None, postures={"fall", "lying"})
    assert n.notify(make_event("alert", "fall")) is True
    assert n.notify(make_event("alert", "bending")) is False


def test_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    n = Notifier.from_env(["alert"])
    assert n.active is False
    assert n.notify(make_event()) is False


def test_worker_sends_and_survives_failures():
    sent = []

    def flaky(text):
        if not sent:
            sent.append("boom")
            raise OSError("network down")
        sent.append(text)

    n = Notifier({"alert"}, send=flaky)
    n.start()
    n.notify(make_event())
    n.notify(make_event())
    n.stop()
    assert len(sent) == 2 and "ALERT" in sent[1]
