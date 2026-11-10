from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default

from alert_engine.notifier import Notifier, format_caption, format_message, telegram_photo_sender
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


class FakeFrames:
    def __init__(self, frame=b"jpeg-bytes", error=None):
        self.frame = frame
        self.error = error

    def get_frame(self, camera_id, at):
        if self.error:
            raise self.error
        return self.frame


def run_one(notifier):
    notifier.start()
    notifier.notify(make_event())
    notifier.stop()


def test_alert_with_frame_sends_one_photo_and_no_text():
    texts, photos = [], []
    run_one(Notifier({"alert"}, send=texts.append, send_photo=lambda p, c: photos.append((p, c)), frames=FakeFrames()))
    assert texts == []
    assert photos[0][0] == b"jpeg-bytes"
    assert "ALERT" in photos[0][1] and "lounge" in photos[0][1]


def test_no_frame_falls_back_to_text():
    texts, photos = [], []
    run_one(Notifier({"alert"}, send=texts.append, send_photo=lambda p, c: photos.append(p), frames=FakeFrames(frame=None)))
    assert photos == [] and len(texts) == 1


def test_failed_photo_upload_still_sends_text():
    texts = []

    def broken_upload(photo, caption):
        raise OSError("upload failed")

    run_one(Notifier({"alert"}, send=texts.append, send_photo=broken_upload, frames=FakeFrames()))
    assert len(texts) == 1 and "ALERT" in texts[0]


def test_broken_frame_source_still_sends_text():
    texts = []
    run_one(Notifier({"alert"}, send=texts.append, send_photo=lambda p, c: None, frames=FakeFrames(error=RuntimeError("camera gone"))))
    assert len(texts) == 1


def test_no_frame_source_sends_text_only():
    texts, photos = [], []
    run_one(Notifier({"alert"}, send=texts.append, send_photo=lambda p, c: photos.append(p)))
    assert photos == [] and len(texts) == 1


def test_caption_fits_telegram_limit():
    event = make_event().model_copy(update={"person_id": "p" * 2000})
    assert len(format_caption(event)) <= 1024


def test_photo_sender_posts_multipart_request(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        captured["url"], captured["req"] = req.full_url, req
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    telegram_photo_sender("TOKEN", "42")(b"\xff\xd8jpeg", "ALERT: fall")

    req = captured["req"]
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + req.get_header("Content-type").encode() + b"\r\n\r\n" + req.data
    )
    parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
    assert captured["url"].endswith("/botTOKEN/sendPhoto")
    assert parts["chat_id"].get_content().strip() == "42"
    assert parts["caption"].get_content().strip() == "ALERT: fall"
    assert parts["photo"].get_content() == b"\xff\xd8jpeg"
