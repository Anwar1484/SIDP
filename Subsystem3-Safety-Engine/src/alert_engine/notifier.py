"""Phone pop-up notifications through a Telegram bot.

Sends text by default. With a FrameSource (notify.attach_frame in config) an alert
goes out as one camera photo with a short caption, which sends an image to Telegram
and so needs requirement R7 to allow it. If no frame is available or the upload
fails, the alert is sent as text instead, so a photo problem never loses an alert.

Credentials come from the environment / .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
If they are missing the notifier stays disabled and the rest of the engine is
unaffected.

Sending runs on its own worker thread with a queue, so the dispatcher never blocks
on the network, and a network failure never raises into the pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Callable

from alert_engine.frames import FrameSource
from alert_engine.schemas import ClassifiedEvent

logger = logging.getLogger(__name__)

TIER_LABEL = {"warning": "WARNING", "unsafe": "UNSAFE", "alert": "ALERT"}
CAPTION_LIMIT = 1024  # Telegram rejects longer photo captions
SendFn = Callable[[str], None]
SendPhotoFn = Callable[[bytes, str], None]


def format_caption(event: ClassifiedEvent) -> str:
    """Short photo caption for one classified event."""
    where = event.zone or event.camera_id
    label = TIER_LABEL.get(event.tier, event.tier.upper())
    caption = (
        f"{label}: {event.posture} detected in {where}\n"
        f"Resident {event.person_id} - risk {event.risk_score}/100, held {event.duration_s:.1f} s"
    )
    return caption[:CAPTION_LIMIT]


def format_message(event: ClassifiedEvent) -> str:
    """Plain-text phone message for one classified event."""
    reasons = "; ".join(event.reasons[:2])
    return f"{format_caption(event)}\n{reasons}"


def telegram_sender(token: str, chat_id: str, timeout_s: float = 5.0) -> SendFn:
    """Build a function that posts one text message via the Telegram Bot API."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(text: str) -> None:
        body = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()

    return send


def _multipart_body(chat_id: str, caption: str, photo: bytes) -> tuple[bytes, str]:
    """Encode a sendPhoto request as multipart/form-data. Returns (body, content type)."""
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in (("chat_id", chat_id), ("caption", caption)):
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode("utf-8")
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="frame.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    body += photo + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def telegram_photo_sender(token: str, chat_id: str, timeout_s: float = 5.0) -> SendPhotoFn:
    """Build a function that posts one captioned JPEG via the Telegram Bot API."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    def send_photo(photo: bytes, caption: str) -> None:
        body, content_type = _multipart_body(chat_id, caption, photo)
        req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()

    return send_photo


def _delay_s(event: ClassifiedEvent) -> float:
    return (datetime.now(timezone.utc) - event.dispatched_at).total_seconds()


class Notifier:
    """Queues alerts for the configured tiers and sends them on a worker thread."""

    def __init__(
        self,
        tiers: set[str],
        send: SendFn | None,
        postures: set[str] | None = None,
        send_photo: SendPhotoFn | None = None,
        frames: FrameSource | None = None,
    ) -> None:
        self._tiers = tiers
        self._postures = postures or set()  # empty = any posture
        self._send = send
        self._send_photo = send_photo
        self._frames = frames
        self._queue: queue.Queue[ClassifiedEvent | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(
        cls,
        tiers: list[str],
        enabled: bool = True,
        postures: list[str] | None = None,
        frames: FrameSource | None = None,
    ) -> "Notifier":
        """Create a notifier using TELEGRAM_* env vars; disabled if unset."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not enabled:
            logger.info("Phone notifications disabled in config")
            return cls(set(tiers), None)
        if not token or not chat_id:
            logger.info("Telegram credentials not set; phone notifications disabled")
            return cls(set(tiers), None)
        return cls(
            set(tiers),
            telegram_sender(token, chat_id),
            set(postures or []),
            telegram_photo_sender(token, chat_id),
            frames,
        )

    @property
    def active(self) -> bool:
        return self._send is not None

    def notify(self, event: ClassifiedEvent) -> bool:
        """Queue a message if this tier is configured. Returns True if queued."""
        if not self.active or event.tier not in self._tiers:
            return False
        if self._postures and event.posture not in self._postures:
            return False
        self._queue.put(event)
        return True

    def start(self) -> None:
        if not self.active or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="notifier", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                return
            try:
                self._deliver(event)
            except Exception as exc:  # network down, bad token, etc. Never crash the engine.
                logger.warning("Phone notification failed: %s", exc)

    def _deliver(self, event: ClassifiedEvent) -> None:
        """Send the photo alert if possible, otherwise the text alert."""
        frame = self._fetch_frame(event)
        if frame is not None and self._send_photo is not None:
            try:
                self._send_photo(frame, format_caption(event))
                logger.info("Phone alert sent with photo for %s, %.2f s after dispatch", event.alert_id, _delay_s(event))
                return
            except Exception as exc:
                logger.warning("Phone photo failed, sending text instead: %s", exc)
        self._send(format_message(event))
        logger.info("Phone alert sent as text for %s, %.2f s after dispatch", event.alert_id, _delay_s(event))

    def _fetch_frame(self, event: ClassifiedEvent) -> bytes | None:
        if self._frames is None:
            return None
        try:
            return self._frames.get_frame(event.camera_id, event.detected_at)
        except Exception as exc:
            logger.warning("Frame lookup failed, sending text instead: %s", exc)
            return None
