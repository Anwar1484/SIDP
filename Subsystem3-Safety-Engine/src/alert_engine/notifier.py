"""Phone pop-up notifications through a Telegram bot.

Sends text only (no video or images, requirement R7). Credentials come from the
environment / .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). If they are missing the
notifier stays disabled and the rest of the engine is unaffected.

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
from typing import Callable

from alert_engine.schemas import ClassifiedEvent

logger = logging.getLogger(__name__)

TIER_LABEL = {"warning": "WARNING", "unsafe": "UNSAFE", "alert": "ALERT"}
SendFn = Callable[[str], None]


def format_message(event: ClassifiedEvent) -> str:
    """Plain-text phone message for one classified event."""
    where = event.zone or event.camera_id
    label = TIER_LABEL.get(event.tier, event.tier.upper())
    reasons = "; ".join(event.reasons[:2])
    return (
        f"{label}: {event.posture} detected in {where}\n"
        f"Resident {event.person_id} - risk {event.risk_score}/100, held {event.duration_s:.1f} s\n"
        f"{reasons}"
    )


def telegram_sender(token: str, chat_id: str, timeout_s: float = 5.0) -> SendFn:
    """Build a function that posts one text message via the Telegram Bot API."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(text: str) -> None:
        body = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()

    return send


class Notifier:
    """Queues alerts for the configured tiers and sends them on a worker thread."""

    def __init__(self, tiers: set[str], send: SendFn | None, postures: set[str] | None = None) -> None:
        self._tiers = tiers
        self._postures = postures or set()  # empty = any posture
        self._send = send
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls, tiers: list[str], enabled: bool = True, postures: list[str] | None = None) -> "Notifier":
        """Create a notifier using TELEGRAM_* env vars; disabled if unset."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not enabled:
            logger.info("Phone notifications disabled in config")
            return cls(set(tiers), None)
        if not token or not chat_id:
            logger.info("Telegram credentials not set; phone notifications disabled")
            return cls(set(tiers), None)
        return cls(set(tiers), telegram_sender(token, chat_id), set(postures or []))

    @property
    def active(self) -> bool:
        return self._send is not None

    def notify(self, event: ClassifiedEvent) -> bool:
        """Queue a message if this tier is configured. Returns True if queued."""
        if not self.active or event.tier not in self._tiers:
            return False
        if self._postures and event.posture not in self._postures:
            return False
        self._queue.put(format_message(event))
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
            text = self._queue.get()
            if text is None:
                return
            try:
                self._send(text)
            except Exception as exc:  # network down, bad token, etc. Never crash the engine.
                logger.warning("Phone notification failed: %s", exc)
