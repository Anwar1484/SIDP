"""Entry point: wires the full pipeline from posture/events to alert/events."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from alert_engine.audio import AudioController
from alert_engine.config import Config, load_config
from alert_engine.debounce import DebounceTracker
from alert_engine.dispatcher import Dispatcher
from alert_engine.metrics import MetricsLogger
from alert_engine.mqtt_client import MqttClient
from alert_engine.schemas import AudioDeviceSelect, PostureEventIn, PostureRejectedError, validate_posture
from alert_engine.ml_scoring import HybridScorer
from alert_engine.notifier import Notifier
from alert_engine.sinks import EventSink, LocalSqliteSink
from alert_engine.watchdog import CameraSnapshot, CameraWatchdog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("alert_engine.main")

POSTURE_EVENTS_TOPIC = "posture/events"
ALERT_EVENTS_TOPIC = "alert/events"
AUDIO_DEVICES_TOPIC = "alert/audio/devices"
AUDIO_SELECT_TOPIC = "alert/audio/select"
CAMERA_STATUS_TOPIC = "alert/status/cameras"


class RejectionLogger:
    """Appends invalid payloads and their errors to logs/rejected.jsonl (section 8.1)."""

    def __init__(self, path: str | Path = "logs/rejected.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def log(self, raw_payload: str, error: str) -> None:
        self.count += 1
        record = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "raw_payload": raw_payload,
            "error": error,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def make_audio_select_handler(audio: AudioController):
    def handle(topic: str, payload: bytes) -> None:
        try:
            selection = AudioDeviceSelect.model_validate_json(payload)
        except ValidationError as exc:
            logger.warning("Ignoring malformed audio device selection: %s", exc)
            return
        audio.select_device(selection.device_id)

    return handle


class CameraOccupancy:
    """Tracks which person_ids have been seen on each camera_id.

    Used so that when a camera drops out and recovers, we know whose
    debounce state needs resetting (see DebounceTracker.reset_person).
    """

    def __init__(self) -> None:
        self._by_camera: dict[str, set[str]] = defaultdict(set)

    def touch(self, camera_id: str, person_id: str) -> None:
        self._by_camera[camera_id].add(person_id)

    def people_on(self, camera_id: str) -> set[str]:
        return self._by_camera.get(camera_id, set())


def make_camera_status_handler(
    sink: EventSink, debounce: DebounceTracker, occupancy: CameraOccupancy, publish_camera_status
):
    previous_status: dict[str, str] = {}

    def handle(snapshot: CameraSnapshot) -> None:
        publish_camera_status(json.dumps(snapshot))
        for camera_id, info in snapshot.items():
            status = info["status"]
            last_seen = datetime.fromisoformat(info["last_seen"]) if info["last_seen"] else None
            sink.write_camera_status(camera_id, status, last_seen)

            if status == "unmonitored" and previous_status.get(camera_id) != "unmonitored":
                # We have no evidence what happened on this camera during the
                # outage, so residents on it must re-earn confirmation from
                # scratch rather than resuming a duration_s spanning the gap.
                for person_id in occupancy.people_on(camera_id):
                    debounce.reset_person(person_id)
                    logger.info("Camera %s unmonitored; reset debounce state for person %s", camera_id, person_id)
            previous_status[camera_id] = status

    return handle


def parse_event(raw: bytes) -> PostureEventIn:
    """Decode + schema-validate a raw MQTT payload. Does not check the posture allow-list."""
    text = raw.decode("utf-8")
    data = json.loads(text)
    return PostureEventIn.model_validate(data)


def make_posture_handler(
    config: Config,
    rejection_logger: RejectionLogger,
    debounce: DebounceTracker,
    dispatcher: Dispatcher,
    watchdog: CameraWatchdog,
    metrics: MetricsLogger,
    occupancy: CameraOccupancy,
    scorer: HybridScorer,
    clock=lambda: datetime.now(timezone.utc),
):
    def handle(topic: str, payload: bytes) -> None:
        received_at = clock()
        raw_text = payload.decode("utf-8", errors="replace")

        try:
            event = parse_event(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Rejected event: %s", exc)
            rejection_logger.log(raw_text, str(exc))
            return

        # Touched even if the posture itself turns out to be unrecognised: a
        # structurally valid message from this camera proves it is still
        # transmitting, and the allowed-posture list is still being agreed
        # with the S2 team (CLAUDE.md section 7.2).
        watchdog.touch(event.camera_id, at=received_at)
        occupancy.touch(event.camera_id, event.person_id)

        try:
            validate_posture(event, config.postures)
        except PostureRejectedError as exc:
            logger.warning("Rejected event: %s", exc)
            rejection_logger.log(raw_text, str(exc))
            return

        result = debounce.process(event)
        if result.posture is None:
            # Not confirmed yet (still in "monitoring"/"suspected" with no baseline posture).
            return

        score = scorer.score(
            result.posture, result.duration_s, event.person_id,
            confidence=event.confidence, bend_angle_deg=event.bend_angle_deg,
        )
        outcome = dispatcher.handle(
            event_id=event.event_id,
            person_id=event.person_id,
            camera_id=event.camera_id,
            zone=event.zone,
            posture=result.posture,
            tier=score.tier,
            risk_score=score.risk_score,
            duration_s=result.duration_s,
            reasons=score.reasons,
            detected_at=event.timestamp,
            now=received_at,
        )
        logger.info(
            "person=%s posture=%s tier=%s score=%d duration=%.1fs status=%s alert=%s audio=%s",
            event.person_id, result.posture, score.tier, score.risk_score, result.duration_s,
            outcome.sink_status_written, outcome.sink_alert_written, outcome.audio_triggered,
        )

        if outcome.sink_alert_written:
            metrics.log_alert(
                alert_id=outcome.event.alert_id,
                detected_at=event.timestamp,
                received_at=received_at,
                confirmed_at=result.confirmed_at or event.timestamp,
                dispatched_at=outcome.event.dispatched_at,
            )

    return handle


def main() -> None:
    load_dotenv()
    config = load_config("config/config.yaml")
    notifier = Notifier.from_env(config.notify.tiers, enabled=config.notify.enabled, postures=config.notify.postures)
    rejection_logger = RejectionLogger()
    metrics = MetricsLogger()
    debounce = DebounceTracker(config.debounce)
    occupancy = CameraOccupancy()
    scorer = HybridScorer.from_config(config)
    logger.info("Risk scoring mode: %s", scorer.mode)

    client = MqttClient(host=config.mqtt.host, port=config.mqtt.port, client_id=config.mqtt.client_id)

    audio = AudioController(
        config=config.audio,
        publish_devices=lambda devices: client.publish(AUDIO_DEVICES_TOPIC, json.dumps(devices), qos=1, retain=True),
    )
    sink = LocalSqliteSink(sqlite_path=config.sink.sqlite_path)
    dispatcher = Dispatcher(
        config=config.dispatch,
        sink=sink,
        audio=audio,
        publish=lambda payload: client.publish(ALERT_EVENTS_TOPIC, payload, qos=1),
        notifier=notifier,
    )
    watchdog = CameraWatchdog(
        camera_timeout_s=config.watchdog.camera_timeout_s,
        on_status_change=make_camera_status_handler(
            sink, debounce, occupancy, lambda payload: client.publish(CAMERA_STATUS_TOPIC, payload, qos=1, retain=True)
        ),
    )

    client.subscribe(
        POSTURE_EVENTS_TOPIC,
        qos=0,
        handler=make_posture_handler(config, rejection_logger, debounce, dispatcher, watchdog, metrics, occupancy, scorer),
    )
    client.subscribe(AUDIO_SELECT_TOPIC, qos=1, handler=make_audio_select_handler(audio))

    logger.info("Connecting to MQTT broker at %s:%s", config.mqtt.host, config.mqtt.port)
    try:
        client.connect()
    except OSError as exc:
        logger.error("Could not connect to MQTT broker at %s:%s (%s)", config.mqtt.host, config.mqtt.port, exc)
        return

    audio.start()
    sink.start()
    notifier.start()
    watchdog.start_polling(interval_s=1.0)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down (rejected %d events this run)", rejection_logger.count)
    finally:
        watchdog.stop_polling()
        audio.stop()
        notifier.stop()
        sink.stop()
        client.disconnect()


if __name__ == "__main__":
    main()
