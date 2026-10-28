"""Publishes fake posture events to posture/events for manual and scenario testing.

Usage:
    python tools/simulate_publisher.py --scenario normal
    python tools/simulate_publisher.py --scenario fall --host localhost --port 1883

See CLAUDE.md section 10 for the scenario table.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("simulate_publisher")

TOPIC = "posture/events"
RATE_HZ = 10.0


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def make_event(
    camera_id: str,
    person_id: str,
    posture: str,
    confidence: float,
    zone: str | None = None,
    bend_angle_deg: float | None = None,
) -> dict:
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": now_iso(),
        "camera_id": camera_id,
        "person_id": person_id,
        "posture": posture,
        "confidence": confidence,
    }
    if zone is not None:
        event["zone"] = zone
    if bend_angle_deg is not None:
        event["bend_angle_deg"] = bend_angle_deg
    return event


@dataclass
class Frame:
    """One tick of a scenario: either a JSON-ready dict, or a raw string to send verbatim."""

    payload: dict | str
    delay_s: float = field(default=1.0 / RATE_HZ)


def scenario_normal() -> Iterator[Frame]:
    """Residents sitting, standing, walking. Expected: only safe tier, no audio."""
    sequence = ["sitting"] * 20 + ["standing"] * 20 + ["walking"] * 20
    for posture in sequence:
        yield Frame(make_event("cam-01", "p-001", posture, 0.9, zone="lounge"))


def scenario_glitch() -> Iterator[Frame]:
    """One single fall frame inside normal walking. Expected: no alert (R2)."""
    for i in range(60):
        if i == 30:
            yield Frame(make_event("cam-02", "p-002", "fall", 0.75, zone="hallway"))
        else:
            yield Frame(make_event("cam-02", "p-002", "walking", 0.9, zone="hallway"))


def scenario_bending() -> Iterator[Frame]:
    """Sustained bending for 8 s. Expected: warning, then unsafe with audio."""
    frame_count = int(8 * RATE_HZ)
    for _ in range(frame_count):
        yield Frame(make_event("cam-03", "p-003", "bending", 0.85, zone="bathroom", bend_angle_deg=68.0))


def scenario_fall() -> Iterator[Frame]:
    """Walking, then sustained fall. Expected: alert tier and audio within 2 s (R1)."""
    for _ in range(int(2 * RATE_HZ)):
        yield Frame(make_event("cam-04", "p-004", "walking", 0.9, zone="lounge"))
    for _ in range(int(5 * RATE_HZ)):
        yield Frame(make_event("cam-04", "p-004", "fall", 0.9, zone="lounge"))


def scenario_dropout() -> Iterator[Frame]:
    """Events stop for one camera for 8 s. Expected: flagged unmonitored, then recovers."""
    for _ in range(int(3 * RATE_HZ)):
        yield Frame(make_event("cam-05", "p-005", "walking", 0.9, zone="corridor"))
    yield Frame(payload={"__note__": "camera dropout"}, delay_s=8.0)
    for _ in range(int(3 * RATE_HZ)):
        yield Frame(make_event("cam-05", "p-005", "walking", 0.9, zone="corridor"))


def scenario_malformed() -> Iterator[Frame]:
    """Invalid JSON and missing fields mixed with valid events. Expected: engine keeps running (R5)."""
    valid = make_event("cam-06", "p-006", "standing", 0.9, zone="lounge")
    missing_field = dict(valid)
    del missing_field["person_id"]
    bad_confidence = dict(valid)
    bad_confidence["confidence"] = 5.0
    frames: list[dict | str] = [
        valid,
        "{not valid json",
        missing_field,
        bad_confidence,
        "",
        valid,
        json.dumps(valid) + "garbage-trailer",
    ]
    for payload in frames:
        yield Frame(payload)


def scenario_mixed() -> Iterator[Frame]:
    """All scenarios interleaved for 3+ residents. Used for the demo."""
    generators = [
        scenario_normal(),
        scenario_glitch(),
        scenario_bending(),
        scenario_fall(),
        scenario_dropout(),
        scenario_malformed(),
    ]
    exhausted = [False] * len(generators)
    while not all(exhausted):
        for i, gen in enumerate(generators):
            if exhausted[i]:
                continue
            try:
                yield next(gen)
            except StopIteration:
                exhausted[i] = True


SCENARIOS = {
    "normal": scenario_normal,
    "glitch": scenario_glitch,
    "bending": scenario_bending,
    "fall": scenario_fall,
    "dropout": scenario_dropout,
    "malformed": scenario_malformed,
    "mixed": scenario_mixed,
}


def publish_frame(client: mqtt.Client, frame: Frame) -> None:
    if isinstance(frame.payload, str):
        raw = frame.payload
    elif "__note__" in frame.payload:
        logger.info("(%s, sleeping %.1fs)", frame.payload["__note__"], frame.delay_s)
        time.sleep(frame.delay_s)
        return
    else:
        raw = json.dumps(frame.payload)
    client.publish(TOPIC, raw, qos=0)
    logger.info("published: %s", raw[:200])
    time.sleep(frame.delay_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-publisher-{uuid.uuid4().hex[:8]}")
    try:
        client.connect(args.host, args.port)
    except (ConnectionRefusedError, OSError) as exc:
        logger.error("Could not connect to MQTT broker at %s:%s (%s)", args.host, args.port, exc)
        return 1
    client.loop_start()

    logger.info("Running scenario '%s' -> topic %s", args.scenario, TOPIC)
    for frame in SCENARIOS[args.scenario]():
        publish_frame(client, frame)

    client.loop_stop()
    client.disconnect()
    logger.info("Scenario '%s' finished", args.scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
