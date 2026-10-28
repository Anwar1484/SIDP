"""Live terminal view of alert/events and alert/status/cameras.

Usage:
    python tools/monitor.py
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

ALERT_EVENTS_TOPIC = "alert/events"
CAMERA_STATUS_TOPIC = "alert/status/cameras"

TIER_LABEL = {"safe": "SAFE", "warning": "WARN", "unsafe": "UNSAFE", "alert": "ALERT!!"}


def format_alert(payload: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    tier = TIER_LABEL.get(payload.get("tier", ""), payload.get("tier", "?"))
    return (
        f"[{now}] {tier:<8} person={payload.get('person_id'):<8} "
        f"camera={payload.get('camera_id'):<8} zone={payload.get('zone') or '-':<10} "
        f"posture={payload.get('posture'):<10} score={payload.get('risk_score'):>3} "
        f"duration={payload.get('duration_s', 0):.1f}s  {'; '.join(payload.get('reasons', []))}"
    )


def format_camera_status(snapshot: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    lines = [f"[{now}] CAMERA STATUS"]
    for camera_id, info in sorted(snapshot.items()):
        status = info.get("status", "?")
        marker = "UP  " if status == "monitored" else "DOWN"
        lines.append(f"    {marker} {camera_id}  (last seen {info.get('last_seen') or 'never'})")
    return "\n".join(lines)


def on_connect(client, userdata, flags, reason_code, properties=None) -> None:
    print(f"Connected (reason_code={reason_code}). Watching {ALERT_EVENTS_TOPIC} and {CAMERA_STATUS_TOPIC}")
    client.subscribe(ALERT_EVENTS_TOPIC, qos=1)
    client.subscribe(CAMERA_STATUS_TOPIC, qos=1)


def on_message(client, userdata, message: mqtt.MQTTMessage) -> None:
    try:
        payload = json.loads(message.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print(f"[?] could not decode payload on {message.topic}: {message.payload!r}")
        return

    if message.topic == ALERT_EVENTS_TOPIC:
        print(format_alert(payload))
    else:
        print(format_camera_status(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"monitor-{uuid.uuid4().hex[:8]}")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.host, args.port)
    except (ConnectionRefusedError, OSError) as exc:
        print(f"error: could not connect to MQTT broker at {args.host}:{args.port} ({exc})", file=sys.stderr)
        return 1

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping monitor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
