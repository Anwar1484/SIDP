"""Lists available speakers and publishes a device choice to alert/audio/select.

Usage:
    python tools/select_audio.py --list
    python tools/select_audio.py --device-id 3
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import paho.mqtt.client as mqtt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alert_engine.audio import list_output_devices  # noqa: E402

TOPIC = "alert/audio/select"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--list", action="store_true", help="List available output devices and exit")
    parser.add_argument("--device-id", type=int, help="Device id to select (publishes to alert/audio/select)")
    args = parser.parse_args()

    devices = list_output_devices()
    if args.list or args.device_id is None:
        for device in devices:
            print(f"{device['device_id']:>3}  {device['name']}")
        if args.device_id is None:
            return 0

    if args.device_id not in {d["device_id"] for d in devices}:
        print(f"error: device_id {args.device_id} is not a known output device", file=sys.stderr)
        return 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"select-audio-{uuid.uuid4().hex[:8]}")
    try:
        client.connect(args.host, args.port)
    except (ConnectionRefusedError, OSError) as exc:
        print(f"error: could not connect to MQTT broker at {args.host}:{args.port} ({exc})", file=sys.stderr)
        return 1

    client.loop_start()
    client.publish(TOPIC, json.dumps({"device_id": args.device_id}), qos=1)
    client.loop_stop()
    client.disconnect()
    print(f"published device_id={args.device_id} to {TOPIC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
