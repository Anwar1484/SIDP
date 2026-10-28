"""Thin wrapper around paho-mqtt (v2 callback API) with reconnect handling."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

MessageHandler = Callable[[str, bytes], None]


@dataclass
class _Subscription:
    qos: int
    handler: MessageHandler


class MqttClient:
    """Wraps a paho-mqtt client: connect/reconnect, topic subscriptions, publish.

    Callers register handlers per topic with `subscribe()`; this class takes
    care of resubscribing on reconnect and dispatching incoming messages
    (including wildcard subscriptions) to the right handler.
    """

    def __init__(self, host: str, port: int, client_id: str) -> None:
        self._host = host
        self._port = port
        self._subscriptions: dict[str, _Subscription] = {}
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

    def subscribe(self, topic: str, qos: int, handler: MessageHandler) -> None:
        """Register a handler for a topic. Safe to call before connect()."""
        self._subscriptions[topic] = _Subscription(qos=qos, handler=handler)
        if self._client.is_connected():
            self._client.subscribe(topic, qos=qos)

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> None:
        self._client.publish(topic, payload, qos=qos, retain=retain)

    def connect(self) -> None:
        self._client.connect(self._host, self._port)

    def loop_start(self) -> None:
        self._client.loop_start()

    def loop_forever(self) -> None:
        self._client.loop_forever()

    def loop_stop(self) -> None:
        self._client.loop_stop()

    def disconnect(self) -> None:
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if reason_code == 0:
            logger.info("Connected to MQTT broker at %s:%s", self._host, self._port)
            for topic, sub in self._subscriptions.items():
                client.subscribe(topic, qos=sub.qos)
        else:
            logger.error("MQTT connect failed: %s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None) -> None:
        logger.warning("Disconnected from MQTT broker: %s", reason_code)

    def _on_message(self, client, userdata, message: mqtt.MQTTMessage) -> None:
        for topic, sub in self._subscriptions.items():
            if mqtt.topic_matches_sub(topic, message.topic):
                try:
                    sub.handler(message.topic, message.payload)
                except Exception:
                    logger.exception("Handler for topic %s raised", message.topic)
                return
        logger.warning("No handler registered for topic %s", message.topic)
