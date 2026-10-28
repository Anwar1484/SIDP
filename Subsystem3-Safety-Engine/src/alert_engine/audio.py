"""TTS-to-WAV generation, device listing/selection, and a serial playback queue (section 8.5).

pyttsx3 always plays through the OS default device, so it is only used to
render WAV files. Those WAV files are then played with sounddevice on the
selected device, one at a time on a worker thread, so alerts never overlap
or block the dispatcher.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
from pathlib import Path
from typing import Callable, Protocol

from alert_engine.config import AudioConfig

logger = logging.getLogger(__name__)


class TtsEngine(Protocol):
    def save_to_file(self, text: str, path: str) -> None: ...
    def runAndWait(self) -> None: ...


TtsFactory = Callable[[], TtsEngine]
Player = Callable[[Path, int | None], None]
DeviceLister = Callable[[], list[dict]]
DevicesPublisher = Callable[[list[dict]], None]


def _default_tts_factory() -> TtsEngine:
    import pyttsx3

    return pyttsx3.init()


def _default_player(path: Path, device: int | None) -> None:
    import soundfile
    import sounddevice

    data, samplerate = soundfile.read(str(path))
    sounddevice.play(data, samplerate, device=device)
    sounddevice.wait()


def list_output_devices() -> list[dict]:
    """List playback-capable audio devices as [{"device_id": int, "name": str}, ...]."""
    import sounddevice

    devices = []
    for idx, info in enumerate(sounddevice.query_devices()):
        if info.get("max_output_channels", 0) > 0:
            devices.append({"device_id": idx, "name": info.get("name", f"device-{idx}")})
    return devices


_default_device_lister = list_output_devices


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "default"


TONE_SAMPLERATE = 22050


def _make_beep_tone(tier: str, samplerate: int = TONE_SAMPLERATE):
    """Synthesizes a short attention tone played before the spoken message.

    "alert" gets a faster, higher-pitched triple beep (more urgent, like a
    smoke detector); everything else gets a slower double beep. Plain TTS
    speech alone was too easy to miss, hence this.
    """
    import numpy as np

    def beep(freq: float, duration_s: float, amplitude: float = 0.9):
        t = np.linspace(0, duration_s, int(samplerate * duration_s), endpoint=False)
        fade = int(samplerate * 0.008)
        envelope = np.ones_like(t)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        return (amplitude * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    gap = np.zeros(int(samplerate * 0.07), dtype=np.float32)
    trailing_silence = np.zeros(int(samplerate * 0.15), dtype=np.float32)
    if tier == "alert":
        unit = beep(1400.0, 0.11)
        beeps = [unit, gap, unit, gap, unit]
    else:
        unit = beep(900.0, 0.18)
        beeps = [unit, gap, unit]
    return np.concatenate(beeps + [trailing_silence])


class AudioController:
    """Implements the AudioPort the dispatcher expects: enqueue(tier, zone, repeat)."""

    def __init__(
        self,
        config: AudioConfig,
        assets_dir: str | Path = "assets/audio",
        tts_factory: TtsFactory = _default_tts_factory,
        player: Player = _default_player,
        device_lister: DeviceLister = _default_device_lister,
        publish_devices: DevicesPublisher | None = None,
    ) -> None:
        self._config = config
        self._assets_dir = Path(assets_dir)
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        self._tts_factory = tts_factory
        self._player = player
        self._device_lister = device_lister
        self._publish_devices = publish_devices

        self._selected_device: int | None = config.default_device
        self._queue: queue.Queue[tuple[str, str | None] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="audio-playback")

    def start(self) -> None:
        """Pre-generate default WAV clips and attention tones, publish the device list, and start the playback thread."""
        self._pregenerate_defaults()
        for tier in self._config.messages:
            self._ensure_tone(tier)
        if self._publish_devices is not None:
            try:
                self._publish_devices(self.list_devices())
            except Exception:
                logger.exception("Failed to publish audio device list")
        self._worker.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._worker.join(timeout=timeout)

    def list_devices(self) -> list[dict]:
        return self._device_lister()

    def select_device(self, device_id: int) -> None:
        """Switch playback device; falls back to the OS default if device_id is unknown."""
        known_ids = {d["device_id"] for d in self.list_devices()}
        if device_id in known_ids:
            self._selected_device = device_id
            logger.info("Audio output switched to device %d", device_id)
        else:
            logger.warning("Unknown audio device_id %d selected, falling back to OS default", device_id)
            self._selected_device = None

    def enqueue(self, tier: str, zone: str | None, repeat: bool) -> None:
        """Queue a message for tier/zone; queues it twice if repeat is True.

        Only puts a job descriptor on the queue: WAV lookup/generation and
        playback both happen on the worker thread, so this never blocks the
        calling (dispatcher) thread.
        """
        self._queue.put((tier, zone))
        if repeat:
            self._queue.put((tier, zone))

    def _pregenerate_defaults(self) -> None:
        for tier, template in self._config.messages.items():
            path = self._default_path(tier)
            if path.exists():
                continue
            text = template.format(zone="the area")
            try:
                self._generate_wav(text, path)
            except Exception:
                logger.exception("Could not pre-generate default WAV for tier %s", tier)

    def _default_path(self, tier: str) -> Path:
        return self._assets_dir / f"{tier}_default.wav"

    def _tone_path(self, tier: str) -> Path:
        return self._assets_dir / f"_tone_{tier}.wav"

    def _ensure_tone(self, tier: str) -> Path | None:
        """Synthesizes the attention-tone WAV for a tier if it doesn't exist yet.

        Independent of TTS entirely (pure numpy synthesis), so the tone can
        still play even if the speech engine is completely broken.
        """
        path = self._tone_path(tier)
        if path.exists():
            return path
        try:
            import soundfile

            soundfile.write(str(path), _make_beep_tone(tier), TONE_SAMPLERATE)
            return path
        except Exception:
            logger.exception("Could not synthesize attention tone for tier %s", tier)
            return None

    def _wav_path(self, tier: str, zone: str | None) -> Path:
        return self._assets_dir / f"{tier}_{_slugify(zone or 'default')}.wav"

    def _get_or_generate_wav(self, tier: str, zone: str | None) -> Path:
        template = self._config.messages.get(tier)
        if template is None:
            raise ValueError(f"No audio message template configured for tier '{tier}'")

        path = self._wav_path(tier, zone)
        if path.exists():
            return path

        text = template.format(zone=zone or "the area")
        try:
            self._generate_wav(text, path)
            return path
        except Exception:
            logger.exception("TTS generation failed for tier=%s zone=%s, looking for a fallback WAV", tier, zone)
            return self._fallback_wav(tier)

    def _fallback_wav(self, tier: str) -> Path:
        default_path = self._default_path(tier)
        if default_path.exists():
            return default_path
        candidates = sorted(self._assets_dir.glob(f"{tier}_*.wav"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No pre-recorded WAV available for tier '{tier}'")

    def _generate_wav(self, text: str, path: Path) -> None:
        engine = self._tts_factory()
        set_property = getattr(engine, "setProperty", None)
        if set_property is not None:
            try:
                set_property("volume", 1.0)
            except Exception:
                pass
        engine.save_to_file(text, str(path))
        engine.runAndWait()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            tier, zone = item

            tone_path = self._tone_path(tier)
            if tone_path.exists():
                self._safe_play(tone_path)

            try:
                wav_path = self._get_or_generate_wav(tier, zone)
            except Exception:
                logger.exception(
                    "No spoken WAV available for tier=%s zone=%s; attention tone still played", tier, zone
                )
                self._queue.task_done()
                continue

            self._safe_play(wav_path)
            self._queue.task_done()

    def _safe_play(self, path: Path) -> None:
        try:
            self._player(path, self._selected_device)
        except Exception:
            logger.exception("Playback failed on device %s, retrying on OS default", self._selected_device)
            try:
                self._player(path, None)
            except Exception:
                logger.exception("Fallback playback on OS default device also failed")
