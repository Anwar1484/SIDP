from pathlib import Path

import pytest

from alert_engine.audio import AudioController
from alert_engine.config import AudioConfig


def make_config(messages=None) -> AudioConfig:
    return AudioConfig(
        default_device=None,
        messages=messages or {"unsafe": "Warning. Unsafe posture in the {zone}.", "alert": "Alert in the {zone}."},
    )


class FakeTtsEngine:
    def __init__(self, calls: list, fail: bool = False):
        self._calls = calls
        self._fail = fail

    def save_to_file(self, text: str, path: str) -> None:
        if self._fail:
            raise RuntimeError("tts engine broken")
        self._calls.append((text, path))
        Path(path).write_bytes(b"FAKEWAV")

    def runAndWait(self) -> None:
        pass


def working_tts_factory(calls: list):
    return lambda: FakeTtsEngine(calls)


def failing_tts_factory():
    return lambda: FakeTtsEngine([], fail=True)


def make_controller(tmp_path, **overrides) -> AudioController:
    defaults = dict(
        config=make_config(),
        assets_dir=tmp_path,
        tts_factory=working_tts_factory([]),
        player=lambda path, device: None,
        device_lister=lambda: [],
    )
    defaults.update(overrides)
    return AudioController(**defaults)


def test_enqueue_without_repeat_queues_once(tmp_path):
    controller = make_controller(tmp_path)
    controller.enqueue("unsafe", "lounge", repeat=False)
    assert controller._queue.qsize() == 1


def test_enqueue_with_repeat_queues_twice(tmp_path):
    controller = make_controller(tmp_path)
    controller.enqueue("alert", "lounge", repeat=True)
    assert controller._queue.qsize() == 2


def test_wav_generated_once_and_cached(tmp_path):
    calls: list = []
    controller = make_controller(tmp_path, tts_factory=working_tts_factory(calls))
    controller._get_or_generate_wav("unsafe", "lounge")
    controller._get_or_generate_wav("unsafe", "lounge")
    assert len(calls) == 1


def test_different_zones_generate_different_wavs(tmp_path):
    calls: list = []
    controller = make_controller(tmp_path, tts_factory=working_tts_factory(calls))
    path_a = controller._get_or_generate_wav("unsafe", "lounge")
    path_b = controller._get_or_generate_wav("unsafe", "bathroom")
    assert path_a != path_b
    assert len(calls) == 2


def test_falls_back_to_default_wav_when_tts_fails(tmp_path):
    calls: list = []
    controller = make_controller(tmp_path, tts_factory=working_tts_factory(calls))
    controller.start()
    controller.stop()

    controller._tts_factory = failing_tts_factory()
    path = controller._get_or_generate_wav("unsafe", "brand-new-zone")
    assert path == controller._default_path("unsafe")


def test_raises_when_no_fallback_available(tmp_path):
    controller = make_controller(tmp_path, tts_factory=failing_tts_factory())
    with pytest.raises(FileNotFoundError):
        controller._get_or_generate_wav("unsafe", "lounge")


def test_worker_still_plays_tone_when_speech_wav_unavailable(tmp_path):
    """Tone synthesis is independent of TTS, so a broken speech engine should still play the beep."""
    played = []
    controller = make_controller(
        tmp_path, tts_factory=failing_tts_factory(), player=lambda path, device: played.append(path)
    )
    controller.start()
    controller.enqueue("unsafe", "lounge", repeat=False)  # no speech WAV possible: must not crash the worker
    controller.enqueue("alert", "lounge", repeat=False)  # worker must still be alive for the next job
    controller._queue.join()
    controller.stop()
    assert [p.name for p in played] == ["_tone_unsafe.wav", "_tone_alert.wav"]


def test_select_known_device_switches(tmp_path):
    controller = make_controller(tmp_path, device_lister=lambda: [{"device_id": 3, "name": "Speaker"}])
    controller.select_device(3)
    assert controller._selected_device == 3


def test_select_unknown_device_falls_back_to_os_default(tmp_path):
    controller = make_controller(tmp_path, device_lister=lambda: [{"device_id": 3, "name": "Speaker"}])
    controller._selected_device = 3
    controller.select_device(99)
    assert controller._selected_device is None


def test_start_publishes_device_list(tmp_path):
    published = []
    controller = make_controller(
        tmp_path,
        device_lister=lambda: [{"device_id": 1, "name": "A"}],
        publish_devices=lambda devices: published.append(devices),
    )
    controller.start()
    controller.stop()
    assert published == [[{"device_id": 1, "name": "A"}]]


def test_worker_thread_plays_tone_then_speech(tmp_path):
    played = []
    controller = make_controller(tmp_path, player=lambda path, device: played.append((path, device)))
    controller.start()
    controller.enqueue("unsafe", "lounge", repeat=False)
    controller._queue.join()
    controller.stop()
    assert len(played) == 2
    assert played[0][0].name == "_tone_unsafe.wav"
    assert played[1][0].name.startswith("unsafe_lounge")
    assert played[0][1] is None and played[1][1] is None


def test_worker_plays_tone_and_speech_twice_when_repeated(tmp_path):
    played = []
    controller = make_controller(tmp_path, player=lambda path, device: played.append(path.name))
    controller.start()
    controller.enqueue("alert", "lounge", repeat=True)
    controller._queue.join()
    controller.stop()
    assert played == ["_tone_alert.wav", "alert_lounge.wav", "_tone_alert.wav", "alert_lounge.wav"]


def test_alert_tone_pattern_differs_from_unsafe(tmp_path):
    controller = make_controller(tmp_path)
    controller._ensure_tone("unsafe")
    controller._ensure_tone("alert")
    unsafe_bytes = (tmp_path / "_tone_unsafe.wav").read_bytes()
    alert_bytes = (tmp_path / "_tone_alert.wav").read_bytes()
    assert unsafe_bytes != alert_bytes


def test_worker_falls_back_to_os_default_when_playback_fails(tmp_path):
    attempts = []

    def flaky_player(path, device):
        attempts.append(device)
        if device is not None:
            raise RuntimeError("device busy")

    controller = make_controller(tmp_path, player=flaky_player)
    controller._selected_device = 7
    controller.start()
    controller.enqueue("unsafe", "lounge", repeat=False)
    controller._queue.join()
    controller.stop()
    assert attempts == [7, None, 7, None]  # tone attempt+fallback, then speech attempt+fallback


def test_pregenerate_defaults_creates_one_wav_per_message(tmp_path):
    calls: list = []
    controller = make_controller(tmp_path, tts_factory=working_tts_factory(calls))
    controller.start()
    controller.stop()
    assert (tmp_path / "unsafe_default.wav").exists()
    assert (tmp_path / "alert_default.wav").exists()
