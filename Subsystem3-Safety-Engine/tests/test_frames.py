import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alert_engine.frames import FolderFrameSource

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def write_frame(folder, camera_id="cam-01", age_s=0.0):
    path = folder / f"{camera_id}.jpg"
    path.write_bytes(b"jpeg-bytes")
    stamp = (NOW - timedelta(seconds=age_s)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def test_returns_recent_frame(tmp_path):
    write_frame(tmp_path, age_s=1.0)
    assert FolderFrameSource(tmp_path, max_age_s=5).get_frame("cam-01", NOW) == b"jpeg-bytes"


def test_stale_frame_is_not_used(tmp_path):
    write_frame(tmp_path, age_s=30.0)
    assert FolderFrameSource(tmp_path, max_age_s=5).get_frame("cam-01", NOW) is None


def test_missing_frame_returns_none(tmp_path):
    assert FolderFrameSource(tmp_path, max_age_s=5).get_frame("cam-01", NOW) is None


def test_camera_id_cannot_escape_frame_folder(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    write_frame(tmp_path, camera_id="secret")
    assert FolderFrameSource(frames, max_age_s=5).get_frame("../secret", NOW) is None


def test_read_retries_while_windows_locks_the_file(tmp_path, monkeypatch):
    write_frame(tmp_path)
    real_read = Path.read_bytes
    calls = []

    def locked_once(path):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("file in use")
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", locked_once)
    assert FolderFrameSource(tmp_path, max_age_s=5).get_frame("cam-01", NOW) == b"jpeg-bytes"


def test_permanently_locked_file_gives_up(tmp_path, monkeypatch):
    write_frame(tmp_path)

    def always_locked(path):
        raise PermissionError("file in use")

    monkeypatch.setattr(Path, "read_bytes", always_locked)
    assert FolderFrameSource(tmp_path, max_age_s=5).get_frame("cam-01", NOW) is None
