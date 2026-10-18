"""Official ByteTrack association with session-local IDs and wall-time expiry."""
from types import SimpleNamespace

import numpy as np

from app.config import TrackerConfig
from app.runtime import prepare_runtime
from app.schemas import PersonPose
from app.tracking.person_tracker import Track


class ByteTrackTracker:
    def __init__(self, config: TrackerConfig):
        prepare_runtime()
        from ultralytics.trackers.byte_tracker import BYTETracker

        class SessionByteTracker(BYTETracker):
            @staticmethod
            def reset_id():
                # Ultralytics' global ID reset would affect another live session.
                # Native IDs may grow globally; we map them to local IDs below.
                pass

        self.config = config
        self.native = SessionByteTracker(SimpleNamespace(
            track_high_thresh=config.high_threshold, track_low_thresh=config.low_threshold,
            new_track_thresh=config.high_threshold, track_buffer=300,
            match_thresh=config.match_threshold, fuse_score=True,
        ))
        self.tracks: dict[int, Track] = {}
        self.mapping: dict[int, int] = {}
        self.next_id = 1
        self.expired_ids: list[int] = []
        self.last_timestamp: float | None = None

    def update(self, poses: list[PersonPose], timestamp: float, frame_size: tuple[int, int]) -> list[Track]:
        from ultralytics.engine.results import Boxes

        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("tracking timestamps must increase")
        self.last_timestamp = timestamp
        self.expired_ids = [key for key, track in self.tracks.items()
                            if timestamp - track.last_seen > self.config.stale_seconds]
        expired_native = {key for key, value in self.mapping.items() if value in self.expired_ids}
        for name in ("tracked_stracks", "lost_stracks", "removed_stracks"):
            setattr(self.native, name, [t for t in getattr(self.native, name)
                                       if t.track_id not in expired_native])
        for identity in self.expired_ids:
            self.tracks.pop(identity, None)
        for identity in expired_native:
            self.mapping.pop(identity, None)
        rows = np.array([p.bbox.as_list() + [p.detection_confidence, 0] for p in poses],
                        dtype=np.float32).reshape(-1, 6)
        detected = Boxes(rows, (frame_size[1], frame_size[0]))
        observed = self.native.update(detected)
        current = []
        for row in observed:
            native_id, index = int(row[4]), int(row[-1])
            if not 0 <= index < len(poses):
                raise RuntimeError("ByteTrack returned an invalid pose index")
            if native_id not in self.mapping:
                self.mapping[native_id] = self.next_id
                self.next_id += 1
            identity = self.mapping[native_id]
            track = Track(identity, poses[index], timestamp)
            self.tracks[identity] = track
            current.append(track)
        return sorted(current, key=lambda t: t.id)
