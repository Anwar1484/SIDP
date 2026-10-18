from app.tracking.person_tracker import PersonTracker, create_tracker


def test_tracker_factory_honors_config(config, monkeypatch):
    assert isinstance(create_tracker(config.tracker), PersonTracker)
    selected = object()
    monkeypatch.setattr("app.tracking.byte_tracker.ByteTrackTracker", lambda _: selected)
    config.tracker.backend = "bytetrack"
    assert create_tracker(config.tracker) is selected


def test_detection_order_does_not_change_identity(config, make_pose):
    tracker = PersonTracker(config.tracker)
    left, right = make_pose(offset=(-150, 0)), make_pose(offset=(200, 0))
    first = tracker.update([left, right], 0, (640, 480))
    assert [t.id for t in first] == [1, 2]
    second = tracker.update([make_pose(offset=(205, 0)), make_pose(offset=(-145, 0))], 0.1, (640, 480))
    assert [t.id for t in second] == [2, 1]


def test_expiry_and_ids_never_reused(config, make_pose):
    tracker = PersonTracker(config.tracker)
    assert tracker.update([make_pose()], 0, (640, 480))[0].id == 1
    assert tracker.update([], 0.2, (640, 480)) == []
    assert tracker.update([make_pose()], 2, (640, 480))[0].id == 2
    assert tracker.expired_ids == [1]


def test_distant_detection_does_not_hijack_track(config, make_pose):
    tracker = PersonTracker(config.tracker)
    tracker.update([make_pose(offset=(-150, 0))], 0, (640, 480))
    assert tracker.update([make_pose(offset=(210, 0))], 0.1, (640, 480))[0].id == 2
