"""Unit tests for subsystem1_2_3_master.py (Subsystems 1 + 2 + 3 Integration)."""

import numpy as np

from subsystem1_2_3_master import (
    AutoTrackTelemetry,
    CameraMode,
    DatasetCollector,
    PersonTelemetry,
    PostureClass,
    SafetyTier,
    Subsystem3SafetyEngine,
    UnifiedPTZController,
    render_master_hud,
)


def _dummy_telemetry(
    track_id: int,
    posture: PostureClass,
    torso_angle: float | None = 12.0,
    vy_mps: float = 0.0,
) -> PersonTelemetry:
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[:, 2] = 0.9
    return PersonTelemetry(
        track_id=track_id,
        person_id=f"p-{track_id:03d}",
        bbox=(200.0, 100.0, 340.0, 420.0),
        confidence=0.92,
        keypoints_xyc=kps,
        posture=posture,
        torso_angle_deg=torso_angle,
        knee_angle_deg=170.0,
        vertical_velocity_mps=vy_mps,
        rgb_fall_conf=None,
    )


def test_subsystem3_four_tier_state_machine_and_3s_bending_escalation(tmp_path):
    captures_root = tmp_path / "captures"
    engine = Subsystem3SafetyEngine(
        camera_id="tapo_c210",
        zone_name="test room",
        audio_enabled=False,
        telegram_enabled=False,
        cooldown_seconds=15.0,
        captures_root=captures_root,
    )
    try:
        # 1. Standing -> SAFE (no snapshot archived)
        t_stand = engine.evaluate(_dummy_telemetry(1, PostureClass.STANDING, torso_angle=10.0), now_perf=10.0)
        assert t_stand.safety_tier == SafetyTier.SAFE
        assert t_stand.snapshot_path is None

        # 2. Severe Bending (>60 deg) at t=20.0s -> WARNING initially (< 3.0s) -> archives to warning/
        t_bend_start = engine.evaluate(_dummy_telemetry(1, PostureClass.BENDING, torso_angle=68.0), now_perf=20.0)
        assert t_bend_start.safety_tier == SafetyTier.WARNING
        assert t_bend_start.snapshot_path is not None
        assert "/warning/" in t_bend_start.snapshot_path
        assert t_bend_start.snapshot_path.endswith("_resident_1.jpg")

        # At t=22.0s (2.0s elapsed < 3.0s, within 15s cooldown) -> still WARNING, suppressed duplicate snapshot
        t_bend_mid = engine.evaluate(_dummy_telemetry(1, PostureClass.BENDING, torso_angle=68.0), now_perf=22.0)
        assert t_bend_mid.safety_tier == SafetyTier.WARNING
        assert t_bend_mid.snapshot_path is None

        # At t=23.2s (3.2s elapsed >= 3.0s threshold) -> escalates from WARNING to UNSAFE -> archives to unsafe/
        t_bend_unsafe = engine.evaluate(_dummy_telemetry(1, PostureClass.BENDING, torso_angle=68.0), now_perf=23.2)
        assert t_bend_unsafe.safety_tier == SafetyTier.UNSAFE
        assert t_bend_unsafe.risk_score >= 60
        assert t_bend_unsafe.snapshot_path is not None
        assert "/unsafe/" in t_bend_unsafe.snapshot_path
        assert t_bend_unsafe.snapshot_path.endswith("_resident_1.jpg")

        # 3. Falling -> immediate ALERT tier -> archives to alert/
        t_fall = engine.evaluate(_dummy_telemetry(1, PostureClass.FALLING, torso_angle=80.0, vy_mps=2.1), now_perf=24.0)
        assert t_fall.safety_tier == SafetyTier.ALERT
        assert t_fall.risk_score >= 95
        assert t_fall.snapshot_path is not None
        assert "/alert/" in t_fall.snapshot_path
        assert t_fall.snapshot_path.endswith("_resident_1.jpg")

        engine.snapshot_archiver.flush()
        for snap_p in (t_bend_start.snapshot_path, t_bend_unsafe.snapshot_path, t_fall.snapshot_path):
            assert snap_p is not None
            from pathlib import Path
            assert Path(snap_p).is_file()

        # 4. HUD rendering test across all tiers
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ptz = UnifiedPTZController(enabled=False, initial_mode=CameraMode.AUTOMATIC)
        collector = DatasetCollector(root_dir=tmp_path / "dataset")
        try:
            hud = render_master_hud(
                frame=frame,
                persons=[t_stand, t_bend_start, t_bend_unsafe, t_fall],
                ptz=ptz,
                auto_telemetry=AutoTrackTelemetry(
                    active=True,
                    occupant_count=4,
                    group_bbox=(180, 90, 380, 430),
                    collective_center=(280, 260),
                    error_x=-0.12,
                    error_y=0.08,
                    in_deadband=True,
                ),
                fps=28.5,
                ai_overlay_enabled=True,
                collector=collector,
            )
            assert hud.shape == (480, 640, 3)
        finally:
            ptz.shutdown()
    finally:
        engine.shutdown()


def test_s2_to_s3_payload_interface_and_mahdi_activity_and_frames(tmp_path):
    """
    Verifies:
      1. The canonical S2 -> S3 dictionary interface:
         `{person_id: {"bbox": [...], "keypoints": [[x, y, conf], ...], "posture": str}}`
      2. Automatic torso inclination angle calculation from 17 COCO keypoints.
      3. Mahdi's `activity_log` table & `get_activity_summary()` + `UNMONITORED` tracking.
      4. Mahdi's `FolderFrameSource` camera frame attachment sync (`update_camera_frame`).
    """
    from datetime import datetime, timezone

    engine = Subsystem3SafetyEngine(
        camera_id="tapo_c210_test",
        zone_name="living room",
        audio_enabled=False,
        telegram_enabled=False,
        cooldown_seconds=15.0,
        captures_root=tmp_path / "captures",
    )
    try:
        # Build 17 COCO keypoints representing a ~65 deg forward bend:
        # Hip at (200, 300), Shoulder at (290, 258) -> dx=90, dy=42 -> ~65 deg
        kps = [[0.0, 0.0, 0.0] for _ in range(17)]
        kps[5] = [290.0, 258.0, 0.95]   # left shoulder
        kps[6] = [290.0, 258.0, 0.95]   # right shoulder
        kps[11] = [200.0, 300.0, 0.95]  # left hip
        kps[12] = [200.0, 300.0, 0.95]  # right hip

        s2_payload = {
            "p-007": {
                "bbox": [160.0, 140.0, 320.0, 410.0],
                "keypoints": kps,
                "posture": "bending",
            }
        }

        out_start = engine.evaluate_payload(s2_payload, now_perf=100.0)
        assert len(out_start) == 1
        p0 = out_start[0]
        assert p0.person_id == "p-007"
        assert p0.track_id == 7
        assert p0.torso_angle_deg is not None and p0.torso_angle_deg > 60.0
        assert p0.safety_tier == SafetyTier.WARNING

        # Sustain bending > 3.0s -> escalates to UNSAFE
        out_unsafe = engine.evaluate_payload(s2_payload, now_perf=103.5)
        assert out_unsafe[0].safety_tier == SafetyTier.UNSAFE

        # Test camera frame sync for FolderFrameSource
        dummy_bgr = np.full((240, 320, 3), 128, dtype=np.uint8)
        engine.update_camera_frame(dummy_bgr, now_perf=104.0)
        jpg_bytes = engine.frame_source.get_frame("tapo_c210_test", datetime.now(timezone.utc))
        assert jpg_bytes is not None and jpg_bytes.startswith(b"\xff\xd8")

        # Reset resident -> writes UNMONITORED to activity_log
        engine.reset_resident("p-007")
        summary = engine.get_activity_summary("p-007")
        assert summary is not None
        assert summary.person_id == "p-007"
    finally:
        engine.shutdown()


