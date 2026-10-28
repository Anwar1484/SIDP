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
    engine = Subsystem3SafetyEngine(
        camera_id="tapo_c210",
        zone_name="test room",
        audio_enabled=False,
        telegram_enabled=False,
        cooldown_seconds=15.0,
    )
    try:
        # 1. Standing -> SAFE
        t_stand = engine.evaluate(_dummy_telemetry(1, PostureClass.STANDING, torso_angle=10.0), now_perf=10.0)
        assert t_stand.safety_tier == SafetyTier.SAFE

        # 2. Severe Bending (>60 deg) at t=20.0s -> WARNING initially (< 3.0s)
        t_bend_start = engine.evaluate(_dummy_telemetry(1, PostureClass.BENDING, torso_angle=68.0), now_perf=20.0)
        assert t_bend_start.safety_tier == SafetyTier.WARNING

        # At t=22.0s (2.0s elapsed < 3.0s) -> still WARNING
        t_bend_mid = engine.evaluate(_dummy_telemetry(1, PostureClass.BENDING, torso_angle=68.0), now_perf=22.0)
        assert t_bend_mid.safety_tier == SafetyTier.WARNING

        # At t=23.2s (3.2s elapsed >= 3.0s threshold) -> escalates from WARNING to UNSAFE
        t_bend_unsafe = engine.evaluate(_dummy_telemetry(1, PostureClass.BENDING, torso_angle=68.0), now_perf=23.2)
        assert t_bend_unsafe.safety_tier == SafetyTier.UNSAFE
        assert t_bend_unsafe.risk_score >= 60

        # 3. Falling -> immediate ALERT tier (force-alert posture)
        t_fall = engine.evaluate(_dummy_telemetry(1, PostureClass.FALLING, torso_angle=80.0, vy_mps=2.1), now_perf=24.0)
        assert t_fall.safety_tier == SafetyTier.ALERT
        assert t_fall.risk_score >= 95

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
