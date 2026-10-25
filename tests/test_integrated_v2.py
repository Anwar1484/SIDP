"""Unit tests for subsystem1_2_integrated_v2.py (6-Class Posture Engine & Tri-Mode PTZ)."""

import numpy as np
import pytest

from subsystem1_2_integrated_v2 import (
    CameraMode,
    DatasetCollector,
    KP_L_ANKLE,
    KP_L_HIP,
    KP_L_KNEE,
    KP_L_SHOULDER,
    KP_R_ANKLE,
    KP_R_HIP,
    KP_R_KNEE,
    KP_R_SHOULDER,
    PostureClass,
    SixClassPostureEngine,
    TriModePTZController,
    render_osd_hud,
)


def _blank_kps() -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    return kps


def _set_kp(kps: np.ndarray, idx: int, x: float, y: float, conf: float = 0.92) -> None:
    kps[idx] = [x, y, conf]


def test_six_class_standing_sitting_bending_lying(tmp_path):
    engine = SixClassPostureEngine(kp_conf_threshold=0.30)

    # 1. STANDING: upright torso (<25 deg), extended knees, stationary
    kps_stand = _blank_kps()
    _set_kp(kps_stand, KP_L_SHOULDER, 300, 140)
    _set_kp(kps_stand, KP_R_SHOULDER, 340, 140)
    _set_kp(kps_stand, KP_L_HIP, 305, 250)
    _set_kp(kps_stand, KP_R_HIP, 335, 250)
    _set_kp(kps_stand, KP_L_KNEE, 305, 340)
    _set_kp(kps_stand, KP_R_KNEE, 335, 340)
    _set_kp(kps_stand, KP_L_ANKLE, 305, 430)
    _set_kp(kps_stand, KP_R_ANKLE, 335, 430)
    res_stand = engine.evaluate_person(
        track_id=1,
        bbox=(280.0, 100.0, 360.0, 440.0),
        det_conf=0.95,
        keypoints_xyc=kps_stand,
        timestamp=1.0,
        frame_height=480,
    )
    assert res_stand.posture == PostureClass.STANDING

    # 2. SITTING: upright torso (<35 deg), thigh vertical distance |y_hip - y_knee| < 0.65 * torso_length
    engine_sit = SixClassPostureEngine(kp_conf_threshold=0.30)
    kps_sit = _blank_kps()
    _set_kp(kps_sit, KP_L_SHOULDER, 300, 170)
    _set_kp(kps_sit, KP_R_SHOULDER, 340, 170)
    _set_kp(kps_sit, KP_L_HIP, 305, 280)
    _set_kp(kps_sit, KP_R_HIP, 335, 280)
    # Knees horizontally forward, small vertical drop from hips (25 px < 0.65 * 110)
    _set_kp(kps_sit, KP_L_KNEE, 360, 305)
    _set_kp(kps_sit, KP_R_KNEE, 380, 305)
    _set_kp(kps_sit, KP_L_ANKLE, 360, 400)
    _set_kp(kps_sit, KP_R_ANKLE, 380, 400)
    res_sit = engine_sit.evaluate_person(
        track_id=2,
        bbox=(280.0, 140.0, 400.0, 410.0),
        det_conf=0.91,
        keypoints_xyc=kps_sit,
        timestamp=1.0,
        frame_height=480,
    )
    assert res_sit.posture == PostureClass.SITTING

    # 3. BENDING: torso inclined >= 45 deg (and >= 60 deg triggers warning), knees extended
    engine_bend = SixClassPostureEngine(kp_conf_threshold=0.30)
    kps_bend = _blank_kps()
    # Hips at (300, 260), Shoulders pitched forward to (400, 210) -> dx=100, dy=50 -> ~63.4 deg
    _set_kp(kps_bend, KP_L_SHOULDER, 400, 210)
    _set_kp(kps_bend, KP_R_SHOULDER, 400, 210)
    _set_kp(kps_bend, KP_L_HIP, 300, 260)
    _set_kp(kps_bend, KP_R_HIP, 300, 260)
    _set_kp(kps_bend, KP_L_KNEE, 300, 350)
    _set_kp(kps_bend, KP_R_KNEE, 300, 350)
    _set_kp(kps_bend, KP_L_ANKLE, 300, 440)
    _set_kp(kps_bend, KP_R_ANKLE, 300, 440)
    res_bend = engine_bend.evaluate_person(
        track_id=3,
        bbox=(270.0, 190.0, 430.0, 445.0),
        det_conf=0.93,
        keypoints_xyc=kps_bend,
        timestamp=1.0,
        frame_height=480,
    )
    assert res_bend.posture == PostureClass.BENDING
    assert res_bend.bending_warning is True
    assert "WARN" in res_bend.display_label

    # 4. LYING DOWN: horizontal aspect ratio > 1.15, shoulder & hip similar y, low to floor, low velocity
    engine_lie = SixClassPostureEngine(kp_conf_threshold=0.30)
    kps_lie = _blank_kps()
    _set_kp(kps_lie, KP_L_SHOULDER, 180, 390)
    _set_kp(kps_lie, KP_R_SHOULDER, 180, 395)
    _set_kp(kps_lie, KP_L_HIP, 310, 395)
    _set_kp(kps_lie, KP_R_HIP, 310, 400)
    _set_kp(kps_lie, KP_L_KNEE, 410, 400)
    _set_kp(kps_lie, KP_R_KNEE, 410, 405)
    _set_kp(kps_lie, KP_L_ANKLE, 490, 405)
    _set_kp(kps_lie, KP_R_ANKLE, 490, 410)
    res_lie = engine_lie.evaluate_person(
        track_id=4,
        bbox=(150.0, 350.0, 510.0, 450.0),
        det_conf=0.90,
        keypoints_xyc=kps_lie,
        timestamp=1.0,
        frame_height=480,
    )
    assert res_lie.posture == PostureClass.LYING_DOWN


def test_six_class_walking_and_falling():
    engine = SixClassPostureEngine(kp_conf_threshold=0.30)

    # 5. WALKING: upright torso + alternating ankle separation and lateral centroid movement over 5-10 frames
    for i in range(6):
        t = 1.0 + i * 0.1
        cx = 200.0 + i * 14.0
        ank_spread = 10.0 if (i % 2 == 0) else 42.0
        kps = _blank_kps()
        _set_kp(kps, KP_L_SHOULDER, cx - 15, 150)
        _set_kp(kps, KP_R_SHOULDER, cx + 15, 150)
        _set_kp(kps, KP_L_HIP, cx - 12, 250)
        _set_kp(kps, KP_R_HIP, cx + 12, 250)
        _set_kp(kps, KP_L_KNEE, cx - 10, 340)
        _set_kp(kps, KP_R_KNEE, cx + 10, 340)
        _set_kp(kps, KP_L_ANKLE, cx - ank_spread / 2, 430)
        _set_kp(kps, KP_R_ANKLE, cx + ank_spread / 2, 430)
        res_walk = engine.evaluate_person(
            track_id=10,
            bbox=(cx - 40, 110.0, cx + 40, 440.0),
            det_conf=0.92,
            keypoints_xyc=kps,
            timestamp=t,
            frame_height=480,
        )
    assert res_walk.posture == PostureClass.WALKING

    # 6. FALLING: sudden transition from upright to horizontal with high downward velocity (v_y > 1.5 m/s)
    fall_engine = SixClassPostureEngine(kp_conf_threshold=0.30)
    kps_up = _blank_kps()
    _set_kp(kps_up, KP_L_SHOULDER, 300, 140)
    _set_kp(kps_up, KP_R_SHOULDER, 340, 140)
    _set_kp(kps_up, KP_L_HIP, 305, 250)
    _set_kp(kps_up, KP_R_HIP, 335, 250)
    _set_kp(kps_up, KP_L_ANKLE, 320, 430)
    fall_engine.evaluate_person(
        track_id=99,
        bbox=(280.0, 100.0, 360.0, 440.0),
        det_conf=0.95,
        keypoints_xyc=kps_up,
        timestamp=1.0,
        frame_height=480,
    )

    kps_down = _blank_kps()
    _set_kp(kps_down, KP_L_SHOULDER, 220, 400)
    _set_kp(kps_down, KP_R_SHOULDER, 220, 405)
    _set_kp(kps_down, KP_L_HIP, 350, 405)
    _set_kp(kps_down, KP_R_HIP, 350, 410)
    res_fall = fall_engine.evaluate_person(
        track_id=99,
        bbox=(180.0, 360.0, 490.0, 455.0),
        det_conf=0.95,
        keypoints_xyc=kps_down,
        timestamp=1.25,
        frame_height=480,
        rgb_fall_confidence=0.82,
    )
    assert res_fall.posture == PostureClass.FALLING
    assert res_fall.vertical_velocity_mps > 1.5


def test_tri_mode_ptz_controller_and_osd(tmp_path):
    ptz = TriModePTZController(enabled=False, initial_mode=CameraMode.STATIC)
    try:
        assert ptz.mode == CameraMode.STATIC
        # STATIC mode blocks manual steering
        assert ptz.manual_steer("W") is False

        # Cycle to MANUAL
        assert ptz.cycle_mode() == CameraMode.MANUAL
        assert ptz.manual_steer("W") is True
        assert ptz.manual_steer("A") is True

        # Cycle to AUTOMATIC
        assert ptz.cycle_mode() == CameraMode.AUTOMATIC

        # Occupants centered inside 15% deadband -> no pan/tilt command
        telem_centered = ptz.update_automatic_tracking(
            bboxes=[(290.0, 200.0, 350.0, 280.0)],
            frame_width=640,
            frame_height=480,
            now=10.0,
        )
        assert telem_centered.in_deadband is True
        assert telem_centered.cmd_pan == 0.0
        assert telem_centered.cmd_tilt == 0.0

        # Multi-person group shifted to top-right -> e_x > 0.15, e_y < -0.15 -> pan > 0, tilt > 0
        telem_off = ptz.update_automatic_tracking(
            bboxes=[(450.0, 40.0, 520.0, 140.0), (500.0, 50.0, 610.0, 160.0)],
            frame_width=640,
            frame_height=480,
            now=11.0,
        )
        assert telem_off.group_bbox == (450, 40, 610, 160)
        assert telem_off.collective_center == (530, 100)
        assert telem_off.in_deadband is False
        assert 0.0 < telem_off.cmd_pan <= 0.4
        assert 0.0 < telem_off.cmd_tilt <= 0.3

        # Render OSD HUD with target reticle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        collector = DatasetCollector(root_dir=tmp_path / "dataset")
        rendered = render_osd_hud(
            frame=frame,
            persons=[],
            camera_mode=ptz.mode,
            auto_telemetry=telem_off,
            fps=30.0,
            ai_overlay_enabled=True,
            ptz_online=False,
            dataset_collector=collector,
        )
        assert rendered.shape == (480, 640, 3)

        # Cycle back to STATIC
        assert ptz.cycle_mode() == CameraMode.STATIC
    finally:
        ptz.shutdown()
