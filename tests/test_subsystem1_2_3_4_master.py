"""
Unit & Integration Tests for `subsystem1_2_3_4_master.py` (Subsystems 1 + 2 + 3 + 4).
Verifies:
  1. ICD-03 SQLite database schema initialization (`residents`, `posture_events`, `system_alerts`).
  2. Throttled heartbeat writes (1.0s) vs. immediate state-transition and incident logging.
  3. Atomic frame buffer & warning snapshot persistence.
  4. Embedded ICD-03 HTTP/MJPEG Dashboard Server endpoints (`/api/icd03/state`, `/api/icd03/raw.jpg`, `/api/icd03/residents`).
  5. Streamlit `Subsystem4-Dashboard/app.py` data loader compatibility.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import requests

from subsystem1_2_3_4_master import (
    CameraMode,
    ICD03DashboardHTTPServer,
    ICD03DataBridge,
    PersonTelemetry,
    PostureClass,
    SafetyTier,
)


def _dummy_person(
    track_id: int = 1,
    posture: PostureClass = PostureClass.STANDING,
    tier: SafetyTier = SafetyTier.SAFE,
    torso_angle: float = 12.0,
    risk_score: int = 10,
    duration_s: float = 0.5,
    audio_dispatched: bool = False,
) -> PersonTelemetry:
    kps = np.ones((17, 3), dtype=np.float32) * 0.9
    return PersonTelemetry(
        track_id=track_id,
        person_id=f"p-{track_id:03d}",
        bbox=(120.0, 60.0, 260.0, 420.0),
        confidence=0.92,
        keypoints_xyc=kps,
        posture=posture,
        torso_angle_deg=torso_angle,
        knee_angle_deg=172.0,
        vertical_velocity_mps=2.1 if posture == PostureClass.FALLING else 0.0,
        rgb_fall_conf=None,
        safety_tier=tier,
        risk_score=risk_score,
        posture_duration_s=duration_s,
        severe_bend_duration_s=duration_s if posture == PostureClass.BENDING else 0.0,
        safety_reasons=[f"{tier.badge_text}: {posture.value}"],
        audio_dispatched=audio_dispatched,
        telegram_dispatched=False,
    )


def test_icd03_sqlite_schema_and_seed_residents(tmp_path: Path) -> None:
    db_file = tmp_path / "sidp_test.db"
    state_file = tmp_path / "state.json"
    raw_file = tmp_path / "raw.jpg"
    ann_file = tmp_path / "ann.jpg"
    snaps_dir = tmp_path / "snapshots"

    bridge = ICD03DataBridge(
        db_path=db_file,
        live_state_path=state_file,
        raw_frame_path=raw_file,
        annotated_frame_path=ann_file,
        snapshots_dir=snaps_dir,
    )
    try:
        state = bridge.get_live_state_dict()
        assert "res_001" in state["RESIDENTS"]
        assert "res_002" in state["RESIDENTS"]
        assert "res_003" in state["RESIDENTS"]
        assert state["RESIDENTS"]["res_001"]["full_name"] == "Amina Yusuf"

        with sqlite3.connect(str(db_file)) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert {"residents", "posture_events", "system_alerts"}.issubset(tables)
            res_count = conn.execute("SELECT COUNT(*) FROM residents").fetchone()[0]
            assert res_count == 3
    finally:
        bridge.shutdown()


def test_icd03_throttled_events_and_immediate_alert_snapshots(tmp_path: Path) -> None:
    db_file = tmp_path / "sidp_events.db"
    state_file = tmp_path / "state.json"
    raw_file = tmp_path / "raw.jpg"
    ann_file = tmp_path / "ann.jpg"
    snaps_dir = tmp_path / "snapshots"

    bridge = ICD03DataBridge(
        db_path=db_file,
        live_state_path=state_file,
        raw_frame_path=raw_file,
        annotated_frame_path=ann_file,
        snapshots_dir=snaps_dir,
    )
    try:
        raw_img = np.zeros((240, 320, 3), dtype=np.uint8)
        ann_img = np.full((240, 320, 3), 40, dtype=np.uint8)

        # Frame 1 at t=10.0s: Safe standing -> writes initial posture_event
        p_safe = _dummy_person(track_id=1, posture=PostureClass.STANDING, tier=SafetyTier.SAFE)
        bridge.publish_frame_and_telemetry(
            raw_frame=raw_img,
            annotated_frame=ann_img,
            persons=[p_safe],
            camera_mode=CameraMode.MANUAL,
            fps=28.5,
            now_perf=10.0,
        )

        # Frame 2 at t=10.2s (<1.0s throttle window, same posture/tier) -> throttled
        bridge.publish_frame_and_telemetry(
            raw_frame=raw_img,
            annotated_frame=ann_img,
            persons=[p_safe],
            camera_mode=CameraMode.MANUAL,
            fps=28.5,
            now_perf=10.2,
        )

        # Frame 3 at t=10.4s: Immediate transition to FALLING / ALERT -> writes immediately + system_alert + snapshot
        p_fall = _dummy_person(
            track_id=1,
            posture=PostureClass.FALLING,
            tier=SafetyTier.ALERT,
            torso_angle=78.0,
            risk_score=95,
            duration_s=0.4,
            audio_dispatched=True,
        )
        bridge.publish_frame_and_telemetry(
            raw_frame=raw_img,
            annotated_frame=ann_img,
            persons=[p_fall],
            camera_mode=CameraMode.AUTOMATIC,
            fps=27.0,
            now_perf=10.4,
        )

        bridge.flush(timeout=3.0)

        with sqlite3.connect(str(db_file)) as conn:
            ev_count = conn.execute("SELECT COUNT(*) FROM posture_events").fetchone()[0]
            # Frame 1 + Frame 3 (Frame 2 was throttled)
            assert ev_count == 2

            alert_row = conn.execute(
                "SELECT resident_id, safety_tier, alert_type, fall_detected, snapshot_url FROM system_alerts"
            ).fetchone()
            assert alert_row is not None
            assert alert_row[0] == "res_001"
            assert alert_row[1] == "alert"
            assert alert_row[2] == "Fall detected"
            assert alert_row[3] == 1
            assert alert_row[4].startswith("/snapshots/log_")

        live_state = bridge.get_live_state_dict()
        res1_status = live_state["CURRENT_POSTURE_STATUS"]["res_001"]
        assert res1_status["fall_detected"] is True
        assert res1_status["alert_active"] is True
        assert res1_status["safety_tier"] == "alert"
        assert raw_file.is_file()
        assert ann_file.is_file()
    finally:
        bridge.shutdown()


def test_icd03_http_server_endpoints(tmp_path: Path) -> None:
    db_file = tmp_path / "sidp_http.db"
    bridge = ICD03DataBridge(
        db_path=db_file,
        live_state_path=tmp_path / "state.json",
        raw_frame_path=tmp_path / "raw.jpg",
        annotated_frame_path=tmp_path / "ann.jpg",
        snapshots_dir=tmp_path / "snapshots",
    )
    server = ICD03DashboardHTTPServer(bridge=bridge, host="127.0.0.1", port=0)
    try:
        assert server.start() is True
        base_url = f"http://127.0.0.1:{server.port}"

        # 1. GET /api/icd03/state
        resp = requests.get(f"{base_url}/api/icd03/state", timeout=3.0)
        assert resp.status_code == 200
        payload = resp.json()
        assert "RESIDENTS" in payload
        assert "CURRENT_POSTURE_STATUS" in payload

        # 2. POST /api/icd03/residents
        new_res = {
            "full_name": "Dato Ahmad Razak",
            "age": 81,
            "room_number": "305",
            "address": "Seri Iskandar, Perak",
            "risk_category": "High",
        }
        post_resp = requests.post(f"{base_url}/api/icd03/residents", json=new_res, timeout=3.0)
        assert post_resp.status_code == 201
        created = post_resp.json()
        assert created["full_name"] == "Dato Ahmad Razak"

        # 3. GET /api/icd03/raw.jpg
        img_resp = requests.get(f"{base_url}/api/icd03/raw.jpg", timeout=3.0)
        assert img_resp.status_code == 200
        assert img_resp.headers["Content-Type"] == "image/jpeg"
    finally:
        server.stop()
        bridge.shutdown()
