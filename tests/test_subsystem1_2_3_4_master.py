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
        captures_dir=tmp_path / "captures",
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
        captures_dir=tmp_path / "captures",
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

        # Verify date-partitioned incident snapshot archiving (`captures/YYYY-MM-DD/alert/HH-MM-SS_resident_1.jpg`)
        archived_list = bridge.list_archived_captures()
        assert len(archived_list) == 1
        assert archived_list[0]["tier"] == "alert"
        assert archived_list[0]["filename"].endswith("_resident_1.jpg")
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
        captures_dir=tmp_path / "captures",
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

        # 4. Trigger an incident snapshot and fetch via /api/icd03/captures and /captures/...
        dummy_frame = np.full((240, 320, 3), 90, dtype=np.uint8)
        p_warn = _dummy_person(track_id=2, posture=PostureClass.BENDING, tier=SafetyTier.WARNING, torso_angle=52.0)
        bridge.publish_frame_and_telemetry(
            raw_frame=dummy_frame,
            annotated_frame=dummy_frame,
            persons=[p_warn],
            camera_mode=CameraMode.MANUAL,
            fps=25.0,
            now_perf=30.0,
        )
        bridge.flush()

        cap_resp = requests.get(f"{base_url}/api/icd03/captures", timeout=3.0)
        assert cap_resp.status_code == 200
        caps = cap_resp.json()["captures"]
        assert len(caps) == 1
        assert caps[0]["tier"] == "warning"
        assert caps[0]["filename"].endswith("_resident_2.jpg")

        cap_img_resp = requests.get(f"{base_url}{caps[0]['url']}", timeout=3.0)
        assert cap_img_resp.status_code == 200
        assert cap_img_resp.headers["Content-Type"] == "image/jpeg"
    finally:
        server.stop()
        bridge.shutdown()


def test_archiver_strict_cooldown_and_on_saved_callback(tmp_path: Path) -> None:
    from subsystem1_2_3_4_master import IncidentSnapshotArchiver

    saved: list[dict] = []
    archiver = IncidentSnapshotArchiver(captures_root=tmp_path / "captures", cooldown_seconds=15.0, on_saved=saved.append)
    try:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        warn = _dummy_person(track_id=3, posture=PostureClass.BENDING, tier=SafetyTier.WARNING, torso_angle=52.0)
        unsafe = _dummy_person(track_id=3, posture=PostureClass.BENDING, tier=SafetyTier.UNSAFE, torso_angle=66.0)

        assert archiver.maybe_archive(warn, frame, now_perf=100.0) is not None   # first entry -> immediate
        assert archiver.maybe_archive(unsafe, frame, now_perf=101.0) is not None  # new (resident, state)
        assert archiver.maybe_archive(warn, frame, now_perf=102.0) is None        # flicker back inside 15 s -> suppressed
        assert archiver.maybe_archive(warn, frame, now_perf=116.0) is not None    # cooldown elapsed
        archiver.flush()

        assert len(saved) == 3
        assert saved[0]["tier"] == "warning" and saved[1]["tier"] == "unsafe"
        assert saved[0]["resident_id"] == 3 and saved[0]["torso_angle"] == 52.0
        for meta in saved:
            assert Path(meta["snapshot_path"]).is_file()
    finally:
        archiver.shutdown()


def test_telemetry_dispatcher_telegram_photo_and_firebase_payloads(tmp_path: Path, monkeypatch) -> None:
    import subsystem1_2_3_4_master as m

    calls: list[dict] = []

    class _Resp:
        status_code = 200
        text = "{}"

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _Resp()

    monkeypatch.setattr(m.requests, "post", fake_post)

    jpg = tmp_path / "10-11-12_resident_1.jpg"
    ok, buf = m.cv2.imencode(".jpg", np.zeros((60, 80, 3), dtype=np.uint8))
    assert ok
    jpg.write_bytes(buf.tobytes())

    disp = m.TelemetryDispatcher(
        telegram_token="123:TESTTOKEN",
        telegram_chat_id="-100200",
        firebase_url="https://example-rtdb.firebasedatabase.app/",
        firebase_auth="",
        resident_lookup=lambda rid: {"full_name": "Tok Mat", "room_number": "101"},
    )
    try:
        meta = {
            "timestamp": "2026-11-11 10:11:12",
            "resident_id": 1,
            "tier": "unsafe",
            "posture": "Bending",
            "torso_angle": 64.5,
            "snapshot_path": str(jpg),
        }
        result = disp.dispatch_now(meta)
        assert result["telegram"] == 200
        photo_call = calls[0]
        assert photo_call["url"].endswith("/bot123:TESTTOKEN/sendPhoto")
        assert photo_call["data"]["chat_id"] == "-100200"
        assert "[SIDP SAFETY ALERT]" in photo_call["data"]["caption"]
        assert "Severity: UNSAFE" in photo_call["data"]["caption"]
        assert "photo" in photo_call["files"]

        incidents = next(c for c in calls if c["url"].endswith("/incidents.json"))
        assert incidents["url"] == "https://example-rtdb.firebasedatabase.app/incidents.json"
        assert incidents["json"] == {
            "timestamp": "2026-11-11 10:11:12",
            "resident_id": 1,
            "resident_name": "Tok Mat",
            "room_number": "101",
            "posture": "Bending",
            "torso_angle": 64.5,
            "severity": "Unsafe",
            "snapshot_path": str(jpg),
        }
        logs = next(c for c in calls if c["url"].endswith("/ACTIVITY_LOGS.json"))
        assert logs["json"]["resident_id"] == "res_001" and logs["json"]["angle_degrees"] == 64.5

        # Warning tier: Firebase only (Telegram skipped unless explicitly enabled)
        calls.clear()
        warn_result = disp.dispatch_now({**meta, "tier": "warning"})
        assert "telegram" not in warn_result
        assert all("telegram.org" not in c["url"] for c in calls)

        # Missing image -> falls back to sendMessage
        calls.clear()
        disp.dispatch_now({**meta, "snapshot_path": str(tmp_path / "missing.jpg")})
        assert calls[0]["url"].endswith("/sendMessage")
    finally:
        disp.shutdown(drain_timeout=0.5)


def test_master_1234_s2_s3_payload_interface(tmp_path: Path) -> None:
    from subsystem1_2_3_4_master import Subsystem3SafetyEngine

    engine = Subsystem3SafetyEngine(
        camera_id="tapo_c210_1234",
        zone_name="bedroom",
        audio_enabled=False,
        telegram_enabled=False,
        captures_root=tmp_path / "captures",
    )
    try:
        kps = [[200.0, 150.0, 0.9] for _ in range(17)]
        kps[5] = [200.0, 150.0, 0.95]
        kps[6] = [200.0, 150.0, 0.95]
        kps[11] = [200.0, 280.0, 0.95]
        kps[12] = [200.0, 280.0, 0.95]

        payload = {
            "p-001": {
                "bbox": [150.0, 100.0, 250.0, 420.0],
                "keypoints": kps,
                "posture": "standing",
            }
        }
        res = engine.evaluate_payload(payload, now_perf=50.0)
        assert len(res) == 1
        assert res[0].person_id == "p-001"
        assert res[0].safety_tier == SafetyTier.SAFE
        assert res[0].to_s2_payload_entry()["posture"] == "standing"
    finally:
        engine.shutdown()


def test_whole_body_visibility_gate_and_4class_ai_posture_engine(tmp_path: Path) -> None:
    from subsystem1_2_3_4_master import (
        SixClassPostureEngine,
        Subsystem3SafetyEngine,
        is_whole_body_visible,
    )

    # 1. Whole body visible (shoulders 5,6 + hips 11,12 + knees 13,14 + ankles 15,16 inside [0.02, 0.98])
    full_xyn = np.full((17, 2), 0.5, dtype=np.float32)
    full_xyn[0] = [0.50, 0.15]   # nose
    full_xyn[5] = [0.45, 0.28]   # left shoulder
    full_xyn[6] = [0.55, 0.28]   # right shoulder
    full_xyn[11] = [0.46, 0.52]  # left hip
    full_xyn[12] = [0.54, 0.52]  # right hip
    full_xyn[13] = [0.46, 0.72]  # left knee
    full_xyn[14] = [0.54, 0.72]  # right knee
    full_xyn[15] = [0.46, 0.90]  # left ankle
    full_xyn[16] = [0.54, 0.90]  # right ankle
    full_conf = np.full(17, 0.92, dtype=np.float32)

    assert is_whole_body_visible(full_xyn, full_conf, conf_thresh=0.45) is True

    # 2. Cut-off ankles (outside frame or (0, 0)) -> False
    cutoff_ankles = full_xyn.copy()
    cutoff_ankles[15] = [0.0, 0.0]
    cutoff_ankles[16] = [0.50, 0.995]  # >= 0.98 boundary
    assert is_whole_body_visible(cutoff_ankles, full_conf, conf_thresh=0.45) is False

    # 3. Low confidence on hips (< 0.45) -> False
    low_conf = full_conf.copy()
    low_conf[11] = 0.20
    assert is_whole_body_visible(full_xyn, low_conf, conf_thresh=0.45) is False

    # 4. SixClassPostureEngine + Subsystem3SafetyEngine integration
    posture_engine = SixClassPostureEngine()
    assert posture_engine.ai_classifier is not None
    assert set(posture_engine.ai_classifier.classes_) == {
        "standing",
        "sitting",
        "bending",
        "lying_down",
    }

    s3_engine = Subsystem3SafetyEngine(
        camera_id="tapo_c210_gate",
        zone_name="living room",
        audio_enabled=False,
        telegram_enabled=False,
        captures_root=tmp_path / "captures",
    )
    try:
        # Partial body (missing legs/ankles) -> INSUFFICIENT_EVIDENCE and SAFE tier
        partial_xyc = np.zeros((17, 3), dtype=np.float32)
        partial_xyc[5] = [280.0, 140.0, 0.95]
        partial_xyc[6] = [340.0, 140.0, 0.95]
        partial_xyc[11] = [290.0, 260.0, 0.95]
        partial_xyc[12] = [330.0, 260.0, 0.95]
        # Knees and ankles at (0,0,0) -> occluded
        t_partial = posture_engine.evaluate_person(
            track_id=1,
            bbox=(250.0, 100.0, 370.0, 300.0),
            det_conf=0.90,
            keypoints_xyc=partial_xyc,
            timestamp=10.0,
            frame_height=480,
            frame_width=640,
        )
        assert t_partial.whole_body_visible is False
        assert t_partial.posture == PostureClass.INSUFFICIENT_EVIDENCE

        s3_partial = s3_engine.evaluate(t_partial, now_perf=10.0)
        assert s3_partial.safety_tier == SafetyTier.SAFE
        assert s3_partial.risk_score == 0

        # Full body visible -> AI classifier predicts one of the 4 trained postures
        full_xyc = np.column_stack(
            (full_xyn[:, 0] * 640.0, full_xyn[:, 1] * 480.0, full_conf)
        ).astype(np.float32)
        t_full = posture_engine.evaluate_person(
            track_id=1,
            bbox=(240.0, 60.0, 380.0, 440.0),
            det_conf=0.95,
            keypoints_xyc=full_xyc,
            timestamp=11.0,
            frame_height=480,
            frame_width=640,
        )
        assert t_full.whole_body_visible is True
        assert t_full.posture in {
            PostureClass.STANDING,
            PostureClass.SITTING,
            PostureClass.BENDING,
            PostureClass.LYING_DOWN,
        }
        assert t_full.ai_confidence is not None and t_full.ai_confidence > 0.0
    finally:
        s3_engine.shutdown()



