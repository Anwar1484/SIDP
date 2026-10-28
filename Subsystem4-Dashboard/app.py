#!/usr/bin/env python3
"""
Subsystem4-Dashboard/app.py
================================================================================
RBB4013 System Integrated Design Project (SIDP) - Universiti Teknologi PETRONAS
Subsystem 4: Caregiver Web Dashboard & SQLite Analytics Portal (Streamlit)

Reads real-time telemetry and incident logs written via the ICD-03 Data Bridge
from `subsystem1_2_3_4_master.py`:
  - SQLite Database: `outputs/sidp_subsystem4.db`
    (`residents`, `posture_events`, `system_alerts`)
  - Shared Live Frame Buffers:
    `outputs/live_raw.jpg` (Original unmasked Tapo C210 stream)
    `outputs/live_annotated.jpg` (YOLO26-Pose 17-keypoint + Subsystem 3 HUD)
  - Live ICD-03 JSON State: `outputs/icd03_live_state.json`
================================================================================
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
DB_PATH = OUTPUTS_DIR / "sidp_subsystem4.db"
LIVE_RAW_JPG = OUTPUTS_DIR / "live_raw.jpg"
LIVE_ANNOTATED_JPG = OUTPUTS_DIR / "live_annotated.jpg"
LIVE_STATE_JSON = OUTPUTS_DIR / "icd03_live_state.json"


def _connect_readonly_or_rw(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema_exists(db_path: Path = DB_PATH) -> None:
    with _connect_readonly_or_rw(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS residents (
                resident_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                age INTEGER NOT NULL,
                room_number TEXT NOT NULL,
                address TEXT NOT NULL,
                risk_category TEXT NOT NULL,
                photo_url TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS posture_events (
                event_id TEXT PRIMARY KEY,
                resident_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                posture_class TEXT NOT NULL,
                activity_recognition TEXT NOT NULL,
                pose_status TEXT NOT NULL,
                torso_angle_deg REAL,
                knee_angle_deg REAL,
                safety_tier TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                duration_s REAL NOT NULL,
                fall_detected INTEGER NOT NULL DEFAULT 0,
                inactivity_detected INTEGER NOT NULL DEFAULT 0,
                camera_mode TEXT DEFAULT 'MANUAL',
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS system_alerts (
                alert_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                resident_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                safety_tier TEXT NOT NULL,
                posture_class TEXT NOT NULL,
                activity_recognition TEXT NOT NULL,
                pose_status TEXT NOT NULL,
                torso_angle_deg REAL,
                risk_score INTEGER NOT NULL,
                duration_s REAL NOT NULL,
                fall_detected INTEGER NOT NULL DEFAULT 0,
                inactivity_detected INTEGER NOT NULL DEFAULT 0,
                alert_active INTEGER NOT NULL DEFAULT 1,
                notes TEXT DEFAULT '',
                snapshot_url TEXT DEFAULT '',
                audio_dispatched INTEGER NOT NULL DEFAULT 0,
                telegram_dispatched INTEGER NOT NULL DEFAULT 0
            );
            """
        )


def load_dashboard_data(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ensure_schema_exists(db_path)
    with _connect_readonly_or_rw(db_path) as conn:
        residents_df = pd.read_sql_query("SELECT * FROM residents ORDER BY resident_id ASC", conn)
        events_df = pd.read_sql_query(
            "SELECT * FROM posture_events ORDER BY timestamp DESC LIMIT 200",
            conn,
        )
        alerts_df = pd.read_sql_query(
            "SELECT * FROM system_alerts ORDER BY timestamp DESC LIMIT 100",
            conn,
        )

    live_state: dict[str, Any] = {}
    if LIVE_STATE_JSON.is_file():
        try:
            live_state = json.loads(LIVE_STATE_JSON.read_text(encoding="utf-8"))
        except Exception:
            live_state = {}

    return residents_df, events_df, alerts_df, live_state


def _read_image_bytes(path: Path) -> bytes | None:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
        return data if len(data) > 128 else None
    except Exception:
        return None


def main() -> None:
    st.set_page_config(
        page_title="SIDP Subsystem 4 — Caregiver Posture Monitoring Dashboard",
        page_icon="🏥",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .kpi-card {
            background: #ffffff;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 5px solid #4e73df;
        }
        .kpi-title {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #2e3440;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("🏥 SIDP Subsystem 4")
        st.caption("RBB4013 Elderly Posture & Safety Monitoring")
        auto_refresh = st.toggle("Live Auto-Refresh (1.0s)", value=True)
        refresh_sec = st.slider("Refresh Interval (s)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
        st.divider()
        st.markdown("**ICD-03 Data Sources**")
        st.code(f"SQLite: outputs/{DB_PATH.name}\nRaw:    outputs/{LIVE_RAW_JPG.name}\nPose:   outputs/{LIVE_ANNOTATED_JPG.name}")
        st.markdown(
            "🔗 **[Open Native SB Admin 2 Portal (:8502)](http://localhost:8502/dashboard.html?live=1)**"
        )

    residents_df, events_df, alerts_df, live_state = load_dashboard_data()
    current_status_map: dict[str, dict[str, Any]] = live_state.get("CURRENT_POSTURE_STATUS", {})
    pipeline_meta: dict[str, Any] = live_state.get("PIPELINE_META", {})

    total_residents = len(residents_df)
    active_occupants = int(pipeline_meta.get("active_occupants", 0))
    safe_count = 0
    unsafe_count = 0
    alert_count = 0

    for _, res_row in residents_df.iterrows():
        rid = str(res_row["resident_id"])
        st_rec = current_status_map.get(rid, {})
        tier = str(st_rec.get("safety_tier", "")).lower()
        posture_str = str(st_rec.get("posture", "")).lower()
        if tier == "safe" or "safe" in posture_str:
            safe_count += 1
        elif tier in ("warning", "unsafe", "alert") or "unsafe" in posture_str:
            unsafe_count += 1
        if tier == "alert" or st_rec.get("alert_active") is True or st_rec.get("fall_detected") is True:
            alert_count += 1

    st.subheader("Elderly Caregiver Safety & Posture Monitoring Dashboard")
    meta_line = (
        f"**ICD-03 Pipeline Status:** `{'ONLINE' if pipeline_meta else 'STANDBY'}`  |  "
        f"**Camera Mode:** `{pipeline_meta.get('camera_mode', 'MANUAL')}`  |  "
        f"**FPS:** `{pipeline_meta.get('fps', 0.0):.1f}`  |  "
        f"**Active Camera Occupants:** `{active_occupants}`  |  "
        f"**Total Registered Residents:** `{total_residents}`"
    )
    st.markdown(meta_line)

    # Top 4 KPI Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:#4e73df;">'
            f'<div class="kpi-title" style="color:#4e73df;">Active Occupants / Residents</div>'
            f'<div class="kpi-value">{active_occupants} / {total_residents}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:#1cc88a;">'
            f'<div class="kpi-title" style="color:#1cc88a;">Safe / Normal Posture</div>'
            f'<div class="kpi-value">{safe_count}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:#f6c23e;">'
            f'<div class="kpi-title" style="color:#d49b19;">Warning / Unsafe Posture</div>'
            f'<div class="kpi-value">{unsafe_count}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:#e74a3b;">'
            f'<div class="kpi-title" style="color:#e74a3b;">Fall &amp; Active Alerts</div>'
            f'<div class="kpi-value">{alert_count}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    tab_live, tab_logs, tab_sbadmin = st.tabs(
        [
            "📹 Live Camera & Resident Telemetry",
            "📋 Chronological Incident & SQLite Logs",
            "🌐 Native SB Admin 2 Dashboard View",
        ]
    )

    with tab_live:
        col_cam, col_res = st.columns([2.1, 1.0])
        with col_cam:
            st.markdown("#### Live Camera Comparison (Subsystem 1 RTSP vs. Subsystem 2+3 Skeleton HUD)")
            v_left, v_right = st.columns(2)
            raw_bytes = _read_image_bytes(LIVE_RAW_JPG)
            ann_bytes = _read_image_bytes(LIVE_ANNOTATED_JPG)

            with v_left:
                st.caption("👁️ **Original · Unmasked (Subsystem 1)**")
                if raw_bytes:
                    st.image(raw_bytes, use_container_width=True)
                else:
                    st.info("Waiting for live frame from `subsystem1_2_3_4_master.py` (`outputs/live_raw.jpg`)...")

            with v_right:
                st.caption("🦴 **Skeleton & Safety HUD (Subsystems 2 & 3)**")
                if ann_bytes:
                    st.image(ann_bytes, use_container_width=True)
                else:
                    st.info("Waiting for annotated skeleton frame (`outputs/live_annotated.jpg`)...")

        with col_res:
            st.markdown("#### Selected Resident Telemetry")
            if not residents_df.empty:
                options = [
                    f"{row['resident_id']} — {row['full_name']} (Room {row['room_number']})"
                    for _, row in residents_df.iterrows()
                ]
                chosen = st.selectbox("Select Resident", options, index=0)
                chosen_id = chosen.split(" — ")[0]
                profile = residents_df[residents_df["resident_id"] == chosen_id].iloc[0]
                status = current_status_map.get(chosen_id, {})

                tier_badge = str(status.get("safety_tier", "safe")).upper()
                posture_lbl = status.get("pose_status") or status.get("posture") or "No data"
                activity_lbl = status.get("activity_recognition", "—")
                angle_val = status.get("angle_degrees", "—")
                risk_val = status.get("risk_score", 0)
                fall_flag = "🚨 DETECTED" if status.get("fall_detected") else "✅ Not detected"
                inact_flag = "⚠️ DETECTED" if status.get("inactivity_detected") else "✅ Not detected"

                st.markdown(f"**Resident:** {profile['full_name']} ({profile['age']} yrs · Room {profile['room_number']})")
                st.markdown(f"**Baseline Risk:** `{profile['risk_category']}`")
                st.markdown(f"**Safety Tier (S3):** `{tier_badge}` (Risk Score: `{risk_val}/100`)")
                st.markdown(f"**Current Posture:** `{posture_lbl}`")
                st.markdown(f"**Detected Activity:** `{activity_lbl}`")
                st.markdown(f"**Torso Inclination Angle:** `{angle_val}°`")
                st.markdown(f"**Fall Detection:** {fall_flag}")
                st.markdown(f"**Inactivity Detection:** {inact_flag}")
                st.caption(f"Last updated: {status.get('updated_at', '—')}")
            else:
                st.warning("No residents registered in SQLite database yet.")

        st.divider()
        st.markdown("#### Resident Profiles (`residents` SQLite Table)")
        if not residents_df.empty:
            enriched_rows = []
            for _, r in residents_df.iterrows():
                rid = str(r["resident_id"])
                st_rec = current_status_map.get(rid, {})
                enriched_rows.append(
                    {
                        "Resident ID": rid,
                        "Full Name": r["full_name"],
                        "Age": r["age"],
                        "Room": r["room_number"],
                        "Baseline Risk": r["risk_category"],
                        "Live Posture": st_rec.get("pose_status", st_rec.get("posture", "Standby")),
                        "Safety Tier": str(st_rec.get("safety_tier", "safe")).upper(),
                        "Torso Angle (°)": st_rec.get("angle_degrees", 0.0),
                        "Risk Score": st_rec.get("risk_score", 0),
                    }
                )
            st.dataframe(pd.DataFrame(enriched_rows), use_container_width=True, hide_index=True)

    with tab_logs:
        st.markdown("#### Chronological System Alerts (`system_alerts` SQLite Table)")
        if not alerts_df.empty:
            st.dataframe(
                alerts_df[
                    [
                        "timestamp",
                        "resident_id",
                        "track_id",
                        "safety_tier",
                        "alert_type",
                        "severity",
                        "posture_class",
                        "torso_angle_deg",
                        "risk_score",
                        "duration_s",
                        "notes",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            csv_bytes = alerts_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download System Alerts CSV",
                data=csv_bytes,
                file_name=f"sidp_system_alerts_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No Warning/Unsafe/Alert incidents recorded yet.")

        st.markdown("#### Recent Posture Telemetry Stream (`posture_events` SQLite Table)")
        if not events_df.empty:
            st.dataframe(
                events_df[
                    [
                        "timestamp",
                        "resident_id",
                        "track_id",
                        "posture_class",
                        "safety_tier",
                        "torso_angle_deg",
                        "risk_score",
                        "duration_s",
                        "camera_mode",
                    ]
                ].head(50),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No posture telemetry events recorded yet.")

    with tab_sbadmin:
        st.markdown(
            "Embedded **Subsystem 4 SB Admin 2 Caregiver Portal** served live by `subsystem1_2_3_4_master.py` at "
            "[http://localhost:8502/dashboard.html?live=1](http://localhost:8502/dashboard.html?live=1):"
        )
        components.iframe("http://localhost:8502/dashboard.html?live=1", height=920, scrolling=True)

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
