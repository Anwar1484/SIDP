@echo off
title SIDP Full 4-Subsystem Master Pipeline (Subsystems 1 + 2 + 3 + 4)
cd /d "%~dp0"

echo ================================================================================
echo RBB4013 System Integrated Design Project (SIDP) - Universiti Teknologi PETRONAS
echo Launching Full End-to-End Pipeline:
echo   [Subsystem 1] Tapo C210 RTSP Stream + Tri-Mode ONVIF PTZ + Dataset Capture
echo   [Subsystem 2] YOLO26m-Pose (RTX 4050 FP16) + 6-Class Posture & Fall Engine
echo   [Subsystem 3] 4-Tier Safety State Machine + 3s Timer + Voice/Telegram Alerts
echo   [Subsystem 4] ICD-03 SQLite Database + Caregiver Web Dashboards:
echo                 * Streamlit Analytics Portal : http://localhost:8501
echo                 * SB Admin 2 Live Web Portal : http://localhost:8502/dashboard.html?live=1
echo ================================================================================
echo.

".\.venv\Scripts\python.exe" "subsystem1_2_3_4_master.py" --open-browser %*
