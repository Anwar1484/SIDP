import time
from pytapo import Tapo

# Camera connection credentials
CAMERA_IP = "172.20.10.4"
USERNAME = "SIDPProject"
PASSWORD = "SIDPUTP2026"

print(f"[INFO] Connecting to Tapo C210 motor controller at {CAMERA_IP}...")

try:
    # Initialize the camera control interface
    camera = Tapo(CAMERA_IP, USERNAME, PASSWORD)
    
    # Query basic camera information to verify authentication
    device_info = camera.getBasicInfo()
    print("[SUCCESS] Connected to Tapo motor daemon!")
    print(f"Device Name: {device_info.get('device_info', {}).get('basic_info', {}).get('device_name', 'Tapo C210')}")

    print("\n--- Starting Motor Actuation Sequence ---")
    
    # 1. Pan Right by 20 degrees
    print("[ACTION] Panning Right (+20 degrees)...")
    camera.moveMotor(20, 0)
    time.sleep(2)

    # 2. Pan Left by 20 degrees (back to original position)
    print("[ACTION] Panning Left (-20 degrees)...")
    camera.moveMotor(-20, 0)
    time.sleep(2)

    # 3. Tilt Up by 10 degrees
    print("[ACTION] Tilting Up (+10 degrees)...")
    camera.moveMotor(0, 10)
    time.sleep(2)

    # 4. Tilt Down by 10 degrees (back to original level)
    print("[ACTION] Tilting Down (-10 degrees)...")
    camera.moveMotor(0, -10)
    time.sleep(2)

    print("\n[SUCCESS] Motor calibration complete! All axes operational.")

except Exception as e:
    print(f"\n[ERROR] Motor control communication failed: {e}")
    print("Troubleshooting: Ensure the camera IP is correct and no other app is actively steering the motors.")