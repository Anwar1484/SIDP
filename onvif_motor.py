import time
from onvif import ONVIFCamera

# Connection parameters for local ONVIF service
CAMERA_IP = "172.20.10.4"
PORT = 2020
USER = "SIDPProject"
PASS = "SIDPUTP2026"

print(f"[INFO] Connecting to Tapo C210 via ONVIF at {CAMERA_IP}:{PORT}...")

# 1. Initialize ONVIF Camera
mycam = ONVIFCamera(CAMERA_IP, PORT, USER, PASS)

# 2. Bind the Media Service and get the active Profile Token
media_service = mycam.create_media_service()
profiles = media_service.GetProfiles()
media_profile = profiles[0]
token = media_profile.token
print(f"[SUCCESS] Obtained Media Profile Token: {token}")

# 3. Bind the PTZ (Pan/Tilt/Zoom) Control Service
ptz_service = mycam.create_ptz_service()

def move_camera(pan_speed, tilt_speed, duration=1.0):
    """
    Commands the Tapo motor to move at a specific velocity vector 
    and stops after the given duration.
    
    pan_speed:  -1.0 (full left) to +1.0 (full right)
    tilt_speed: -1.0 (full down) to +1.0 (full up)
    duration:   Time in seconds to move before stopping
    """
    # Create the ContinuousMove request object
    request = ptz_service.create_type('ContinuousMove')
    request.ProfileToken = token
    
    # Define speed vector
    request.Velocity = {
        'PanTilt': {
            'x': pan_speed,
            'y': tilt_speed
        }
    }
    
    # Issue move command
    ptz_service.ContinuousMove(request)
    time.sleep(duration)
    
    # Issue stop command to hold the new position
    ptz_service.Stop({'ProfileToken': token})
    time.sleep(0.5)

# ==========================================
# MOTOR ACTUATION TEST SEQUENCE
# ==========================================
if __name__ == "__main__":
    try:
        print("\n--- Starting Programmatic PTZ Sequence ---")
        
        # 1. Pan Right
        print("[ACTION] Panning Right for 1.2s...")
        move_camera(pan_speed=0.4, tilt_speed=0.0, duration=1.2)

        # 2. Pan Left (return to center)
        print("[ACTION] Panning Left for 1.2s...")
        move_camera(pan_speed=-0.4, tilt_speed=0.0, duration=1.2)

        # 3. Tilt Up
        print("[ACTION] Tilting Up for 0.8s...")
        move_camera(pan_speed=0.0, tilt_speed=0.3, duration=0.8)

        # 4. Tilt Down (return to level)
        print("[ACTION] Tilting Down for 0.8s...")
        move_camera(pan_speed=0.0, tilt_speed=-0.3, duration=0.8)

        print("\n[SUCCESS] ONVIF motor sequence completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] PTZ command failed: {e}")