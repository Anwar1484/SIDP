import cv2
import time
from threading import Thread
from onvif import ONVIFCamera

class CameraController:
    """
    Subsystem 1 Production Module:
    - Multi-threaded zero-lag RTSP frame acquisition (/stream2)
    - Local ONVIF PTZ motor steering over port 2020
    - Non-stop toggleable panning loop (Press Z to start / stop)
    - Responsive emergency stop and socket cleanup
    """
    def __init__(self, ip="172.20.10.4", user="SIDPProject", password="SIDPUTP2026"):
        self.ip = ip
        self.user = user
        self.password = password
        self.rtsp_url = f"rtsp://{self.user}:{self.password}@{self.ip}:554/stream2"
        
        # Operational State Flags
        self.stopped = False
        self.is_patrolling = False  # Toggle flag for non-stop sweeping

        # 1. Initialize RTSP Video Capture
        print(f"[S1] Initializing video capture from {self.ip}...")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ConnectionError(f"[ERROR] Failed to open RTSP stream at {self.rtsp_url}")
            
        self.ret, self.frame = self.cap.read()

        # 2. Initialize ONVIF PTZ Service
        print(f"[S1] Binding ONVIF motor daemon at {self.ip}:2020...")
        try:
            self.mycam = ONVIFCamera(self.ip, 2020, self.user, self.password)
            self.media = self.mycam.create_media_service()
            self.ptz = self.mycam.create_ptz_service()
            self.profile = self.media.GetProfiles()[0]
            self.token = self.profile.token
            self.ptz_available = True
            print("[S1] ONVIF PTZ service online and ready.")
        except Exception as e:
            print(f"[WARNING] PTZ service offline: {e}. Camera will operate in video-only mode.")
            self.ptz_available = False

        # 3. Start Background Ingestion Thread
        self.worker = Thread(target=self._update_frames, daemon=True)
        self.worker.start()
        print("[S1] Multi-threaded frame grabber active.")

    def _update_frames(self):
        """Worker loop continuously flushing stale frames to keep latency near 0ms."""
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Frame dropped. Maintaining stream...")
                time.sleep(0.01)
                continue
            self.frame = frame

    def get_frame(self):
        """Returns the newest decoded BGR frame for Subsystem 2 (AI)."""
        return self.frame

    def move(self, pan_speed=0.0, tilt_speed=0.0, duration=0.4):
        """Manual step movement for W/A/S/D keys."""
        if not self.ptz_available or self.is_patrolling:
            return

        def _actuate():
            try:
                request = self.ptz.create_type('ContinuousMove')
                request.ProfileToken = self.token
                request.Velocity = {'PanTilt': {'x': pan_speed, 'y': tilt_speed}}
                self.ptz.ContinuousMove(request)
                time.sleep(duration)
                self.ptz.Stop({'ProfileToken': self.token})
            except Exception as err:
                print(f"[ERROR] Motor command failed: {err}")

        Thread(target=_actuate, daemon=True).start()

    def pan_left(self):
        self.move(pan_speed=-0.4, tilt_speed=0.0)

    def pan_right(self):
        self.move(pan_speed=0.4, tilt_speed=0.0)

    def tilt_up(self):
        self.move(pan_speed=0.0, tilt_speed=0.3)

    def tilt_down(self):
        self.move(pan_speed=0.0, tilt_speed=-0.3)

    def toggle_continuous_pan(self):
        """
        Toggles continuous left-and-right panning on and off via 'Z'.
        """
        if not self.ptz_available:
            print("[WARNING] PTZ service offline; cannot toggle pan.")
            return

        if self.is_patrolling:
            # STOP COMMAND
            print("\n[CMD] Halting continuous pan...")
            self.is_patrolling = False
            try:
                self.ptz.Stop({'ProfileToken': self.token})
            except Exception as e:
                print(f"[ERROR] Failed to send stop command: {e}")
        else:
            # START COMMAND
            print("\n[CMD] Continuous pan ACTIVATED. Press 'Z' again to stop.")
            self.is_patrolling = True
            Thread(target=self._patrol_worker, daemon=True).start()

    def _patrol_worker(self):
        """Background worker executing continuous back-and-forth sweeps."""
        PAN_SPEED = 0.4
        SWEEP_DURATION = 1.0  # ~35 degrees of travel at speed 0.4
        direction = -1        # -1 = Left, +1 = Right

        while self.is_patrolling and not self.stopped:
            try:
                # 1. Start moving in current direction
                request = self.ptz.create_type('ContinuousMove')
                request.ProfileToken = self.token
                request.Velocity = {'PanTilt': {'x': direction * PAN_SPEED, 'y': 0.0}}
                self.ptz.ContinuousMove(request)

                # 2. Slice sweep time into 0.1s checks so 'Z' stops the motor immediately
                steps = int(SWEEP_DURATION / 0.1)
                for _ in range(steps):
                    if not self.is_patrolling or self.stopped:
                        break
                    time.sleep(0.1)

                # 3. Brake motor at edge of arc
                self.ptz.Stop({'ProfileToken': self.token})

                # 4. Brief dwell pause at each turn (0.4s)
                for _ in range(4):
                    if not self.is_patrolling or self.stopped:
                        break
                    time.sleep(0.1)

                # 5. Reverse sweep direction
                direction *= -1

            except Exception as e:
                print(f"[ERROR] Continuous pan error: {e}")
                break

        # Ensure motor completely brakes when loop terminates
        try:
            self.ptz.Stop({'ProfileToken': self.token})
        except Exception:
            pass

    def release(self):
        """Clean shutdown of threads and camera sockets."""
        self.is_patrolling = False
        self.stopped = True
        time.sleep(0.2)
        if self.ptz_available:
            try:
                self.ptz.Stop({'ProfileToken': self.token})
            except Exception:
                pass
        if self.cap.isOpened():
            self.cap.release()
        print("[S1] Resources freed cleanly.")


# ==========================================
# INTERACTIVE DEMO CONSOLE
# ==========================================
if __name__ == "__main__":
    controller = CameraController()
    
    print("\n==============================================")
    print("      SUBSYSTEM 1: LIVE INTERACTIVE CONSOLE   ")
    print("==============================================")
    print(" Controls (Click the video window first):")
    print("   [W / A / S / D]  : Manual Tilt / Pan")
    print("   [Z]              : Toggle Continuous Pan (ON / OFF)")
    print("   [Q]              : Quit Program")
    print("==============================================\n")

    while not controller.stopped:
        frame = controller.get_frame()
        if frame is None:
            continue

        # Dynamic On-Screen Display (OSD) HUD
        if controller.is_patrolling:
            status_text = "CONTINUOUS PANNING (Press Z to Stop)"
            status_color = (0, 165, 255)  # Orange
        else:
            status_text = "READY (Manual Control)"
            status_color = (0, 255, 0)    # Green

        cv2.putText(frame, "Subsystem 1: Live Stream & PTZ", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Mode: {status_text}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, "Keys: W/A/S/D (Manual) | Z (Toggle Sweep) | Q (Quit)", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("SIDP - Subsystem 1 Operational Feed", frame)
        
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('z') or key == ord('Z'):
            controller.toggle_continuous_pan()
        elif key == ord('a') or key == 81:
            controller.pan_left()
        elif key == ord('d') or key == 83:
            controller.pan_right()
        elif key == ord('w') or key == 82:
            controller.tilt_up()
        elif key == ord('s') or key == 84:
            controller.tilt_down()

    controller.release()
    cv2.destroyAllWindows()