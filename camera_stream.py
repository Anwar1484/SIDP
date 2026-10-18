import cv2
from threading import Thread
import time

class TapoStream:
    """
    Multi-threaded RTSP frame grabber for Tapo C210.
    Continuously flushes the internal OpenCV buffer on a background thread
    so the main application always reads the newest frame with zero lag.
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Verify connection
        if not self.cap.isOpened():
            raise ConnectionError(f"[ERROR] Could not open RTSP stream at: {self.rtsp_url}")
            
        # Read the first initial frame to set up dimensions
        self.ret, self.frame = self.cap.read()
        
        # Thread control flags
        self.stopped = False

    def start(self):
        """Spawns the background worker thread to constantly pull frames."""
        worker = Thread(target=self._update, daemon=True)
        worker.start()
        return self

    def _update(self):
        """Continuous background loop: grabs and overwrites the newest frame."""
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
                
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Frame drop or camera disconnected.")
                self.stopped = True
                break
            
            # Overwrite with the latest frame
            self.frame = frame

    def read(self):
        """Returns the single freshest frame stored in memory."""
        return self.frame

    def stop(self):
        """Signals background thread to terminate and cleans up resources."""
        self.stopped = True
        time.sleep(0.1)
        if self.cap.isOpened():
            self.cap.release()
        print("[INFO] Camera stream stopped and socket released.")


# ==========================================
# TEST HARNESS & SIMULATED AI DELAY
# ==========================================
if __name__ == "__main__":
    RTSP_URL = "rtsp://SIDPProject:SIDPUTP2026@172.20.10.4:554/stream2"
    
    print("[INFO] Starting threaded RTSP reader...")
    stream = TapoStream(RTSP_URL).start()
    
    # Allow 1 second for the camera to settle
    time.sleep(1.0)
    print("[SUCCESS] Threaded ingestion active! Press 'q' to stop.")

    prev_time = time.time()
    
    while not stream.stopped:
        frame = stream.read()
        if frame is None:
            continue

        # -----------------------------------------------------------------
        # VIVA SIMULATION:
        # We simulate a 50ms delay (as if YOLOv8-Pose is running inference)
        # -----------------------------------------------------------------
        time.sleep(0.05) 

        # Calculate and display current operational FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw FPS counter on the live video frame
        cv2.putText(
            frame, 
            f"Threaded Mode | FPS: {fps:.1f}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )

        cv2.imshow("Subsystem 1: Zero-Lag Ingestion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()