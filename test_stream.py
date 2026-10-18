import cv2

# Tapo C210 Stream 2 endpoint (Optimized resolution for low latency)
RTSP_URL = "rtsp://SIDPProject:SIDPUTP2026@172.20.10.4:554/stream2"

print("[INFO] Attempting to connect to Tapo C210...")

# Initialize OpenCV video capture using FFmpeg backend
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# Verify if connection succeeded
if not cap.isOpened():
    print("[ERROR] Could not connect to camera. Check IP, network, or credentials.")
    exit()

print("[SUCCESS] Stream active! Rendering live feed. Press 'q' on the window to exit.")

while True:
    # Read the next frame from the camera buffer
    ret, frame = cap.read()

    # If ret is False, the frame was not grabbed or connection dropped
    if not ret:
        print("[WARNING] Frame drop or stream disconnected.")
        break

    # Display the current frame
    cv2.imshow("Subsystem 1: Tapo C210 Live Feed", frame)

    # Poll keyboard input: Exit loop immediately if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up memory and destroy OpenCV GUI windows
cap.release()
cv2.destroyAllWindows()
print("[INFO] Stream terminated cleanly.")