from onvif import ONVIFCamera

CAMERA_IP = "172.20.10.4"
PORT = 2020
USER = "SIDPProject"
PASS = "SIDPUTP2026"

print("[INFO] Connecting via standard ONVIF protocol...")
mycam = ONVIFCamera(CAMERA_IP, PORT, USER, PASS)
resp = mycam.devicemgmt.GetDeviceInformation()
print(f"[SUCCESS] ONVIF Device Info: {resp.Manufacturer} {resp.Model}")