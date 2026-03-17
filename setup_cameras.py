"""Interactive phone camera setup utility.

Helps calibrate and test phone cameras for SkyWatch.
Run: python setup_cameras.py
"""
import os
import sys
import json
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture.phone_camera import PhoneCamera, STREAM_TEMPLATES, get_stream_url


def print_header():
    print("\n" + "=" * 60)
    print("  SkyWatch — Phone Camera Setup")
    print("=" * 60)


def print_instructions():
    print("""
  PHONE CAMERA SETUP GUIDE
  ========================

  You need 3 phones as cameras. Here's how to set them up:

  ANDROID PHONE:
    1. Install "IP Webcam" from Google Play (free)
    2. Open the app → scroll down → tap "Start server"
    3. Note the URL shown (e.g., http://192.168.1.5:8080)
    4. Stream URL for SkyWatch: http://<phone-ip>:8080/video

  iPHONE (Option A - Recommended):
    1. Install "IP Camera Lite" from App Store
    2. Open app → Settings → Enable "Video streaming"
    3. Note the stream URL shown in the app
    4. Stream URL varies by app, usually: http://<phone-ip>:8080/video

  iPHONE (Option B):
    1. Install "DroidCam" from App Store (yes, it works on iPhone too)
    2. Open app → note the WiFi IP shown
    3. Stream URL: http://<phone-ip>:4747/video

  IMPORTANT:
    - All phones MUST be on the same WiFi network as this computer
    - Place phones 2-5 meters apart for indoor testing
    - For outdoor testing, spread them 50-200 meters apart
    - Point cameras toward the same area of sky
    - Use stable mounts (tripods, tape to fence, etc.)

  RECORDING MODE (Alternative):
    If streaming doesn't work, you can:
    1. Open the default camera app on all 3 phones
    2. Have someone clap hands in front of all cameras (for sync)
    3. Start recording on all phones as close to simultaneously as possible
    4. Transfer videos to computer
    5. Load in SkyWatch dashboard via the "Load Videos" option
""")


def test_camera_connection(url: str) -> bool:
    """Test if a camera stream URL is accessible."""
    print(f"  Testing: {url}")
    cam = PhoneCamera("test", url)
    success = cam.connect()
    if success:
        frame = cam.get_frame()
        if frame is not None:
            print(f"  Connected! Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cam.stop()
            return True
    cam.stop()
    print(f"  Failed: {cam.error}")
    return False


def interactive_setup():
    """Walk the user through setting up their cameras."""
    print_header()
    print_instructions()

    cameras = []
    phone_names = ["Phone 1 (Android)", "Phone 2 (iPhone)", "Phone 3 (iPhone)"]

    for i, name in enumerate(phone_names):
        print(f"\n--- {name} ---")
        ip = input(f"  Enter {name}'s IP address (or 'skip'): ").strip()
        if ip.lower() == 'skip':
            continue

        # Try common stream URLs
        urls_to_try = [
            f"http://{ip}:8080/video",     # IP Webcam (Android)
            f"http://{ip}:4747/video",     # DroidCam
            f"http://{ip}:8000/video",     # iVCam
        ]

        connected = False
        for url in urls_to_try:
            if test_camera_connection(url):
                cameras.append({
                    "camera_id": f"cam{i}",
                    "name": name,
                    "stream_url": url,
                    "ip": ip,
                })
                connected = True
                break

        if not connected:
            custom_url = input("  Enter full stream URL manually (or 'skip'): ").strip()
            if custom_url and custom_url.lower() != 'skip':
                if test_camera_connection(custom_url):
                    cameras.append({
                        "camera_id": f"cam{i}",
                        "name": name,
                        "stream_url": custom_url,
                        "ip": ip,
                    })

    if cameras:
        print(f"\n  Successfully connected {len(cameras)} cameras!")
        # Save camera config
        config_path = os.path.join(os.path.dirname(__file__), "phone_cameras.json")
        with open(config_path, "w") as f:
            json.dump(cameras, f, indent=2)
        print(f"  Saved to: {config_path}")
        print(f"\n  Now run: python run.py")
        print(f"  Then click 'Load Videos' or set up live streaming in the dashboard")
    else:
        print("\n  No cameras connected.")
        print("  Try the 'Recording Mode' approach described above,")
        print("  or run 'python run.py --demo' for the synthetic demo.")


def preview_camera(url: str):
    """Show a live preview from a camera URL."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Cannot open {url}")
        return

    print("Preview window opened. Press 'q' to close.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"SkyWatch Preview - {url}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        if len(sys.argv) > 2:
            preview_camera(sys.argv[2])
        else:
            print("Usage: python setup_cameras.py --preview <stream_url>")
    else:
        interactive_setup()
