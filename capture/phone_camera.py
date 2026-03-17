"""Phone camera integration via RTSP/HTTP streams.

Supported apps:
  - Android: 'IP Webcam' (free) -> http://<phone-ip>:8080/video
  - iPhone:  'IPCamera Lite' or 'Camo' -> varies by app
  - Both:    'DroidCam' / 'iVCam' -> USB or WiFi webcam emulation

Usage:
  1. Install a camera streaming app on each phone
  2. Connect all phones to the same WiFi network
  3. Note each phone's IP address and stream URL
  4. Configure in SkyWatch dashboard or config file
"""
import cv2
import time
import threading
import numpy as np
from typing import Optional


class PhoneCamera:
    """Captures frames from a phone camera stream."""

    def __init__(self, camera_id: str, stream_url: str,
                 resolution: tuple = (1280, 720)):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0
        self.connected = False
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """Attempt to connect to the camera stream."""
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                self.error = f"Cannot open stream: {self.stream_url}"
                self.connected = False
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = self.cap.read()
            if not ret:
                self.error = "Connected but cannot read frames"
                self.connected = False
                return False

            self.last_frame = frame
            self.last_frame_time = time.time()
            self.connected = True
            self.error = None
            return True
        except Exception as e:
            self.error = str(e)
            self.connected = False
            return False

    def start_capture(self):
        """Start background frame capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.last_frame = frame
                    self.last_frame_time = time.time()
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        with self._lock:
            return self.last_frame.copy() if self.last_frame is not None else None

    def stop(self):
        """Stop capture and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.connected = False

    def get_status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "connected": self.connected,
            "stream_url": self.stream_url,
            "error": self.error,
            "last_frame_age": time.time() - self.last_frame_time if self.last_frame_time else None,
            "resolution": self.resolution,
        }


class PhoneCameraManager:
    """Manages multiple phone camera connections."""

    def __init__(self):
        self.cameras: dict[str, PhoneCamera] = {}

    def add_camera(self, camera_id: str, stream_url: str,
                   resolution: tuple = (1280, 720)) -> PhoneCamera:
        cam = PhoneCamera(camera_id, stream_url, resolution)
        self.cameras[camera_id] = cam
        return cam

    def connect_all(self) -> dict:
        """Try to connect all cameras. Returns status dict."""
        results = {}
        for cam_id, cam in self.cameras.items():
            success = cam.connect()
            if success:
                cam.start_capture()
            results[cam_id] = cam.get_status()
        return results

    def get_synchronized_frames(self) -> dict:
        """Get one frame from each connected camera."""
        frames = {}
        for cam_id, cam in self.cameras.items():
            if cam.connected:
                frame = cam.get_frame()
                if frame is not None:
                    frames[cam_id] = frame
        return frames

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()

    def get_all_status(self) -> list:
        return [cam.get_status() for cam in self.cameras.values()]


# Common stream URL templates for popular apps
STREAM_TEMPLATES = {
    "ip_webcam_android": "http://{ip}:8080/video",
    "ip_webcam_android_shot": "http://{ip}:8080/shot.jpg",
    "droidcam": "http://{ip}:4747/video",
    "ivcam": "http://{ip}:8000/video",
    "rtsp_generic": "rtsp://{ip}:{port}/live",
}


def get_stream_url(app_name: str, ip: str, port: int = 8080) -> str:
    """Generate stream URL for a known camera app."""
    template = STREAM_TEMPLATES.get(app_name, "http://{ip}:{port}/video")
    return template.format(ip=ip, port=port)
