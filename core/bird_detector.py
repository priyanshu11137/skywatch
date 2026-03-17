"""Optional YOLOv8-based bird detection for enhanced accuracy.

Falls back gracefully if ultralytics is not installed.
"""
import numpy as np

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    pass


# COCO class index for 'bird' is 14
BIRD_CLASS_ID = 14


class BirdDetector:
    """YOLOv8-based bird detector. Optional enhancement to motion-based detection."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.3,
                 device: str = "auto"):
        self.enabled = _YOLO_AVAILABLE
        self.confidence = confidence
        self.model = None
        self.device = device

        if self.enabled:
            try:
                self.model = YOLO(model_path)
            except Exception:
                self.enabled = False

    def detect(self, frame: np.ndarray) -> list:
        """Detect birds in a frame.

        Returns list of dicts with 'bbox' (x1,y1,x2,y2), 'confidence', 'center'.
        """
        if not self.enabled or self.model is None:
            return []

        results = self.model(frame, conf=self.confidence, verbose=False,
                             device=self.device)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if cls_id == BIRD_CLASS_ID:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "center": [float(cx), float(cy)],
                        "area": float((x2 - x1) * (y2 - y1)),
                    })
        return detections

    def detect_as_motion_pixels(self, frame: np.ndarray) -> np.ndarray:
        """Convert YOLO detections to motion-pixel format (Nx3: x, y, brightness).

        This allows YOLO detections to be fed into the voxel engine
        alongside or instead of motion-based detection.
        """
        detections = self.detect(frame)
        if not detections:
            return np.empty((0, 3), dtype=np.float64)

        pixels = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            brightness = det["confidence"]
            # Add center point and a few surrounding points
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    pixels.append([cx + dx, cy + dy, brightness])

        return np.array(pixels, dtype=np.float64)

    @staticmethod
    def is_available() -> bool:
        return _YOLO_AVAILABLE
