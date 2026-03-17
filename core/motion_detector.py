"""Motion detection via frame differencing."""
import cv2
import numpy as np


class MotionDetector:
    """Detects motion between consecutive frames using absolute differencing."""

    def __init__(self, threshold: int = 25, min_pixels: int = 5):
        self.threshold = threshold
        self.min_pixels = min_pixels
        self.prev_frame = None

    def reset(self):
        self.prev_frame = None

    def detect(self, frame: np.ndarray) -> tuple:
        """Detect motion pixels between current and previous frame.

        Args:
            frame: BGR or grayscale image (numpy array).

        Returns:
            (motion_mask, motion_pixels) where motion_pixels is Nx3 array
            of (x, y, brightness). Returns (None, None) if no previous frame.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        gray = gray.astype(np.float32)

        if self.prev_frame is None:
            self.prev_frame = gray
            return None, None

        diff = np.abs(gray - self.prev_frame)
        self.prev_frame = gray.copy()

        motion_mask = (diff > self.threshold).astype(np.uint8) * 255

        # Optional: morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        ys, xs = np.where(motion_mask > 0)
        if len(xs) < self.min_pixels:
            return motion_mask, np.empty((0, 3), dtype=np.float64)

        brightnesses = diff[ys, xs] / 255.0
        motion_pixels = np.column_stack([
            xs.astype(np.float64),
            ys.astype(np.float64),
            brightnesses,
        ])

        return motion_mask, motion_pixels
