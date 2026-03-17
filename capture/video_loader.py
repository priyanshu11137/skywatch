"""Load and synchronize pre-recorded video files from multiple cameras."""
import cv2
import numpy as np
from typing import List, Optional, Tuple


class VideoLoader:
    """Loads video files and provides synchronized frame access."""

    def __init__(self):
        self.videos: dict[str, cv2.VideoCapture] = {}
        self.frame_counts: dict[str, int] = {}
        self.fps_values: dict[str, float] = {}
        self.offsets: dict[str, int] = {}  # frame offsets for sync

    def add_video(self, camera_id: str, video_path: str,
                  frame_offset: int = 0) -> bool:
        """Add a video file for a camera.

        Args:
            camera_id: Identifier matching a CameraConfig.
            video_path: Path to video file.
            frame_offset: Frame offset for synchronization (from clap detection etc.)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        self.videos[camera_id] = cap
        self.frame_counts[camera_id] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_values[camera_id] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.offsets[camera_id] = frame_offset
        return True

    @property
    def total_frames(self) -> int:
        """Effective total frames considering offsets."""
        if not self.videos:
            return 0
        return min(
            self.frame_counts[cid] - self.offsets[cid]
            for cid in self.videos
        )

    @property
    def camera_ids(self) -> list:
        return list(self.videos.keys())

    def get_frame(self, camera_id: str, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame from a camera's video."""
        if camera_id not in self.videos:
            return None

        cap = self.videos[camera_id]
        actual_frame = frame_idx + self.offsets.get(camera_id, 0)

        if actual_frame < 0 or actual_frame >= self.frame_counts[camera_id]:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
        ret, frame = cap.read()
        return frame if ret else None

    def get_synchronized_frames(self, frame_idx: int) -> dict:
        """Get the same logical frame from all cameras."""
        frames = {}
        for cam_id in self.videos:
            frame = self.get_frame(cam_id, frame_idx)
            if frame is not None:
                frames[cam_id] = frame
        return frames

    def detect_sync_offset(self, camera_id: str,
                           search_range: int = 300) -> int:
        """Detect audio sync point (loud clap) for a camera's video.

        This is a simplified approach using frame brightness change
        as a proxy for audio clap (works if someone claps + waves).
        For proper audio sync, extract audio tracks and cross-correlate.
        """
        cap = self.videos.get(camera_id)
        if cap is None:
            return 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_brightness = 0
        max_diff = 0
        max_frame = 0

        for i in range(min(search_range, self.frame_counts[camera_id])):
            ret, frame = cap.read()
            if not ret:
                break
            brightness = np.mean(frame)
            diff = abs(brightness - prev_brightness)
            if diff > max_diff:
                max_diff = diff
                max_frame = i
            prev_brightness = brightness

        return max_frame

    def release_all(self):
        """Release all video captures."""
        for cap in self.videos.values():
            cap.release()
        self.videos.clear()
