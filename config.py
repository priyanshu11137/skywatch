"""SkyWatch configuration."""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEMO_DIR = os.path.join(DATA_DIR, "demo")


@dataclass
class CameraConfig:
    """Single camera configuration."""
    camera_id: str = "cam0"
    position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    yaw: float = 0.0       # degrees
    pitch: float = 0.0     # degrees
    roll: float = 0.0      # degrees
    fov: float = 70.0      # degrees
    resolution: Tuple[int, int] = (1280, 720)
    stream_url: str = ""   # RTSP/HTTP URL for phone cameras


@dataclass
class VoxelConfig:
    """Voxel grid configuration."""
    grid_size: int = 200          # N x N x N voxels
    voxel_size: float = 3.0       # meters per voxel
    grid_center: Tuple[float, float, float] = (0.0, 200.0, 50.0)
    brightness_threshold: float = 0.3
    detection_percentile: float = 99.9
    distance_attenuation: float = 0.0005


@dataclass
class DetectionConfig:
    """Detection and tracking settings."""
    motion_threshold: int = 15
    min_motion_pixels: int = 3
    peak_min_distance: int = 5         # voxels
    peak_threshold_ratio: float = 0.3  # fraction of max
    max_track_distance: float = 50.0   # meters
    track_history_length: int = 30
    alert_altitude_min: float = 10.0   # meters
    alert_altitude_max: float = 150.0  # meters
    alert_runway_distance: float = 300.0  # meters


@dataclass
class AppConfig:
    """Full application configuration."""
    cameras: List[CameraConfig] = field(default_factory=lambda: [
        CameraConfig("cam0", (-200, -100, 5), yaw=30, fov=70),
        CameraConfig("cam1", (200, -100, 5), yaw=-30, fov=70),
        CameraConfig("cam2", (0, -350, 5), yaw=0, fov=70),
    ])
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    frame_rate: int = 10
    dashboard_port: int = 8050
    dashboard_host: str = "0.0.0.0"
