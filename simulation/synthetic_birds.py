"""Generate synthetic multi-camera footage of birds flying over a runway.

Creates realistic demo data without needing Blender or real cameras.
Uses OpenCV to render birds from multiple camera perspectives.
"""
import os
import math
import cv2
import numpy as np
from typing import List, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.camera import Camera
from config import CameraConfig


class BirdTrajectory:
    """A single bird's flight path in 3D space."""

    def __init__(self, start_pos: np.ndarray, velocity: np.ndarray,
                 wobble_freq: float = 0.5, wobble_amp: float = 2.0,
                 bird_id: int = 0):
        self.start = start_pos.copy()
        self.velocity = velocity.copy()
        self.wobble_freq = wobble_freq
        self.wobble_amp = wobble_amp
        self.bird_id = bird_id

    def position_at(self, t: float) -> np.ndarray:
        """Get bird position at time t (seconds)."""
        base = self.start + self.velocity * t
        # Add wing-flap wobble (vertical oscillation)
        wobble_z = self.wobble_amp * math.sin(2 * math.pi * self.wobble_freq * t)
        # Add slight lateral drift
        wobble_x = self.wobble_amp * 0.3 * math.sin(
            2 * math.pi * self.wobble_freq * 0.7 * t + self.bird_id
        )
        return base + np.array([wobble_x, 0, wobble_z])


def create_default_birds(num_birds: int = 8, seed: int = 42) -> List[BirdTrajectory]:
    """Create a set of bird trajectories over a runway area."""
    rng = np.random.RandomState(seed)
    birds = []

    # Flock 1: crossing the runway west-to-east at medium altitude
    for i in range(4):
        start = np.array([
            -250 + rng.uniform(-20, 20),
            150 + rng.uniform(-30, 30),
            60 + rng.uniform(-15, 15),
        ])
        vel = np.array([
            12 + rng.uniform(-2, 2),  # ~12 m/s eastward
            rng.uniform(-1, 1),
            rng.uniform(-0.5, 0.5),
        ])
        birds.append(BirdTrajectory(start, vel, wobble_freq=2 + rng.uniform(-0.5, 0.5),
                                     wobble_amp=1.5 + rng.uniform(-0.5, 0.5), bird_id=i))

    # Flock 2: circling pattern at higher altitude
    for i in range(3):
        angle_offset = i * 2 * math.pi / 3
        start = np.array([
            50 * math.cos(angle_offset),
            200 + 50 * math.sin(angle_offset),
            90 + rng.uniform(-10, 10),
        ])
        # Circular velocity (will be overridden by circular motion)
        vel = np.array([0, 0, 0])
        birds.append(CirclingBird(start, radius=60 + rng.uniform(-10, 10),
                                   altitude=90 + rng.uniform(-10, 10),
                                   period=15 + rng.uniform(-3, 3),
                                   bird_id=4 + i, phase=angle_offset))

    # Single high-altitude bird
    start = np.array([-200, 300, 120])
    vel = np.array([8, -3, -0.2])
    birds.append(BirdTrajectory(start, vel, wobble_freq=3, wobble_amp=1.0,
                                 bird_id=7))

    return birds


class CirclingBird(BirdTrajectory):
    """A bird flying in a circular pattern."""

    def __init__(self, center: np.ndarray, radius: float = 50,
                 altitude: float = 80, period: float = 15,
                 bird_id: int = 0, phase: float = 0):
        super().__init__(center, np.zeros(3), bird_id=bird_id)
        self.center = center.copy()
        self.radius = radius
        self.altitude = altitude
        self.period = period
        self.phase = phase

    def position_at(self, t: float) -> np.ndarray:
        angle = 2 * math.pi * t / self.period + self.phase
        x = self.center[0] + self.radius * math.cos(angle)
        y = self.center[1] + self.radius * math.sin(angle)
        z = self.altitude + 2.0 * math.sin(4 * math.pi * t / self.period)
        return np.array([x, y, z])


def render_sky_background(width: int, height: int) -> np.ndarray:
    """Render a gradient blue sky background."""
    sky = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / height
        # Sky blue gradient (top: deep blue, bottom: lighter blue)
        b = int(220 - 40 * ratio)
        g = int(180 - 50 * ratio)
        r = int(140 - 60 * ratio)
        sky[y, :] = [max(b, 0), max(g, 0), max(r, 0)]
    return sky


def render_frame(camera: Camera, birds: List[BirdTrajectory],
                 t: float, bg: np.ndarray) -> np.ndarray:
    """Render a single frame from a camera's perspective with birds."""
    frame = bg.copy()

    for bird in birds:
        pos = bird.position_at(t)
        pixel = camera.project_point(pos)

        if pixel is not None:
            px, py = int(pixel[0]), int(pixel[1])
            # Bird size varies with distance
            dist = np.linalg.norm(pos - camera.position)
            # Apparent size: ~0.5m wingspan at distance
            size_pixels = max(2, int(camera.focal_length * 0.5 / max(dist, 1)))

            # Draw bird as dark spot (silhouette against sky)
            color = (30, 30, 40)  # dark
            cv2.circle(frame, (px, py), size_pixels, color, -1)

            # Add tiny wing shapes for larger appearances
            if size_pixels >= 4:
                wing_span = size_pixels * 2
                cv2.line(frame, (px - wing_span, py - 1),
                         (px, py), color, max(1, size_pixels // 3))
                cv2.line(frame, (px, py),
                         (px + wing_span, py - 1), color, max(1, size_pixels // 3))

    return frame


def generate_demo_data(output_dir: str, num_frames: int = 120,
                       fps: float = 10.0, resolution: Tuple[int, int] = (1280, 720),
                       num_birds: int = 8) -> dict:
    """Generate complete synthetic demo dataset.

    Args:
        output_dir: Directory to save frames and metadata.
        num_frames: Number of frames to generate.
        fps: Frames per second (for time calculation).
        resolution: Image resolution (width, height).
        num_birds: Number of birds to simulate.

    Returns:
        dict with camera configs, bird trajectories, and metadata.
    """
    width, height = resolution

    # Define 3 camera positions (simulating phones placed around an area)
    camera_configs = [
        CameraConfig("cam0", (-200, -100, 5), yaw=30, pitch=-5, fov=70,
                      resolution=resolution),
        CameraConfig("cam1", (200, -100, 5), yaw=-30, pitch=-5, fov=70,
                      resolution=resolution),
        CameraConfig("cam2", (0, -350, 5), yaw=0, pitch=-3, fov=70,
                      resolution=resolution),
    ]

    cameras = [Camera(cfg) for cfg in camera_configs]
    birds = create_default_birds(num_birds)
    bg = render_sky_background(width, height)

    # Create output directories
    for cfg in camera_configs:
        cam_dir = os.path.join(output_dir, cfg.camera_id)
        os.makedirs(cam_dir, exist_ok=True)

    # Store ground-truth bird positions
    ground_truth = []

    print(f"Generating {num_frames} frames for {len(cameras)} cameras...")

    for frame_idx in range(num_frames):
        t = frame_idx / fps

        # Record ground truth
        gt_frame = {"frame": frame_idx, "time": t, "birds": []}
        for bird in birds:
            pos = bird.position_at(t)
            gt_frame["birds"].append({
                "id": bird.bird_id,
                "position": pos.tolist(),
            })
        ground_truth.append(gt_frame)

        # Render from each camera
        for cam_cfg, camera in zip(camera_configs, cameras):
            frame = render_frame(camera, birds, t, bg)
            filename = f"frame_{frame_idx:06d}.png"
            filepath = os.path.join(output_dir, cam_cfg.camera_id, filename)
            cv2.imwrite(filepath, frame)

        if (frame_idx + 1) % 30 == 0:
            print(f"  Generated {frame_idx + 1}/{num_frames} frames")

    # Save metadata
    import json
    metadata = {
        "cameras": [
            {
                "camera_id": cfg.camera_id,
                "position": list(cfg.position),
                "yaw": cfg.yaw, "pitch": cfg.pitch, "roll": cfg.roll,
                "fov": cfg.fov,
                "resolution": list(cfg.resolution),
                "num_frames": num_frames,
            }
            for cfg in camera_configs
        ],
        "fps": fps,
        "num_frames": num_frames,
        "num_birds": num_birds,
        "ground_truth": ground_truth,
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Demo data saved to {output_dir}")
    print(f"  {len(cameras)} cameras x {num_frames} frames = "
          f"{len(cameras) * num_frames} images")

    return metadata


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "demo")
    generate_demo_data(output)
