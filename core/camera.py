"""Camera model with pinhole projection and coordinate transforms.

World coordinate system: X=East, Y=North, Z=Up
Camera local: x=right, y=up, z=forward (depth)
When yaw=0, pitch=0: camera faces north (+Y), right is east (+X), up is world up (+Z).
"""
import math
import numpy as np
from config import CameraConfig


class Camera:
    """Pinhole camera model with position, orientation, and projection."""

    def __init__(self, cfg: CameraConfig):
        self.id = cfg.camera_id
        self.position = np.array(cfg.position, dtype=np.float64)
        self.yaw = math.radians(cfg.yaw)
        self.pitch = math.radians(cfg.pitch)
        self.roll = math.radians(cfg.roll)
        self.fov = math.radians(cfg.fov)
        self.width, self.height = cfg.resolution
        self.focal_length = (self.width / 2.0) / math.tan(self.fov / 2.0)
        self.stream_url = cfg.stream_url
        self.rotation_matrix = self._build_rotation()

    def _build_rotation(self) -> np.ndarray:
        """Build rotation matrix from yaw and pitch using look-direction.

        Returns a 3x3 matrix R where columns are [right, up, forward] in world coords.
        Camera local vector [x, y, z] maps to world via R @ [x, y, z].
        """
        # Forward direction: yaw=0 looks along +Y (north), pitch>0 looks up
        forward = np.array([
            math.sin(self.yaw) * math.cos(self.pitch),
            math.cos(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
        ], dtype=np.float64)
        fn = np.linalg.norm(forward)
        if fn > 1e-12:
            forward /= fn

        # World up
        world_up = np.array([0.0, 0.0, 1.0])

        # Right = forward x up
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn

        # Camera up = right x forward
        up = np.cross(right, forward)
        un = np.linalg.norm(up)
        if un > 1e-12:
            up /= un

        # Apply roll around forward axis
        if abs(self.roll) > 1e-6:
            cr, sr = math.cos(self.roll), math.sin(self.roll)
            right_new = cr * right + sr * up
            up_new = -sr * right + cr * up
            right, up = right_new, up_new

        # Columns: [right, up, forward]
        return np.column_stack([right, up, forward]).astype(np.float64)

    def pixel_to_ray(self, px: float, py: float) -> np.ndarray:
        """Convert pixel coordinate to world-space ray direction (unit vector).

        Image convention: px increases right, py increases DOWN.
        Camera local: x=right, y=up (so flip py), z=forward.
        """
        x_local = px - self.width / 2.0
        y_local = -(py - self.height / 2.0)   # flip: image-down -> camera-up
        z_local = self.focal_length

        ray_local = np.array([x_local, y_local, z_local], dtype=np.float64)
        ray_world = self.rotation_matrix @ ray_local
        norm = np.linalg.norm(ray_world)
        if norm < 1e-12:
            return np.array([0, 0, 1], dtype=np.float64)
        return ray_world / norm

    def project_point(self, point_3d: np.ndarray) -> tuple:
        """Project a 3D world point onto image plane. Returns (px, py) or None."""
        relative = point_3d - self.position
        cam_space = self.rotation_matrix.T @ relative  # world -> camera local

        # cam_space[2] = depth (forward), must be positive
        if cam_space[2] <= 0.1:
            return None

        # cam_space[0] = right, cam_space[1] = up
        px = (cam_space[0] / cam_space[2]) * self.focal_length + self.width / 2.0
        py = -(cam_space[1] / cam_space[2]) * self.focal_length + self.height / 2.0  # flip back

        if 0 <= px < self.width and 0 <= py < self.height:
            return (px, py)
        return None

    def get_params_flat(self):
        """Return flat arrays for Numba-compatible functions."""
        return (
            self.position.copy(),
            self.rotation_matrix.copy(),
            self.focal_length,
            self.width,
            self.height,
        )


def gps_to_local(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Convert GPS coordinates to local ENU (East-North-Up) in meters."""
    R = 6_371_000.0
    m_per_deg_lat = (math.pi / 180.0) * R
    m_per_deg_lon = (math.pi / 180.0) * R * math.cos(math.radians(ref_lat))

    x = (lon - ref_lon) * m_per_deg_lon  # East
    y = (lat - ref_lat) * m_per_deg_lat  # North
    z = alt - ref_alt                    # Up

    return np.array([x, y, z], dtype=np.float64)
