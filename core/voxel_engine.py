"""Voxel engine: casts motion-pixel rays into a 3D voxel grid using DDA.

Uses Numba JIT for near-native performance without C++ compilation.
"""
import numpy as np
import numba


@numba.njit(cache=True)
def _dda_single_ray(origin, direction, grid_min, voxel_size, N,
                    voxel_grid, brightness, attenuation):
    """March a single ray through the voxel grid using DDA, accumulating brightness."""
    grid_max_x = grid_min[0] + N * voxel_size
    grid_max_y = grid_min[1] + N * voxel_size
    grid_max_z = grid_min[2] + N * voxel_size

    # Ray-AABB intersection
    t_min_val = -1e30
    t_max_val = 1e30

    for axis in range(3):
        gmin = grid_min[axis]
        if axis == 0:
            gmax = grid_max_x
        elif axis == 1:
            gmax = grid_max_y
        else:
            gmax = grid_max_z

        d = direction[axis]
        o = origin[axis]

        if abs(d) < 1e-12:
            if o < gmin or o > gmax:
                return
        else:
            t1 = (gmin - o) / d
            t2 = (gmax - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_min_val:
                t_min_val = t1
            if t2 < t_max_val:
                t_max_val = t2

    if t_min_val > t_max_val or t_max_val < 0:
        return

    if t_min_val < 0:
        t_min_val = 0.0

    # Entry point (slightly inside the grid)
    entry_x = origin[0] + direction[0] * (t_min_val + 1e-4)
    entry_y = origin[1] + direction[1] * (t_min_val + 1e-4)
    entry_z = origin[2] + direction[2] * (t_min_val + 1e-4)

    # Starting voxel indices
    ix = int((entry_x - grid_min[0]) / voxel_size)
    iy = int((entry_y - grid_min[1]) / voxel_size)
    iz = int((entry_z - grid_min[2]) / voxel_size)

    ix = min(max(ix, 0), N - 1)
    iy = min(max(iy, 0), N - 1)
    iz = min(max(iz, 0), N - 1)

    # Step directions
    step_x = 1 if direction[0] >= 0 else -1
    step_y = 1 if direction[1] >= 0 else -1
    step_z = 1 if direction[2] >= 0 else -1

    # tDelta: how far along the ray to cross one voxel
    t_delta_x = abs(voxel_size / direction[0]) if abs(direction[0]) > 1e-12 else 1e30
    t_delta_y = abs(voxel_size / direction[1]) if abs(direction[1]) > 1e-12 else 1e30
    t_delta_z = abs(voxel_size / direction[2]) if abs(direction[2]) > 1e-12 else 1e30

    # tMax: distance to next voxel boundary
    if abs(direction[0]) > 1e-12:
        if direction[0] >= 0:
            t_max_x = ((grid_min[0] + (ix + 1) * voxel_size) - entry_x) / direction[0]
        else:
            t_max_x = ((grid_min[0] + ix * voxel_size) - entry_x) / direction[0]
    else:
        t_max_x = 1e30

    if abs(direction[1]) > 1e-12:
        if direction[1] >= 0:
            t_max_y = ((grid_min[1] + (iy + 1) * voxel_size) - entry_y) / direction[1]
        else:
            t_max_y = ((grid_min[1] + iy * voxel_size) - entry_y) / direction[1]
    else:
        t_max_y = 1e30

    if abs(direction[2]) > 1e-12:
        if direction[2] >= 0:
            t_max_z = ((grid_min[2] + (iz + 1) * voxel_size) - entry_z) / direction[2]
        else:
            t_max_z = ((grid_min[2] + iz * voxel_size) - entry_z) / direction[2]
    else:
        t_max_z = 1e30

    # DDA march
    max_steps = N * 3
    for _ in range(max_steps):
        if ix < 0 or ix >= N or iy < 0 or iy >= N or iz < 0 or iz >= N:
            break

        # Distance from camera to current voxel center
        vx = grid_min[0] + (ix + 0.5) * voxel_size
        vy = grid_min[1] + (iy + 0.5) * voxel_size
        vz = grid_min[2] + (iz + 0.5) * voxel_size
        dx = vx - origin[0]
        dy = vy - origin[1]
        dz = vz - origin[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5

        atten = 1.0 / (1.0 + attenuation * dist)
        voxel_grid[ix, iy, iz] += brightness * atten

        # Advance to next voxel
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                ix += step_x
                t_max_x += t_delta_x
            else:
                iz += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                iy += step_y
                t_max_y += t_delta_y
            else:
                iz += step_z
                t_max_z += t_delta_z


@numba.njit(parallel=True, cache=True)
def cast_rays_batch(motion_pixels, cam_pos, cam_rot, focal_length,
                    img_width, img_height, grid_min, voxel_size, N,
                    voxel_grid, attenuation):
    """Cast all motion-pixel rays from one camera into the voxel grid.

    Args:
        motion_pixels: Nx3 array of (px, py, brightness).
        cam_pos: (3,) camera world position.
        cam_rot: (3,3) camera rotation matrix.
        focal_length: derived from FOV.
        img_width, img_height: image dimensions.
        grid_min: (3,) grid minimum corner.
        voxel_size: size of each voxel in meters.
        N: grid dimension count.
        voxel_grid: NxNxN float64 accumulation grid.
        attenuation: distance attenuation factor.
    """
    n_pixels = motion_pixels.shape[0]
    half_w = img_width / 2.0
    half_h = img_height / 2.0

    for i in numba.prange(n_pixels):
        px = motion_pixels[i, 0]
        py = motion_pixels[i, 1]
        brightness = motion_pixels[i, 2]

        # Pixel to camera-local ray direction
        # Camera local: x=right, y=up, z=forward
        # Image: px increases right, py increases DOWN (so flip y)
        local_x = px - half_w
        local_y = -(py - half_h)   # flip y: image-down -> camera-up
        local_z = focal_length

        # Rotate to world space using camera rotation matrix
        # cam_rot columns are [right, up, forward] in world coords
        wx = cam_rot[0, 0] * local_x + cam_rot[0, 1] * local_y + cam_rot[0, 2] * local_z
        wy = cam_rot[1, 0] * local_x + cam_rot[1, 1] * local_y + cam_rot[1, 2] * local_z
        wz = cam_rot[2, 0] * local_x + cam_rot[2, 1] * local_y + cam_rot[2, 2] * local_z

        # Normalize direction
        norm = (wx * wx + wy * wy + wz * wz) ** 0.5
        if norm < 1e-12:
            continue
        dx = wx / norm
        dy = wy / norm
        dz = wz / norm

        direction = np.array([dx, dy, dz])
        _dda_single_ray(cam_pos, direction, grid_min, voxel_size, N,
                        voxel_grid, brightness, attenuation)


class VoxelEngine:
    """Manages the voxel grid and orchestrates ray casting from multiple cameras."""

    def __init__(self, grid_size=200, voxel_size=3.0,
                 grid_center=(0.0, 200.0, 50.0), attenuation=0.0005):
        self.N = grid_size
        self.voxel_size = voxel_size
        self.center = np.array(grid_center, dtype=np.float64)
        self.attenuation = attenuation

        half_extent = (self.N * self.voxel_size) / 2.0
        self.grid_min = np.array([
            self.center[0] - half_extent,
            self.center[1] - half_extent,
            self.center[2] - half_extent,
        ], dtype=np.float64)

        self.voxel_grid = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        # Per-camera grids for multi-camera voting
        self._cam_grids = {}
        self._cam_count = np.zeros((self.N, self.N, self.N), dtype=np.int32)

    def clear(self):
        """Reset the voxel grid to zeros."""
        self.voxel_grid[:] = 0.0
        self._cam_grids.clear()
        self._cam_count[:] = 0

    def accumulate(self, motion_pixels, cam_pos, cam_rot,
                   focal_length, img_width, img_height, camera_id=None):
        """Cast rays from one camera's motion pixels into the grid."""
        if motion_pixels is None or len(motion_pixels) == 0:
            return

        cam_pos_arr = np.ascontiguousarray(cam_pos, dtype=np.float64)
        cam_rot_arr = np.ascontiguousarray(cam_rot, dtype=np.float64)
        mp = np.ascontiguousarray(motion_pixels, dtype=np.float64)
        grid_min = np.ascontiguousarray(self.grid_min, dtype=np.float64)

        if camera_id is not None:
            # Accumulate into per-camera grid for voting
            if camera_id not in self._cam_grids:
                self._cam_grids[camera_id] = np.zeros(
                    (self.N, self.N, self.N), dtype=np.float64
                )
            cast_rays_batch(
                mp, cam_pos_arr, cam_rot_arr,
                float(focal_length), int(img_width), int(img_height),
                grid_min, float(self.voxel_size), int(self.N),
                self._cam_grids[camera_id], float(self.attenuation),
            )
        else:
            cast_rays_batch(
                mp, cam_pos_arr, cam_rot_arr,
                float(focal_length), int(img_width), int(img_height),
                grid_min, float(self.voxel_size), int(self.N),
                self.voxel_grid, float(self.attenuation),
            )

    def finalize(self, min_cameras=2):
        """Combine per-camera grids using multi-camera voting.

        Only voxels where at least min_cameras contributed rays are kept.
        The final value is the geometric mean of contributing cameras.
        """
        if not self._cam_grids:
            return  # Fall back to single combined grid

        # Count how many cameras contributed to each voxel
        self._cam_count[:] = 0
        for cam_grid in self._cam_grids.values():
            self._cam_count += (cam_grid > 0).astype(np.int32)

        # Only keep voxels with sufficient camera coverage
        mask = self._cam_count >= min_cameras

        # Combine using geometric mean (rewards consensus)
        self.voxel_grid[:] = 0.0
        if np.any(mask):
            log_sum = np.zeros_like(self.voxel_grid)
            for cam_grid in self._cam_grids.values():
                safe = np.where(cam_grid > 0, cam_grid, 1.0)
                log_sum += np.log(safe) * (cam_grid > 0).astype(np.float64)

            counts = np.maximum(self._cam_count, 1).astype(np.float64)
            self.voxel_grid[mask] = np.exp(log_sum[mask] / counts[mask])
            # Zero out voxels without enough camera coverage
            self.voxel_grid[~mask] = 0.0

    def get_detections(self, percentile=99.5, min_distance=5, threshold_ratio=0.3,
                       camera_positions=None, min_altitude=8.0):
        """Extract peak detections from the voxel grid.

        Returns list of dicts with 'position' (world coords),
        'voxel_index', and 'intensity'.

        Args:
            camera_positions: list of camera (x,y,z) to filter near-camera artifacts.
            min_altitude: minimum Z coordinate to keep (filters ground-level noise).
        """
        from scipy.ndimage import label

        grid = self.voxel_grid
        max_val = grid.max()
        if max_val < 1e-6:
            return []

        threshold = max(np.percentile(grid, percentile), max_val * threshold_ratio)
        binary = grid > threshold

        # Find connected components
        labeled, num_features = label(binary)

        detections = []
        for feat_id in range(1, num_features + 1):
            region = np.where(labeled == feat_id)
            if len(region[0]) == 0:
                continue

            # Find peak within region
            values = grid[region]
            peak_idx = np.argmax(values)
            ix, iy, iz = region[0][peak_idx], region[1][peak_idx], region[2][peak_idx]

            # Convert voxel index to world position
            wx = self.grid_min[0] + (ix + 0.5) * self.voxel_size
            wy = self.grid_min[1] + (iy + 0.5) * self.voxel_size
            wz = self.grid_min[2] + (iz + 0.5) * self.voxel_size

            # Filter: minimum altitude (removes ground-level camera artifacts)
            if wz < min_altitude:
                continue

            # Filter: too close to any camera position (artifacts)
            if camera_positions:
                too_close = False
                for cp in camera_positions:
                    d = ((wx - cp[0])**2 + (wy - cp[1])**2 + (wz - cp[2])**2)**0.5
                    if d < 30.0:  # within 30m of camera = artifact
                        too_close = True
                        break
                if too_close:
                    continue

            detections.append({
                "position": [float(wx), float(wy), float(wz)],
                "voxel_index": [int(ix), int(iy), int(iz)],
                "intensity": float(values[peak_idx]),
                "volume": int(len(region[0])),
            })

        # Sort by intensity (strongest first)
        detections.sort(key=lambda d: d["intensity"], reverse=True)
        # Limit to top N detections to avoid noise
        return detections[:30]

    def get_grid_bounds(self):
        """Return (min_corner, max_corner) of the voxel grid in world coords."""
        extent = self.N * self.voxel_size
        max_corner = self.grid_min + extent
        return self.grid_min.tolist(), max_corner.tolist()

    def get_occupied_voxels(self, percentile=97.0):
        """Return positions and intensities of high-value voxels for visualization."""
        threshold = np.percentile(self.voxel_grid, percentile)
        if threshold < 1e-6:
            threshold = self.voxel_grid.max() * 0.1
            if threshold < 1e-6:
                return np.empty((0, 4))

        indices = np.argwhere(self.voxel_grid > threshold)
        if len(indices) == 0:
            return np.empty((0, 4))

        positions = self.grid_min + (indices + 0.5) * self.voxel_size
        intensities = self.voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

        # Normalize intensities to [0, 1]
        max_i = intensities.max()
        if max_i > 0:
            intensities = intensities / max_i

        return np.column_stack([positions, intensities])
