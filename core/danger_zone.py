"""Runway danger volume geometry and collision probability calculations."""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.stats import norm


@dataclass
class DangerZone:
    """Axis-aligned rectangular prism representing the runway danger volume.

    The box is defined by a center point, half-width (X-axis), half-length
    (Y-axis), and an altitude band (Z-axis).
    """
    runway_center: Tuple[float, float, float] = (0.0, 200.0, 0.0)
    runway_half_width: float = 300.0
    runway_half_length: float = 400.0
    altitude_min: float = 10.0
    altitude_max: float = 150.0

    def __post_init__(self):
        self._bounds_min = np.array([
            self.runway_center[0] - self.runway_half_width,
            self.runway_center[1] - self.runway_half_length,
            self.altitude_min,
        ])
        self._bounds_max = np.array([
            self.runway_center[0] + self.runway_half_width,
            self.runway_center[1] + self.runway_half_length,
            self.altitude_max,
        ])

    def contains(self, point: np.ndarray) -> bool:
        """Test whether a 3D point lies inside the danger volume."""
        p = np.asarray(point)
        return bool(np.all(p >= self._bounds_min) and np.all(p <= self._bounds_max))

    def overlap_probability(self, mean: np.ndarray, covariance: np.ndarray) -> float:
        """Approximate probability that a Gaussian distribution overlaps the box.

        Treats each axis as independent (uses diagonal of covariance) and
        computes the product of per-axis CDF intervals.
        """
        mean = np.asarray(mean, dtype=np.float64)
        cov = np.asarray(covariance, dtype=np.float64)

        prob = 1.0
        for i in range(3):
            sigma = np.sqrt(max(cov[i, i], 1e-12))
            p_lower = norm.cdf(self._bounds_min[i], loc=mean[i], scale=sigma)
            p_upper = norm.cdf(self._bounds_max[i], loc=mean[i], scale=sigma)
            axis_prob = max(p_upper - p_lower, 0.0)
            prob *= axis_prob
            if prob < 1e-15:
                return 0.0

        return float(np.clip(prob, 0.0, 1.0))

    def time_to_entry(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """Linear estimate of time for a point to reach the danger volume.

        Returns float('inf') if the point is heading away or stationary.
        Returns 0.0 if the point is already inside.
        """
        pos = np.asarray(position, dtype=np.float64)
        vel = np.asarray(velocity, dtype=np.float64)

        if self.contains(pos):
            return 0.0

        speed = np.linalg.norm(vel)
        if speed < 1e-6:
            return float('inf')

        t_min = float('inf')

        for i in range(3):
            if abs(vel[i]) < 1e-9:
                if pos[i] < self._bounds_min[i] or pos[i] > self._bounds_max[i]:
                    return float('inf')
                continue

            t_lo = (self._bounds_min[i] - pos[i]) / vel[i]
            t_hi = (self._bounds_max[i] - pos[i]) / vel[i]
            t_enter = min(t_lo, t_hi)
            t_exit = max(t_lo, t_hi)

            if t_exit < 0:
                continue
            if t_enter < 0:
                t_enter = 0.0

            entry_point = pos + vel * t_enter
            inside = True
            for j in range(3):
                if j == i:
                    continue
                if entry_point[j] < self._bounds_min[j] - 1e-6 or entry_point[j] > self._bounds_max[j] + 1e-6:
                    inside = False
                    break

            if inside and t_enter < t_min:
                t_min = t_enter

        return t_min

    def get_bounds(self) -> Dict:
        """Return box min/max corners for frontend rendering."""
        return {
            "min": self._bounds_min.tolist(),
            "max": self._bounds_max.tolist(),
            "center": list(self.runway_center),
        }
