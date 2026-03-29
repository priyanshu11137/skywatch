"""Predictive flight path forecasting with Kalman filtering and collision risk scoring."""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from core.danger_zone import DangerZone


@dataclass
class KalmanState:
    """Per-track Kalman filter state for constant-acceleration model.

    State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    """
    x: np.ndarray = field(default_factory=lambda: np.zeros(9))
    P: np.ndarray = field(default_factory=lambda: np.eye(9) * 100.0)
    last_update: float = 0.0
    innovation_norm: float = 0.0


def _build_F(dt: float) -> np.ndarray:
    """Build the 9x9 state transition matrix for constant-acceleration kinematics."""
    F = np.eye(9)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    F[0, 6] = 0.5 * dt * dt
    F[1, 7] = 0.5 * dt * dt
    F[2, 8] = 0.5 * dt * dt
    F[3, 6] = dt
    F[4, 7] = dt
    F[5, 8] = dt
    return F


# Observation matrix: we only observe position [x, y, z]
H = np.zeros((3, 9))
H[0, 0] = 1.0
H[1, 1] = 1.0
H[2, 2] = 1.0


def _build_Q(dt: float, accel_noise: float) -> np.ndarray:
    """Build process noise matrix for constant-acceleration model.

    Uses a piecewise-white-noise jerk model where acceleration changes are
    the noise source.
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt

    q = accel_noise * accel_noise
    Q_block = np.array([
        [dt5 / 20, dt4 / 8, dt3 / 6],
        [dt4 / 8,  dt3 / 3, dt2 / 2],
        [dt3 / 6,  dt2 / 2, dt],
    ]) * q

    Q = np.zeros((9, 9))
    for i in range(3):
        Q[i, i] = Q_block[0, 0]
        Q[i, i + 3] = Q_block[0, 1]
        Q[i, i + 6] = Q_block[0, 2]
        Q[i + 3, i] = Q_block[1, 0]
        Q[i + 3, i + 3] = Q_block[1, 1]
        Q[i + 3, i + 6] = Q_block[1, 2]
        Q[i + 6, i] = Q_block[2, 0]
        Q[i + 6, i + 3] = Q_block[2, 1]
        Q[i + 6, i + 6] = Q_block[2, 2]
    return Q


class FlightPredictor:
    """Predicts bird flight paths using Kalman filtering and scores collision risk."""

    def __init__(self, config, danger_zone: DangerZone):
        self.config = config
        self.danger_zone = danger_zone
        self._states: Dict[int, KalmanState] = {}
        self._last_alert_time: Dict[int, float] = {}

        self._R = np.eye(3) * (config.measurement_noise_pos ** 2)

        self._forecast_steps = int(config.forecast_horizon / config.forecast_step)

    def reset(self):
        """Clear all internal state."""
        self._states.clear()
        self._last_alert_time.clear()

    def _init_state(self, track) -> KalmanState:
        """Initialize a Kalman state from a track's position history."""
        pos = np.array(track.last_position, dtype=np.float64)
        vel = np.array(track.velocity, dtype=np.float64)

        x = np.zeros(9)
        x[0:3] = pos
        x[3:6] = vel

        P = np.eye(9)
        P[0, 0] = P[1, 1] = P[2, 2] = self.config.measurement_noise_pos ** 2
        P[3, 3] = P[4, 4] = P[5, 5] = 25.0   # velocity uncertainty ~5 m/s
        P[6, 6] = P[7, 7] = P[8, 8] = 9.0     # acceleration uncertainty ~3 m/s^2

        now = track.timestamps[-1] if track.timestamps else time.time()
        return KalmanState(x=x, P=P, last_update=now)

    def _predict(self, ks: KalmanState, dt: float) -> None:
        """Kalman predict step (in-place)."""
        if dt < 1e-6:
            return

        F = _build_F(dt)
        Q = _build_Q(dt, self.config.process_noise_accel)

        if ks.innovation_norm > 2.0:
            Q *= self.config.adaptive_noise_factor

        ks.x = F @ ks.x
        ks.P = F @ ks.P @ F.T + Q

    def _update(self, ks: KalmanState, measurement: np.ndarray) -> None:
        """Kalman update step (in-place)."""
        z = np.asarray(measurement, dtype=np.float64)
        y = z - H @ ks.x  # innovation

        S = H @ ks.P @ H.T + self._R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self._reinit_covariance(ks)
            return

        K = ks.P @ H.T @ S_inv
        ks.x = ks.x + K @ y
        I_KH = np.eye(9) - K @ H
        ks.P = I_KH @ ks.P @ I_KH.T + K @ self._R @ K.T  # Joseph form for stability

        try:
            ks.innovation_norm = float(np.sqrt(y @ S_inv @ y) / 3.0)
        except (ValueError, FloatingPointError):
            ks.innovation_norm = 0.0

    def _reinit_covariance(self, ks: KalmanState) -> None:
        """Reset covariance when it becomes degenerate."""
        ks.P = np.eye(9)
        ks.P[0, 0] = ks.P[1, 1] = ks.P[2, 2] = self.config.measurement_noise_pos ** 2
        ks.P[3, 3] = ks.P[4, 4] = ks.P[5, 5] = 25.0
        ks.P[6, 6] = ks.P[7, 7] = ks.P[8, 8] = 9.0

    def _forecast(self, ks: KalmanState) -> List[List[float]]:
        """Propagate state forward to generate forecast trajectory points."""
        dt = self.config.forecast_step
        F = _build_F(dt)
        Q = _build_Q(dt, self.config.process_noise_accel)

        x = ks.x.copy()
        P = ks.P.copy()
        points = []

        for _ in range(self._forecast_steps):
            x = F @ x
            P = F @ P @ F.T + Q
            points.append(x[0:3].tolist())

        return points

    def _compute_risk(self, ks: KalmanState) -> tuple:
        """Compute risk score and time-to-incursion from forecast.

        Returns (risk_score, risk_level, time_to_incursion).
        """
        dt = self.config.forecast_step
        F = _build_F(dt)
        Q = _build_Q(dt, self.config.process_noise_accel)

        x = ks.x.copy()
        P = ks.P.copy()

        max_risk = 0.0
        first_incursion_time = None

        for step in range(self._forecast_steps):
            x = F @ x
            P = F @ P @ F.T + Q

            pos_mean = x[0:3]
            pos_cov = P[0:3, 0:3]

            if not np.all(np.isfinite(pos_mean)) or not np.all(np.isfinite(pos_cov)):
                break

            prob = self.danger_zone.overlap_probability(pos_mean, pos_cov)

            if prob > max_risk:
                max_risk = prob

            if first_incursion_time is None and prob > 0.3:
                first_incursion_time = (step + 1) * dt

        risk_score = float(np.clip(max_risk, 0.0, 1.0))

        if risk_score >= self.config.risk_threshold_high:
            risk_level = "HIGH"
        elif risk_score >= self.config.risk_threshold_low:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if first_incursion_time is None:
            vel = ks.x[3:6]
            pos = ks.x[0:3]
            tte = self.danger_zone.time_to_entry(pos, vel)
            if np.isfinite(tte) and tte < self.config.forecast_horizon:
                first_incursion_time = tte

        return risk_score, risk_level, first_incursion_time

    def update(self, tracks) -> List[dict]:
        """Run prediction cycle for all active tracks.

        Mutates each track's kalman_state, forecast, risk_score, risk_level,
        and time_to_incursion fields in-place.

        Returns a list of per-track prediction dicts (for optional external use).
        """
        now = time.time()
        active_ids = set()
        results = []

        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)

            if tid not in self._states:
                if len(track.positions) < 1:
                    continue
                self._states[tid] = self._init_state(track)

            ks = self._states[tid]

            dt = now - ks.last_update if ks.last_update > 0 else 0.1
            dt = max(min(dt, 5.0), 0.01)  # clamp to sane range

            self._predict(ks, dt)
            self._update(ks, track.last_position)
            ks.last_update = now

            if not np.all(np.isfinite(ks.x)) or not np.all(np.isfinite(ks.P)):
                self._states[tid] = self._init_state(track)
                ks = self._states[tid]

            forecast = self._forecast(ks)
            risk_score, risk_level, tti = self._compute_risk(ks)

            track.kalman_state = ks
            track.forecast = forecast
            track.risk_score = risk_score
            track.risk_level = risk_level
            track.time_to_incursion = tti

            results.append({
                "track_id": tid,
                "forecast": forecast,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "time_to_incursion": tti,
            })

        stale = [tid for tid in self._states if tid not in active_ids]
        for tid in stale:
            del self._states[tid]
            self._last_alert_time.pop(tid, None)

        return results

    def should_alert(self, track_id: int) -> bool:
        """Check if enough time has passed since the last predictive alert for this track."""
        now = time.time()
        last = self._last_alert_time.get(track_id, 0.0)
        if now - last >= self.config.alert_cooldown:
            self._last_alert_time[track_id] = now
            return True
        return False
