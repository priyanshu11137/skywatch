"""Tests for the predictive flight path forecasting system."""
import time
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PredictionConfig
from core.danger_zone import DangerZone
from core.predictor import FlightPredictor, KalmanState, _build_F, _build_Q
from core.tracker import Track


def _make_track(track_id, positions, dt=0.1):
    """Create a Track with evenly spaced timestamps."""
    t = Track(track_id=track_id)
    now = time.time()
    for i, pos in enumerate(positions):
        t.positions.append(list(pos))
        t.timestamps.append(now - (len(positions) - 1 - i) * dt)
        t.intensities.append(1.0)
    return t


def _default_predictor():
    cfg = PredictionConfig()
    dz = DangerZone(
        runway_center=(0.0, 200.0, 0.0),
        runway_half_width=300.0,
        runway_half_length=400.0,
        altitude_min=10.0,
        altitude_max=150.0,
    )
    return FlightPredictor(cfg, dz), dz


class TestKalmanConvergence:
    def test_kalman_converges_constant_velocity(self):
        """Kalman filter should converge on a constant-velocity track within 10 frames."""
        predictor, _ = _default_predictor()

        true_vel = np.array([0.0, 10.0, 0.0])
        positions = []
        for i in range(10):
            positions.append([0.0, i * 1.0, 50.0])  # 10 m/s in Y at dt=0.1

        track = _make_track(1, positions, dt=0.1)
        predictor.update([track])

        ks = track.kalman_state
        assert ks is not None
        estimated_vel = ks.x[3:6]
        vel_error = np.linalg.norm(estimated_vel - true_vel) / np.linalg.norm(true_vel)
        assert vel_error < 0.5, f"Velocity error {vel_error:.2f} exceeds 50% tolerance"


class TestForecastRisk:
    def test_forecast_straight_line_toward_runway(self):
        """A bird heading toward the runway danger zone should produce high risk."""
        predictor, _ = _default_predictor()

        positions = []
        for i in range(10):
            positions.append([0.0, i * 1.0, 50.0])  # heading +Y toward runway at y=200

        track = _make_track(1, positions, dt=0.1)
        predictor.update([track])

        assert track.risk_score > 0.1, \
            f"Risk score {track.risk_score} should be > 0.1 for a bird heading toward the runway"
        assert track.forecast is not None
        assert len(track.forecast) > 0

    def test_forecast_heading_away(self):
        """A bird heading away from the runway should produce low risk."""
        predictor, _ = _default_predictor()

        positions = []
        for i in range(10):
            positions.append([0.0, 700.0 + i * 1.0, 50.0])  # heading away from runway

        track = _make_track(1, positions, dt=0.1)
        predictor.update([track])

        assert track.risk_score < 0.1, \
            f"Risk score {track.risk_score} should be < 0.1 for a bird heading away"
        assert track.risk_level == "LOW"


class TestCovarianceGrowth:
    def test_covariance_grows_with_horizon(self):
        """Position covariance trace should be monotonically non-decreasing during forecast."""
        dt = 0.5
        F = _build_F(dt)
        Q = _build_Q(dt, 3.0)

        P = np.eye(9) * 10.0
        x = np.zeros(9)
        x[3] = 5.0  # some velocity

        traces = []
        for _ in range(60):
            x = F @ x
            P = F @ P @ F.T + Q
            traces.append(np.trace(P[0:3, 0:3]))

        for i in range(1, len(traces)):
            assert traces[i] >= traces[i - 1] - 1e-9, \
                f"Covariance trace decreased at step {i}: {traces[i]} < {traces[i-1]}"


class TestNoTracks:
    def test_risk_score_zero_no_tracks(self):
        """Empty track list should return empty results."""
        predictor, _ = _default_predictor()
        results = predictor.update([])
        assert results == []


class TestDangerZone:
    def test_contains_inside(self):
        """Point inside the box should return True."""
        dz = DangerZone(
            runway_center=(0.0, 200.0, 0.0),
            runway_half_width=300.0,
            runway_half_length=400.0,
            altitude_min=10.0,
            altitude_max=150.0,
        )
        assert dz.contains(np.array([0.0, 200.0, 50.0]))
        assert dz.contains(np.array([100.0, 100.0, 80.0]))

    def test_contains_outside(self):
        """Point outside the box should return False."""
        dz = DangerZone(
            runway_center=(0.0, 200.0, 0.0),
            runway_half_width=300.0,
            runway_half_length=400.0,
            altitude_min=10.0,
            altitude_max=150.0,
        )
        assert not dz.contains(np.array([500.0, 200.0, 50.0]))
        assert not dz.contains(np.array([0.0, 200.0, 200.0]))
        assert not dz.contains(np.array([0.0, 200.0, 5.0]))

    def test_overlap_probability_inside(self):
        """Gaussian centered well inside the box with small sigma should have high overlap."""
        dz = DangerZone(
            runway_center=(0.0, 200.0, 0.0),
            runway_half_width=300.0,
            runway_half_length=400.0,
            altitude_min=10.0,
            altitude_max=150.0,
        )
        mean = np.array([0.0, 200.0, 80.0])
        cov = np.diag([1.0, 1.0, 1.0])
        prob = dz.overlap_probability(mean, cov)
        assert prob > 0.99, f"Overlap probability {prob} should be ~1.0"

    def test_overlap_probability_outside(self):
        """Gaussian centered far outside the box should have near-zero overlap."""
        dz = DangerZone(
            runway_center=(0.0, 200.0, 0.0),
            runway_half_width=300.0,
            runway_half_length=400.0,
            altitude_min=10.0,
            altitude_max=150.0,
        )
        mean = np.array([2000.0, 2000.0, 500.0])
        cov = np.diag([1.0, 1.0, 1.0])
        prob = dz.overlap_probability(mean, cov)
        assert prob < 1e-10, f"Overlap probability {prob} should be ~0.0"


class TestTimeToEntry:
    def test_time_to_entry_heading_toward(self):
        """Linear estimate should match expected time for straight approach."""
        dz = DangerZone(
            runway_center=(0.0, 200.0, 0.0),
            runway_half_width=300.0,
            runway_half_length=400.0,
            altitude_min=10.0,
            altitude_max=150.0,
        )
        pos = np.array([0.0, -300.0, 50.0])
        vel = np.array([0.0, 10.0, 0.0])
        tte = dz.time_to_entry(pos, vel)
        # Box Y range: [-200, 600]. From y=-300 heading +Y at 10 m/s:
        # entry at y=-200, time = (-200 - (-300)) / 10 = 10s
        assert abs(tte - 10.0) < 1.0, f"Time to entry {tte} should be ~10s"

    def test_time_to_entry_inside(self):
        """Point already inside should return 0."""
        dz = DangerZone(
            runway_center=(0.0, 200.0, 0.0),
            runway_half_width=300.0,
            runway_half_length=400.0,
            altitude_min=10.0,
            altitude_max=150.0,
        )
        pos = np.array([0.0, 200.0, 50.0])
        vel = np.array([0.0, 10.0, 0.0])
        assert dz.time_to_entry(pos, vel) == 0.0

    def test_time_to_entry_heading_away(self):
        """Point heading away should return inf."""
        dz = DangerZone(
            runway_center=(0.0, 200.0, 0.0),
            runway_half_width=300.0,
            runway_half_length=400.0,
            altitude_min=10.0,
            altitude_max=150.0,
        )
        pos = np.array([0.0, 700.0, 50.0])
        vel = np.array([0.0, 10.0, 0.0])
        tte = dz.time_to_entry(pos, vel)
        assert tte == float('inf'), f"Time to entry {tte} should be inf when heading away"


class TestPredictorCleanup:
    def test_cleanup_stale_states(self):
        """Kalman states for inactive tracks should be removed."""
        predictor, _ = _default_predictor()

        track1 = _make_track(1, [[0, 0, 50], [0, 1, 50], [0, 2, 50]])
        track2 = _make_track(2, [[100, 0, 50], [100, 1, 50], [100, 2, 50]])

        predictor.update([track1, track2])
        assert 1 in predictor._states
        assert 2 in predictor._states

        # Only pass track1 next time (track2 went inactive)
        predictor.update([track1])
        assert 1 in predictor._states
        assert 2 not in predictor._states


class TestAdaptiveNoise:
    def test_adaptive_noise_increases_during_maneuver(self):
        """A sharp turn should increase the innovation norm, triggering adaptive Q."""
        predictor, _ = _default_predictor()

        # Straight line first
        straight_positions = [[0, i * 1.0, 50] for i in range(8)]
        track = _make_track(1, straight_positions, dt=0.1)
        predictor.update([track])

        ks_straight = track.kalman_state
        innov_straight = ks_straight.innovation_norm

        # Sharp turn: suddenly move in X direction
        turn_positions = straight_positions + [[20, 8.0, 50], [40, 8.0, 50]]
        track2 = _make_track(1, turn_positions, dt=0.1)
        predictor._states[1] = ks_straight  # keep the old state
        predictor.update([track2])

        ks_turn = track2.kalman_state
        innov_turn = ks_turn.innovation_norm

        assert innov_turn > innov_straight, \
            f"Innovation after turn ({innov_turn}) should exceed straight-line ({innov_straight})"
