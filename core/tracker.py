"""3D object tracker: associates detections across frames into trajectories."""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Track:
    """A tracked object with position history."""
    track_id: int
    positions: List[List[float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    active: bool = True
    missed_frames: int = 0
    label: str = "bird"

    @property
    def last_position(self) -> np.ndarray:
        return np.array(self.positions[-1]) if self.positions else np.zeros(3)

    @property
    def velocity(self) -> np.ndarray:
        if len(self.positions) < 2:
            return np.zeros(3)
        p1 = np.array(self.positions[-1])
        p0 = np.array(self.positions[-2])
        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt < 1e-6:
            return np.zeros(3)
        return (p1 - p0) / dt

    @property
    def altitude(self) -> float:
        return self.positions[-1][2] if self.positions else 0.0

    @property
    def speed(self) -> float:
        v = self.velocity
        return float(np.linalg.norm(v))

    def predicted_position(self, dt: float = 0.1) -> np.ndarray:
        return self.last_position + self.velocity * dt

    def to_dict(self) -> dict:
        return {
            "id": self.track_id,
            "position": self.positions[-1] if self.positions else [0, 0, 0],
            "altitude": self.altitude,
            "speed": round(self.speed, 1),
            "velocity": self.velocity.tolist(),
            "trajectory": self.positions[-30:],  # last 30 points
            "intensity": self.intensities[-1] if self.intensities else 0,
            "label": self.label,
            "active": self.active,
            "age": len(self.positions),
        }


class Tracker:
    """Simple nearest-neighbor tracker with prediction."""

    def __init__(self, max_distance: float = 50.0, max_missed: int = 5,
                 history_length: int = 60):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.history_length = history_length
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.total_detections = 0

    def update(self, detections: list) -> List[Track]:
        """Associate new detections with existing tracks or create new ones.

        Args:
            detections: list of dicts with 'position' and 'intensity'.

        Returns:
            List of currently active tracks.
        """
        now = time.time()
        self.total_detections += len(detections)

        if not detections:
            for track in self.tracks.values():
                if track.active:
                    track.missed_frames += 1
                    if track.missed_frames > self.max_missed:
                        track.active = False
            return self._active_tracks()

        det_positions = np.array([d["position"] for d in detections])
        det_intensities = [d.get("intensity", 1.0) for d in detections]

        active = {tid: t for tid, t in self.tracks.items() if t.active}
        matched_dets = set()
        matched_tracks = set()

        if active:
            # Build cost matrix (distance between predicted positions and detections)
            track_ids = list(active.keys())
            predicted = np.array([active[tid].predicted_position() for tid in track_ids])

            for i, tid in enumerate(track_ids):
                dists = np.linalg.norm(det_positions - predicted[i], axis=1)
                best_j = np.argmin(dists)
                if dists[best_j] < self.max_distance and best_j not in matched_dets:
                    # Match found
                    track = active[tid]
                    track.positions.append(det_positions[best_j].tolist())
                    track.timestamps.append(now)
                    track.intensities.append(det_intensities[best_j])
                    track.missed_frames = 0

                    # Trim history
                    if len(track.positions) > self.history_length:
                        track.positions = track.positions[-self.history_length:]
                        track.timestamps = track.timestamps[-self.history_length:]
                        track.intensities = track.intensities[-self.history_length:]

                    matched_dets.add(best_j)
                    matched_tracks.add(tid)

        # Mark unmatched tracks
        for tid, track in active.items():
            if tid not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed:
                    track.active = False

        # Create new tracks for unmatched detections
        for j in range(len(detections)):
            if j not in matched_dets:
                track = Track(track_id=self.next_id)
                track.positions.append(det_positions[j].tolist())
                track.timestamps.append(now)
                track.intensities.append(det_intensities[j])
                self.tracks[self.next_id] = track
                self.next_id += 1

        return self._active_tracks()

    def _active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.active]

    def get_alerts(self, runway_center=(0, 0, 0), alert_distance=300.0,
                   alt_min=10.0, alt_max=150.0) -> list:
        """Check active tracks for approach-path incursions."""
        alerts = []
        rc = np.array(runway_center[:2])

        for track in self._active_tracks():
            pos = track.last_position
            dist_horiz = np.linalg.norm(pos[:2] - rc)
            alt = pos[2]

            if dist_horiz < alert_distance and alt_min < alt < alt_max:
                severity = "HIGH" if dist_horiz < 100 or alt < 30 else "MEDIUM"
                alerts.append({
                    "track_id": track.track_id,
                    "severity": severity,
                    "message": f"Bird #{track.track_id} at {alt:.0f}m alt, "
                               f"{dist_horiz:.0f}m from runway",
                    "position": pos.tolist(),
                    "speed": track.speed,
                })
        return alerts

    def get_stats(self) -> dict:
        active = self._active_tracks()
        return {
            "active_tracks": len(active),
            "total_tracks": len(self.tracks),
            "total_detections": self.total_detections,
            "avg_altitude": np.mean([t.altitude for t in active]) if active else 0,
        }

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.total_detections = 0
