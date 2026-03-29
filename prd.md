Proposed Feature: Predictive Flight Path Forecasting with Collision Risk Scoring
This is the highest-value natural extension of the current system. Right now, SkyWatch alerts are reactive — they fire only when a bird is already inside the runway danger zone. A predictive system would forecast bird trajectories seconds into the future and score the probability of a runway incursion before it happens, giving ATC operators actionable lead time.
PRD: Predictive Flight Path Forecasting & Collision Risk Scoring
1. Problem Statement
The current alerting system in SkyWatch (Tracker.get_alerts()) evaluates bird positions against a static geofence at the current instant. By the time an alert fires, the bird is already in the danger zone and there may be only 2–5 seconds before a potential strike event. Airport bird management teams need predictive lead time — ideally 10–30 seconds of advance warning — to activate deterrent systems (pyrotechnics, laser hazing, broadcast distress calls) or to advise ATC to delay a departure/go-around.
The existing Track dataclass already stores position history and computes instantaneous velocity, but this data is not used for forecasting. The linear predicted_position() method (used only for tracker association) is too simplistic for real bird flight paths, which involve turns, altitude changes, and flock dynamics.
2. Goals
Goal	Metric
Provide advance warning of runway incursions	Alert lead time >= 10 seconds before a bird enters the danger zone
Reduce false positive predictive alerts	Predictive alert precision >= 70% (i.e., 70% of predicted incursions actually occur within 30s)
Quantify collision risk as a continuous score	Expose a 0.0–1.0 risk score per track, updated every frame
Visualize predicted paths in the 3D dashboard	Show probability cones / forecast trajectories in the Three.js viewport
Zero regression to existing detection/tracking	Existing motion detection, voxel engine, and reactive alerts remain unchanged
3. Non-Goals
Automated deterrent activation (future feature; this PRD only covers detection + scoring + visualization)
Species-specific flight models (use a generic kinematic model for v1)
Integration with external ATC systems or NOTAM feeds
Replacing the existing reactive alert system (predictive alerts supplement, not replace)
4. Technical Design
4.1 New Module: core/predictor.py
This is the core addition. It consumes Track objects from the existing Tracker and produces forecasted trajectories + risk scores.
Kalman Filter per Track
Replace the naive linear velocity estimate with a constant-acceleration Kalman filter. State vector per track:
Prediction step: Standard kinematic projection using dt between frames.
Update step: Incorporate each new 3D detection as a measurement (position-only observation, H matrix selects x/y/z).
Process noise Q: Tuned to bird maneuverability — birds can change direction at ~2–5 m/s² laterally, ~3 m/s² vertically. Use adaptive process noise that increases Q when recent residuals are large (the bird is maneuvering).
The Kalman filter gives us not just a predicted position but a covariance ellipsoid at each future timestep, which naturally encodes uncertainty — the further into the future, the larger the ellipsoid.
Trajectory Forecast
Given the current Kalman state, propagate forward in 0.5-second steps for up to 30 seconds (60 forecast points). At each step, record:
Mean predicted position (x, y, z)
3-sigma covariance envelope (for the probability cone)
Collision Risk Score
For each forecast timestep, compute the probability that the bird's position distribution overlaps with the runway danger volume. The danger volume is a rectangular prism defined by:
Runway centerline ± half-width (existing config: alert_runway_distance)
Altitude band: alert_altitude_min to alert_altitude_max
The overlap probability at each timestep is approximated by evaluating the Gaussian CDF over the box bounds. The track risk score is the maximum overlap probability across all forecast timesteps:
This yields a float in [0.0, 1.0] where:
< 0.2 → LOW (bird heading away or at safe altitude)
0.2 – 0.6 → MEDIUM (possible incursion in 10–30s)
> 0.6 → HIGH (likely incursion within 10s)
Integration Point: The predictor is called after Tracker.update() in the main process_frames loop. It reads active tracks, runs the Kalman predict/update cycle, computes forecasts, and returns enriched track data.
4.2 Changes to Existing Modules
core/tracker.py
Add a kalman_state field to the Track dataclass (initialized to None, populated by the predictor on first use).
Add forecast and risk_score fields to Track.to_dict() output so the dashboard receives them.
No changes to the association logic — the predictor is a read-only consumer of tracks.
config.py
Add a new PredictionConfig dataclass:
Add prediction: PredictionConfig to AppConfig.
dashboard/app.py
Import and instantiate the predictor alongside the tracker.
After tracker.update(detections), call predictor.update(active_tracks).
Include forecast (list of [x, y, z] points), risk_score (float), and risk_level (string) in the WebSocket frame_update message per track.
Add a new REST endpoint GET /api/risk-summary returning aggregate risk state.
4.3 Dashboard Frontend Changes
dashboard/templates/index.html
Forecast trajectory cones: For each active track, draw a semi-transparent tapered tube (or line with expanding width) from the current position along the forecast path. Color-code by risk:
Cyan (low risk) → Orange (medium) → Red (high)
Risk score badge: Add a per-track risk score display in the right panel's track list items:
Danger zone visualization: Render the rectangular danger prism as a semi-transparent red box in the 3D scene (currently only the 2D ring exists).
Predictive alert panel entry: New alert type PREDICTIVE with estimated time-to-incursion:
Risk heatmap timeline: A thin colored bar above the existing timeline showing aggregate risk score over time (green → yellow → red).
4.4 New Module: core/danger_zone.py
Encapsulates the runway danger volume geometry and provides:
contains(point) -> bool — point-in-box test
overlap_probability(mean, covariance) -> float — Gaussian-box overlap via scipy.stats.norm.cdf
time_to_entry(position, velocity) -> float — simple linear estimate for display purposes
Configurable from PredictionConfig + existing DetectionConfig
This is separated from the predictor to keep concerns clean and make the danger zone reusable for future features (e.g., automated deterrent triggering).
5. Data Flow (Updated Pipeline)
6. Dependencies
Dependency	Purpose	Already in repo?
numpy	Kalman filter linear algebra	Yes
scipy	scipy.stats.norm.cdf for Gaussian-box overlap, scipy.linalg for matrix operations	Yes
numba (optional)	JIT-compile the Kalman predict step if perf becomes an issue	Yes
No new dependencies required.
7. File Changes Summary
File	Change Type	Description
core/predictor.py	New	Kalman filter, trajectory forecasting, risk scoring
core/danger_zone.py	New	Runway danger volume geometry + overlap probability
core/tracker.py	Modify	Add kalman_state, forecast, risk_score fields to Track
config.py	Modify	Add PredictionConfig dataclass
dashboard/app.py	Modify	Instantiate predictor, call in processing loop, include in WebSocket messages, add /api/risk-summary
dashboard/templates/index.html	Modify	Forecast cones, risk badges, danger zone prism, predictive alerts, risk timeline bar
8. Testing Strategy
Unit tests (tests/test_predictor.py):
Kalman filter converges on a constant-velocity synthetic track within 5 frames
Forecast for a straight-line track at 10 m/s heading toward the runway produces risk > 0.6 when bird is 100m away
Forecast for a track heading away from the runway produces risk < 0.1
Covariance grows monotonically with forecast horizon
Risk score is 0.0 when no tracks exist
Integration test with synthetic demo:
Run the existing demo (--demo --birds 8 --frames 120) and verify:
Predictive alerts fire before reactive alerts for birds crossing the runway
Lead time is measurably > 0 seconds for at least 50% of incursion events
No crash or performance regression (frame processing stays under 100ms)
Visual validation:
Forecast cones visually match bird flight direction
Cones expand over time (uncertainty grows)
Risk color transitions are smooth and match expected thresholds
9. Rollout Plan
Phase	Scope	Acceptance Criteria
Phase 1	core/predictor.py + core/danger_zone.py + unit tests	Kalman filter + risk scoring works on synthetic Track data
Phase 2	Integration into dashboard/app.py processing loop	Risk scores appear in WebSocket messages; existing alerts unaffected
Phase 3	Frontend visualization (forecast cones + risk badges)	Forecast cones render in 3D viewport, risk shows in track list
Phase 4	Predictive alert panel + risk timeline bar	Full end-to-end predictive alerting visible in dashboard
10. Open Questions
Adaptive process noise: Should we use an IMM (Interacting Multiple Model) filter with separate "cruising" and "maneuvering" models, or is adaptive Q sufficient for v1?
Flock coherence: When multiple birds fly in formation, should we model flock-level risk (center-of-mass trajectory) in addition to individual risk?
Runway operations context: Should the risk score be modulated by whether a takeoff/landing is actually scheduled? (This would require external ATC data integration — likely out of scope for v1.)
Alert fatigue: What cooldown period should predictive alerts have to avoid spamming the operator when a bird hovers near the threshold boundary?
This feature builds directly on the existing tracker infrastructure, requires no new dependencies, and the most compute-intensive part (Kalman filter for ~30 tracks) is trivially cheap compared to the voxel engine that already runs every frame. It turns SkyWatch from a real-time monitoring tool into a proactive safety system