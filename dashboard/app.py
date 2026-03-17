"""SkyWatch dashboard backend — FastAPI + WebSocket for real-time updates."""
import os
import sys
import json
import time
import asyncio
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AppConfig, CameraConfig
from core.voxel_engine import VoxelEngine
from core.motion_detector import MotionDetector
from core.camera import Camera
from core.tracker import Tracker
from capture.video_loader import VideoLoader

app = FastAPI(title="SkyWatch", version="1.0.0")
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)

# Global state
state = {
    "config": AppConfig(),
    "engine": None,
    "cameras": {},
    "detectors": {},
    "tracker": Tracker(),
    "video_loader": VideoLoader(),
    "processing": False,
    "current_frame": 0,
    "total_frames": 0,
    "ws_clients": set(),
    "paused": False,
    "speed": 1.0,
    "demo_metadata": None,
}


def init_engine():
    """Initialize the voxel engine from config."""
    cfg = state["config"]
    state["engine"] = VoxelEngine(
        grid_size=cfg.voxel.grid_size,
        voxel_size=cfg.voxel.voxel_size,
        grid_center=cfg.voxel.grid_center,
        attenuation=cfg.voxel.distance_attenuation,
    )
    state["cameras"] = {}
    state["detectors"] = {}
    for cam_cfg in cfg.cameras:
        cam = Camera(cam_cfg)
        state["cameras"][cam_cfg.camera_id] = cam
        state["detectors"][cam_cfg.camera_id] = MotionDetector(
            threshold=cfg.detection.motion_threshold,
            min_pixels=cfg.detection.min_motion_pixels,
        )


@app.on_event("startup")
async def startup():
    init_engine()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    tracker = state["tracker"]
    return {
        "processing": state["processing"],
        "current_frame": state["current_frame"],
        "total_frames": state["total_frames"],
        "paused": state["paused"],
        "cameras": len(state["cameras"]),
        "tracks": tracker.get_stats(),
        "alerts": tracker.get_alerts(),
    }


@app.get("/api/cameras")
async def get_cameras():
    return [
        {
            "id": cid,
            "position": cam.position.tolist(),
            "yaw": float(np.degrees(cam.yaw)),
            "pitch": float(np.degrees(cam.pitch)),
            "fov": float(np.degrees(cam.fov)),
        }
        for cid, cam in state["cameras"].items()
    ]


@app.get("/api/checkerboard")
async def get_checkerboard():
    """Generate and serve a printable checkerboard image."""
    from core.calibration import generate_checkerboard_image
    import io
    img = generate_checkerboard_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return {"image": b64, "format": "png",
            "instructions": "Print at 100% scale on A4. Default square = 30mm."}


@app.post("/api/calibrate")
async def calibrate_camera(data: dict):
    """Calibrate a single camera from a checkerboard photo.

    Expects: { camera_id, image_b64, fov (optional), square_size_mm (optional),
               camera_height_m (optional) }
    """
    from core.calibration import calibrate_from_image
    import io

    cam_id = data.get("camera_id", "cam0")
    image_b64 = data.get("image_b64", "")
    fov = data.get("fov", 70.0)
    square_mm = data.get("square_size_mm", 30.0)
    height_m = data.get("camera_height_m")

    # Decode image
    img_bytes = base64.b64decode(image_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    result = calibrate_from_image(
        image, square_size_mm=square_mm, camera_fov=fov,
        camera_height_m=height_m,
    )

    if result is None:
        return JSONResponse({"error": "Checkerboard not found in image. "
                             "Ensure the full board is visible and well-lit."},
                            status_code=400)

    # Encode preview image
    _, preview_buf = cv2.imencode(".jpg", result["preview_image"],
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
    preview_b64 = base64.b64encode(preview_buf).decode("ascii")

    return {
        "camera_id": cam_id,
        "position": result["position"],
        "yaw": result["yaw"],
        "pitch": result["pitch"],
        "roll": result["roll"],
        "fov": result["fov"],
        "reprojection_error": result["reprojection_error"],
        "preview_image": preview_b64,
    }


@app.post("/api/apply-calibration")
async def apply_calibration(data: dict):
    """Apply calibration results to camera configuration.

    Expects: { cameras: [ { camera_id, position, yaw, pitch, roll, fov } ] }
    """
    cfg = state["config"]
    cfg.cameras = []
    for cam_data in data.get("cameras", []):
        cam_cfg = CameraConfig(
            camera_id=cam_data["camera_id"],
            position=tuple(cam_data["position"]),
            yaw=cam_data["yaw"],
            pitch=cam_data["pitch"],
            roll=cam_data.get("roll", 0),
            fov=cam_data.get("fov", 70),
        )
        cfg.cameras.append(cam_cfg)

    init_engine()
    state["tracker"].reset()
    for det in state["detectors"].values():
        det.reset()

    return {"status": "ok", "cameras_configured": len(cfg.cameras)}


@app.post("/api/load-demo")
async def load_demo():
    """Load the synthetic demo dataset."""
    demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo"

    meta_path = demo_dir / "metadata.json"
    if not meta_path.exists():
        # Generate demo data first
        from simulation.synthetic_birds import generate_demo_data
        generate_demo_data(str(demo_dir))

    with open(meta_path) as f:
        metadata = json.load(f)

    state["demo_metadata"] = metadata

    # Reconfigure cameras from demo metadata
    cfg = state["config"]
    cfg.cameras = []
    for cam_meta in metadata["cameras"]:
        cam_cfg = CameraConfig(
            camera_id=cam_meta["camera_id"],
            position=tuple(cam_meta["position"]),
            yaw=cam_meta["yaw"],
            pitch=cam_meta["pitch"],
            roll=cam_meta.get("roll", 0),
            fov=cam_meta["fov"],
            resolution=tuple(cam_meta["resolution"]),
        )
        cfg.cameras.append(cam_cfg)

    init_engine()
    state["total_frames"] = metadata["num_frames"]
    state["current_frame"] = 0
    state["processing"] = False
    state["paused"] = False
    state["tracker"].reset()
    # Reset all motion detectors so frame 0 is treated fresh
    for det in state["detectors"].values():
        det.reset()

    return {"status": "ok", "total_frames": metadata["num_frames"],
            "num_cameras": len(metadata["cameras"])}


@app.post("/api/load-videos")
async def load_videos(data: dict):
    """Load pre-recorded video files."""
    loader = state["video_loader"]
    loader.release_all()

    for entry in data.get("videos", []):
        cam_id = entry["camera_id"]
        path = entry["path"]
        offset = entry.get("offset", 0)
        loader.add_video(cam_id, path, offset)

    state["total_frames"] = loader.total_frames
    state["current_frame"] = 0
    state["tracker"].reset()

    return {"status": "ok", "total_frames": loader.total_frames}


@app.post("/api/control")
async def control(data: dict):
    """Playback control: play, pause, step, reset, speed."""
    action = data.get("action")
    if action == "pause":
        state["paused"] = True
    elif action == "play":
        state["paused"] = False
    elif action == "stop":
        # Force-stop processing so demo can be re-loaded
        state["processing"] = False
        state["paused"] = False
    elif action == "reset":
        state["processing"] = False
        state["paused"] = False
        state["current_frame"] = 0
        state["tracker"].reset()
        for det in state["detectors"].values():
            det.reset()
        if state["engine"]:
            state["engine"].clear()
    elif action == "speed":
        state["speed"] = max(0.25, min(10.0, data.get("value", 1.0)))
    elif action == "seek":
        state["current_frame"] = max(0, min(
            data.get("frame", 0), state["total_frames"] - 1
        ))
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state["ws_clients"].add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data.get("action") == "start_processing":
                asyncio.create_task(process_frames(ws))
            elif data.get("action") == "process_single":
                await process_single_frame(ws)
    except WebSocketDisconnect:
        state["ws_clients"].discard(ws)


async def broadcast(msg: dict):
    """Send message to all connected WebSocket clients."""
    dead = set()
    for ws in state["ws_clients"]:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    state["ws_clients"] -= dead


async def process_frames(ws: WebSocket):
    """Process all frames through the voxel pipeline."""
    if state["processing"]:
        return
    state["processing"] = True
    state["paused"] = False

    engine = state["engine"]
    tracker = state["tracker"]
    demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
    has_demo = state["demo_metadata"] is not None
    has_videos = len(state["video_loader"].videos) > 0
    fps = state["demo_metadata"]["fps"] if has_demo else 10.0

    try:
        while state["current_frame"] < state["total_frames"]:
            # Check if processing was force-stopped (e.g. by reload)
            if not state["processing"]:
                break
            if state["paused"]:
                await asyncio.sleep(0.1)
                continue

            frame_idx = state["current_frame"]
            engine.clear()

            # Reset motion detectors every frame
            # (they keep state from previous frame internally)

            camera_frames = {}

            if has_demo:
                # Load demo frames
                for cam_id, cam in state["cameras"].items():
                    img_path = demo_dir / cam_id / f"frame_{frame_idx:06d}.png"
                    if img_path.exists():
                        frame = cv2.imread(str(img_path))
                        if frame is not None:
                            camera_frames[cam_id] = frame
            elif has_videos:
                camera_frames = state["video_loader"].get_synchronized_frames(frame_idx)

            # Process each camera
            for cam_id, frame in camera_frames.items():
                cam = state["cameras"].get(cam_id)
                detector = state["detectors"].get(cam_id)
                if cam is None or detector is None:
                    continue

                _, motion_pixels = detector.detect(frame)
                if motion_pixels is not None and len(motion_pixels) > 0:
                    pos, rot, fl, w, h = cam.get_params_flat()
                    engine.accumulate(motion_pixels, pos, rot, fl, w, h,
                                      camera_id=cam_id)

            # Multi-camera voting: only keep voxels seen by 2+ cameras
            engine.finalize(min_cameras=2)

            # Extract detections and update tracker
            cam_positions = [c.position.tolist() for c in state["cameras"].values()]
            detections = engine.get_detections(
                percentile=state["config"].voxel.detection_percentile,
                threshold_ratio=state["config"].detection.peak_threshold_ratio,
                camera_positions=cam_positions,
                min_altitude=state["config"].detection.alert_altitude_min,
            )
            active_tracks = tracker.update(detections)
            alerts = tracker.get_alerts()

            # Get voxel visualization data (occupied voxels)
            occupied = engine.get_occupied_voxels(percentile=97.0)

            # Encode a camera frame as thumbnail for display
            thumbnails = {}
            for cam_id, frame in camera_frames.items():
                small = cv2.resize(frame, (320, 180))
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                thumbnails[cam_id] = base64.b64encode(buf).decode("ascii")

            # Get ground truth if available
            gt_birds = []
            if has_demo and state["demo_metadata"]:
                gt_data = state["demo_metadata"].get("ground_truth", [])
                if frame_idx < len(gt_data):
                    gt_birds = gt_data[frame_idx].get("birds", [])

            # Build update message
            update = {
                "type": "frame_update",
                "frame": frame_idx,
                "total_frames": state["total_frames"],
                "tracks": [t.to_dict() for t in active_tracks],
                "alerts": alerts,
                "stats": tracker.get_stats(),
                "thumbnails": thumbnails,
                "voxels": occupied.tolist() if len(occupied) > 0 else [],
                "ground_truth": gt_birds,
                "detections": detections[:20],
                "camera_positions": cam_positions,
            }

            await broadcast(update)

            state["current_frame"] += 1
            await asyncio.sleep(max(0.01, (1.0 / fps) / state["speed"]))

    finally:
        state["processing"] = False
        await broadcast({"type": "processing_complete"})


async def process_single_frame(ws: WebSocket):
    """Process just one frame (for step-by-step mode)."""
    state["current_frame"] = min(
        state["current_frame"], state["total_frames"] - 1
    )
    # Temporarily unpause, process, then re-pause
    old_pause = state["paused"]
    state["paused"] = False
    state["total_frames_temp"] = state["current_frame"] + 1
    # Just send the current state
    await broadcast({"type": "step_complete", "frame": state["current_frame"]})


def run_dashboard(host="0.0.0.0", port=8050):
    """Run the dashboard server."""
    import uvicorn
    print(f"\n{'='*60}")
    print(f"  SKYWATCH Dashboard")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
