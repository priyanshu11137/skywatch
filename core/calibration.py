"""Camera calibration using a printed checkerboard pattern.

Workflow:
  1. Generate & print a checkerboard (generate_checkerboard_image)
  2. Lay it flat on the ground at your chosen origin point
  3. From each phone's final mounted position, take a photo that includes the board
  4. Call calibrate_from_image() for each photo
  5. Get back exact camera position + orientation in world coords

The checkerboard defines the world coordinate system:
  - Board center = origin (0, 0, 0)
  - Board lies on the ground plane (Z = 0)
  - Board's long edge = Y axis (north)
  - Board's short edge = X axis (east)
  - Z axis = up
"""
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Default board parameters ──
DEFAULT_BOARD_INNER = (7, 5)   # inner corners (cols, rows)
DEFAULT_SQUARE_MM = 30.0       # millimeters per square side

# Common phone camera horizontal FOV (degrees) by model family
PHONE_FOV_DB = {
    "iphone":   77.0,
    "samsung":  79.0,
    "pixel":    77.0,
    "default":  70.0,
}


def generate_checkerboard_image(board_inner=(7, 5), square_px=80,
                                 margin_px=60, dpi=150):
    """Generate a printable checkerboard image.

    Args:
        board_inner: (cols, rows) of inner corners → (cols+1, rows+1) squares.
        square_px: pixel size of each square in the output image.
        margin_px: white margin around the board.
        dpi: output resolution.

    Returns:
        PIL.Image of the checkerboard, and the save path.
    """
    cols, rows = board_inner[0] + 1, board_inner[1] + 1
    board_w = cols * square_px
    board_h = rows * square_px
    img_w = board_w + 2 * margin_px
    img_h = board_h + 2 * margin_px + 100  # extra for instructions

    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    # Draw squares
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0 = margin_px + c * square_px
                y0 = margin_px + r * square_px
                draw.rectangle([x0, y0, x0 + square_px, y0 + square_px], fill="black")

    # Add text instructions at bottom
    text_y = margin_px + board_h + 20
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_sm = font

    draw.text((margin_px, text_y),
              f"SkyWatch Calibration Board — {board_inner[0]}x{board_inner[1]} inner corners",
              fill="black", font=font)
    draw.text((margin_px, text_y + 22),
              f"Print at 100% scale. Each square = {DEFAULT_SQUARE_MM:.0f}mm. "
              f"Measure actual printed size and enter below.",
              fill="gray", font=font_sm)
    draw.text((margin_px, text_y + 40),
              "Place flat on ground. Photo from each camera must include the full board.",
              fill="gray", font=font_sm)

    return img


def estimate_focal_length_pixels(image_width, fov_degrees=70.0):
    """Estimate focal length in pixels from image width and horizontal FOV."""
    return (image_width / 2.0) / math.tan(math.radians(fov_degrees / 2.0))


def calibrate_from_image(image, board_inner=DEFAULT_BOARD_INNER,
                          square_size_mm=DEFAULT_SQUARE_MM,
                          camera_fov=70.0, camera_height_m=None):
    """Detect checkerboard and compute camera extrinsics.

    Args:
        image: BGR numpy array (photo from the phone).
        board_inner: (cols, rows) inner corners of the checkerboard.
        square_size_mm: physical size of each square in mm.
        camera_fov: estimated horizontal FOV of the phone camera in degrees.
        camera_height_m: if known, the height of the camera above ground.
            Used to improve the Z estimate when PnP is ambiguous.

    Returns:
        dict with:
            position: [x, y, z] in meters (world coords)
            yaw: degrees (0 = looking along +Y / north)
            pitch: degrees (positive = looking up)
            roll: degrees
            fov: estimated FOV used
            corners_found: bool
            preview_image: BGR image with detected corners drawn
        or None if checkerboard not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Try to find the checkerboard
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, board_inner, flags)

    if not ret:
        # Try with more aggressive search
        ret, corners = cv2.findChessboardCorners(
            gray, board_inner,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

    if not ret:
        return None

    # Sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 3D object points: checkerboard on Z=0 plane, centered at origin
    square_m = square_size_mm / 1000.0
    objp = np.zeros((board_inner[0] * board_inner[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_inner[0], 0:board_inner[1]].T.reshape(-1, 2)
    objp *= square_m
    # Center the board at origin
    objp[:, 0] -= (board_inner[0] - 1) * square_m / 2.0
    objp[:, 1] -= (board_inner[1] - 1) * square_m / 2.0

    # Camera intrinsics (estimated from FOV)
    fx = estimate_focal_length_pixels(w, camera_fov)
    fy = fx  # square pixels
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP — get camera pose relative to checkerboard
    success, rvec, tvec = cv2.solvePnP(
        objp, corners, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Camera position in world coordinates: C = -R^T @ t
    cam_pos_world = (-R.T @ tvec).flatten()

    # If camera height is known, override the Z component
    if camera_height_m is not None:
        cam_pos_world[2] = camera_height_m

    # The camera's forward direction in world coords is the 3rd row of R
    # (the Z-axis of the camera in world frame)
    # Actually: camera looks along +Z in camera space.
    # In world space, the forward direction is R^T @ [0, 0, 1]
    forward = R.T @ np.array([0, 0, 1], dtype=np.float64)

    # Yaw: angle from +Y axis in the XY plane
    yaw_rad = math.atan2(forward[0], forward[1])
    yaw_deg = math.degrees(yaw_rad)

    # Pitch: angle from horizontal
    horiz = math.sqrt(forward[0]**2 + forward[1]**2)
    pitch_rad = math.atan2(-forward[2], horiz)  # negative because looking down = negative pitch
    pitch_deg = math.degrees(pitch_rad)

    # Roll: rotation around the forward axis
    # Camera right direction in world space
    right = R.T @ np.array([1, 0, 0], dtype=np.float64)
    # Project right onto the horizontal plane perpendicular to forward
    up_world = np.array([0, 0, 1], dtype=np.float64)
    roll_rad = math.atan2(np.dot(right, up_world),
                           np.dot(np.cross(forward, up_world), right))
    roll_deg = math.degrees(roll_rad)

    # Draw detected corners on preview image
    preview = image.copy()
    cv2.drawChessboardCorners(preview, board_inner, corners, ret)
    # Draw axes
    axis_pts = np.float32([
        [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]
    ])
    img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts = img_pts.astype(int)
    origin = tuple(img_pts[3].ravel())
    cv2.line(preview, origin, tuple(img_pts[0].ravel()), (0, 0, 255), 3)  # X = red
    cv2.line(preview, origin, tuple(img_pts[1].ravel()), (0, 255, 0), 3)  # Y = green
    cv2.line(preview, origin, tuple(img_pts[2].ravel()), (255, 0, 0), 3)  # Z = blue

    return {
        "position": cam_pos_world.tolist(),
        "yaw": float(yaw_deg),
        "pitch": float(pitch_deg),
        "roll": float(roll_deg),
        "fov": float(camera_fov),
        "corners_found": True,
        "reprojection_error": _reprojection_error(objp, corners, rvec, tvec,
                                                   camera_matrix, dist_coeffs),
        "preview_image": preview,
    }


def _reprojection_error(objp, corners, rvec, tvec, camera_matrix, dist_coeffs):
    """Compute mean reprojection error in pixels."""
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(corners, projected, cv2.NORM_L2) / len(projected)
    return float(error)


def calibrate_multiple_cameras(images, camera_ids, board_inner=DEFAULT_BOARD_INNER,
                                square_size_mm=DEFAULT_SQUARE_MM,
                                camera_fov=70.0, camera_height_m=None):
    """Calibrate multiple cameras from checkerboard photos.

    Args:
        images: list of BGR images (one per camera).
        camera_ids: list of camera ID strings.
        board_inner: checkerboard inner corners.
        square_size_mm: physical square size.
        camera_fov: estimated FOV.
        camera_height_m: known camera height (or None).

    Returns:
        dict mapping camera_id → calibration result (or error string).
    """
    results = {}
    for img, cam_id in zip(images, camera_ids):
        result = calibrate_from_image(
            img, board_inner, square_size_mm, camera_fov, camera_height_m
        )
        if result is None:
            results[cam_id] = {"error": "Checkerboard not detected in image"}
        else:
            results[cam_id] = result
    return results
