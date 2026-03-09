import cv2
import numpy as np
from loguru import logger


def lk_stabilize(
    input_path: str, output_path: str, output_size, smoothing_radius: int = 30
):
    """
    Stabilize a video using a 2-pass Lucas-Kanade optical flow method.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # --- PASS 1: Compute frame-to-frame transforms ---
    transforms = []

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame")

    # Resize first frame to target resolution
    prev_frame = cv2.resize(prev_frame, output_size, interpolation=cv2.INTER_CUBIC)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, n_frames):
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_frame = cv2.resize(curr_frame, output_size, interpolation=cv2.INTER_CUBIC)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Detect features to track
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )

        if prev_pts is not None and len(prev_pts) > 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None
            )

            idx = np.where(status.flatten() == 1)[0]
            prev_pts_good = prev_pts[idx]
            curr_pts_good = curr_pts[idx]

            if len(prev_pts_good) >= 3:
                m, _ = cv2.estimateAffinePartial2D(prev_pts_good, curr_pts_good)
                if m is not None:
                    dx = m[0, 2]
                    dy = m[1, 2]
                    da = np.arctan2(m[1, 0], m[0, 0])
                    transforms.append((dx, dy, da))
                else:
                    transforms.append((0, 0, 0))
            else:
                transforms.append((0, 0, 0))
        else:
            transforms.append((0, 0, 0))

        prev_gray = curr_gray
        logger.info(f"LK: Analyzing {i}/{n_frames} frames...")

    # --- CALCULATE SMOOTH TRAJECTORY ---
    transforms = np.array(transforms)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = _smooth(trajectory, smoothing_radius)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # --- PASS 2: Apply stabilized transforms ---
    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_CUBIC)
        out.write(frame)

    for i in range(len(transforms_smooth)):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_CUBIC)

        dx, dy, da = transforms_smooth[i]
        m = np.array(
            [[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]],
            dtype=np.float64,
        )

        stabilized = cv2.warpAffine(frame, m, output_size, flags=cv2.INTER_CUBIC)
        out.write(stabilized)
        logger.info(f"LK: Rendered {i + 1}/{len(transforms_smooth)} frames...")

    cap.release()
    out.release()


def _smooth(trajectory: np.ndarray, radius: int) -> np.ndarray:
    """Applies a simple moving average filter to the trajectory."""
    smoothed = np.copy(trajectory)
    for i in range(trajectory.shape[1]):
        # Create a moving average kernel
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        # Pad the edges so we don't lose data at the beginning/end of the video
        padded = np.pad(trajectory[:, i], radius, mode="edge")
        smoothed[:, i] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def ecc_stabilize(input_path: str, output_path: str, output_size):
    """
    Stabilize video using ECC with 'Warm Start' (persistent warp matrix)
    and Homography mapping.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    # 1. Initialize Target Template
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame.")

    # Resize and Grayscale for the ECC algorithm
    first_resized = cv2.resize(first_frame, output_size, interpolation=cv2.INTER_CUBIC)
    target_gray = cv2.cvtColor(first_resized, cv2.COLOR_BGR2GRAY)

    # Write the first frame as-is
    out.write(first_resized)

    # 2. ECC Configuration
    warp_mode = cv2.MOTION_HOMOGRAPHY
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 5 * 1e-4)

    # Persistent Warp Matrix (The "Warm Start")
    # Instead of resetting this every loop, we evolve it.
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_resized = cv2.resize(frame, output_size, interpolation=cv2.INTER_CUBIC)
        curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)

        try:
            # The Warm Start: We pass the 'warp_matrix' from the PREVIOUS frame
            # as the initial guess for the CURRENT frame.
            _, warp_matrix = cv2.findTransformECC(
                target_gray, curr_gray, warp_matrix, warp_mode, criteria
            )

            # Apply the calculated perspective transformation
            # WARP_INVERSE_MAP is used because ECC calculates the mapping
            # from template to input, but we want to pull input back to template.
            stabilized_frame = cv2.warpPerspective(
                curr_resized,
                warp_matrix,
                output_size,
                flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            )
            out.write(stabilized_frame)

        except cv2.error as e:
            logger.warning(
                f"Frame {frame_idx}: ECC failed to converge. Using previous matrix."
            )
            # If it fails, we apply the last successful matrix to maintain continuity
            fallback_frame = cv2.warpPerspective(
                curr_resized,
                warp_matrix,
                output_size,
                flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            )
            out.write(fallback_frame)

        frame_idx += 1
        if frame_idx % 10 == 0:
            logger.info(f"ECC: Stabilized {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()


def stabilize_video(input_path: str, output_path: str, method: str, output_size):
    match method:
        case "ecc":
            ecc_stabilize(input_path, output_path, output_size)
        case "lk":
            lk_stabilize(input_path, output_path, output_size)

    logger.info(f"[STABILIZE] Output saved to {output_path}")


if __name__ == "__main__":
    stabilize_video(
        "/Users/noah-mac-nb/Developer/TTGUI_Web/test/test_trimmed.mp4",
        "/Users/noah-mac-nb/Developer/TTGUI_Web/test/test_trimmed_stabilized.mp4",
        "ecc",
        (1920, 1080),
    )
