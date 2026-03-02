"""
Video Stabilization using OpenCV.

Uses feature-point-based stabilization:
1. Detect good features (Shi-Tomasi corners) in each frame
2. Track features across frames with optical flow (Lucas-Kanade)
3. Estimate affine transforms between consecutive frames
4. Smooth the trajectory using a moving average
5. Apply corrected transforms to stabilize the video
"""

import cv2
import numpy as np
import os


def stabilize_video(input_path: str, output_path: str, smoothing_radius: int = 30):
    """
    Stabilize a video file and write the result to output_path.

    Args:
        input_path: Path to the input video file.
        output_path: Path to write the stabilized video.
        smoothing_radius: Number of frames to average for trajectory smoothing.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Pass 1: Compute frame-to-frame transforms ---
    transforms = []
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, n_frames):
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )

        if prev_pts is not None and len(prev_pts) > 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            assert curr_pts is not None

            # Filter valid points
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

    cap.release()

    # --- Compute trajectory and smooth it ---
    transforms = np.array(transforms)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = _smooth(trajectory, smoothing_radius)
    difference = smoothed_trajectory - trajectory

    # Apply corrections to transforms
    transforms_smooth = transforms + difference

    # --- Pass 2: Apply stabilized transforms ---
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    ret, frame = cap.read()
    if ret:
        out.write(frame)  # Write first frame as-is

    for i in range(len(transforms_smooth)):
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy, da = transforms_smooth[i]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da),  np.cos(da), dy],
        ], dtype=np.float64)

        stabilized = cv2.warpAffine(frame, m, (w, h))
        out.write(stabilized)

    cap.release()
    out.release()
    print(f"[STABILIZE] Output saved to {output_path}")


def _smooth(trajectory: np.ndarray, radius: int) -> np.ndarray:
    """Moving average filter on each column of the trajectory."""
    smoothed = np.copy(trajectory)
    for i in range(trajectory.shape[1]):
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        padded = np.pad(trajectory[:, i], radius, mode='edge')
        smoothed[:, i] = np.convolve(padded, kernel, mode='valid')
    return smoothed
