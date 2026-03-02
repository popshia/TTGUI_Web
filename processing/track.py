"""
Object Tracking using Ultralytics YOLOv8 built-in tracker.

Tracks detected objects across frames, assigns persistent IDs,
and draws bounding boxes with track IDs on each frame.
"""

import cv2
from ultralytics import YOLO


def track_objects(input_path: str, output_path: str, model_name: str = 'yolov8n.pt', conf: float = 0.35):
    """
    Run object tracking on a video and write annotated output with track IDs.

    Args:
        input_path: Path to the input video file.
        output_path: Path to write the tracked video.
        model_name: YOLOv8 model to use (auto-downloads if needed).
        conf: Confidence threshold for detections.
    """
    model = YOLO(model_name)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Store trail history: { track_id: [(cx, cy), ...] }
    trail_history: dict[int, list[tuple[int, int]]] = {}
    trail_max_len = 50

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking
        results = model.track(frame, conf=conf, persist=True, verbose=False)
        annotated = results[0].plot()

        # Draw trails
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                cx, cy = int(box[0]), int(box[1])

                if track_id not in trail_history:
                    trail_history[track_id] = []
                trail = trail_history[track_id]
                trail.append((cx, cy))
                if len(trail) > trail_max_len:
                    trail.pop(0)

                # Draw trail as fading polyline
                for j in range(1, len(trail)):
                    alpha = j / len(trail)
                    color = (
                        int(139 * alpha),   # blue channel
                        int(92 * alpha),    # green channel
                        int(246 * alpha),   # red channel  → purple-ish
                    )
                    thickness = max(1, int(3 * alpha))
                    cv2.line(annotated, trail[j - 1], trail[j], color, thickness)

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[TRACK] Processed {frame_idx} frames → {output_path}")
