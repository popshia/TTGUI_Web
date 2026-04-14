import argparse
import csv
from collections import defaultdict

import cv2
import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO


def track_and_ouput_csv(
    input_video_path,
    output_video_path,
    model_path,
    output_csv_path=None,
    show=False,
    plot_track=False,
):
    # Load the YOLO26 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set up the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Store information for CSV export
    track_info = {}
    frame_index = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            frame_index += 1
            # Run YOLO26 tracking on the frame, persisting tracks between frames
            result = model.track(
                frame,
                persist=True,
                tracker="./botsort.yaml",
                device="mps" if torch.backends.mps.is_available() else "0",
            )[0]
            obb = result.obb if result.obb is not None else None

            # Collect tracking data for CSV export
            if obb is not None and getattr(obb, "id", None) is not None:
                track_ids = obb.id.int().cpu().tolist()
                class_map = {
                    0: "c",
                    1: "t",
                    2: "b",
                    3: "h",
                    4: "g",
                    5: "p",
                    6: "u",
                    7: "m",
                }
                cls_indices = [
                    class_map.get(c, str(c)) for c in obb.cls.int().cpu().tolist()
                ]

                # Get 4 corner coordinates (OBB or standard BBox)
                corners = obb.xyxyxyxy.cpu().numpy().astype(int)  # Shape (N, 4, 2)

                for t_id, corner, cls_idx in zip(track_ids, corners, cls_indices):
                    if t_id not in track_info:
                        track_info[t_id] = {
                            "enter_frame": frame_index,
                            "exit_frame": frame_index,
                            "cls_idx": cls_idx,
                            "coords": {},
                        }
                    track_info[t_id]["exit_frame"] = frame_index
                    track_info[t_id]["coords"][frame_index] = corner.flatten().tolist()

            # Visualize the result on the frame unconditionally
            frame = result.plot(line_width=2, font_size=2, conf=False)

            # Optional: Get the boxes and track IDs for custom track plotting
            if plot_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=10,
                    )

            # Display the annotated frame
            if show:
                cv2.imshow("YOLO26 Tracking", frame)

            # Write the annotated frame to the output video
            out.write(frame)

            # Break the loop if 'q' is pressed
            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture and writer objects and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if output_csv_path:
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            for obj_id, info in track_info.items():
                row = [
                    obj_id,
                    info["enter_frame"],
                    info["exit_frame"],
                    info["cls_idx"],
                    "X",
                    "X",
                ]
                for frame_num in sorted(info["coords"].keys()):
                    row.extend(info["coords"][frame_num])
                writer.writerow(row)

    logger.info(
        f"[TRACKING] Saved tracked video to {output_video_path}, raw csv to {output_csv_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("model")
    parser.add_argument("csv")
    args = parser.parse_args()

    track_and_ouput_csv(
        args.input_file, args.output_file, args.model, output_csv_path=args.csv
    )
