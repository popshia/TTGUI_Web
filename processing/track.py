import argparse

import cv2
from ultralytics import YOLO


def detect_and_track(
    input_video_path, output_video_path, model_path, show=False, plot_track=False
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

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO26 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True, tracker="./bytetrack.yaml")[0]

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
                cv2.imshow("YOLO30 Tracking", frame)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("model")
    args = parser.parse_args()

    detect_and_track(args.input_file, args.output_file, args.model)
