"""
Processing Pipeline – orchestrates stabilization → detection → tracking.
"""

import os
from typing import Callable, Optional

from processing.stabilize import stabilize_video
from processing.detect import detect_objects
from processing.track import track_objects


def run_pipeline(
    input_path: str,
    output_dir: str,
    job_id: str,
    on_progress: Optional[Callable[[str, int], None]] = None,
) -> str:
    """
    Run the full 3-stage video processing pipeline.

    Args:
        input_path: Path to the uploaded video.
        output_dir: Directory to store intermediate and final output files.
        job_id: Unique job identifier.
        on_progress: Optional callback(stage_name, percent) for progress updates.

    Returns:
        Path to the final processed video.
    """
    def update(stage: str, pct: int = 0):
        if on_progress:
            on_progress(stage, pct)
        print(f"[PIPELINE] {job_id} | {stage} ({pct}%)")

    ext = os.path.splitext(input_path)[1] or '.mp4'

    # ── Stage 1: Video Stabilization ──
    stabilized_path = os.path.join(output_dir, f"{input_path.split("/")[-1].split(".")[0]}_stabilized{ext}")
    update('stabilizing', 0)
    stabilize_video(input_path, stabilized_path)
    update('stabilizing', 100)

    # ── Stage 2: Object Detection ──
    detected_path = os.path.join(output_dir, f"{input_path.split("/")[-1].split(".")[0]}_detected{ext}")
    update('detecting', 0)
    detect_objects(stabilized_path, detected_path)
    update('detecting', 100)

    # ── Stage 3: Object Tracking ──
    final_path = os.path.join(output_dir, f"{input_path.split("/")[-1].split(".")[0]}_tracked{ext}")
    update('tracking', 0)
    track_objects(detected_path, final_path)
    update('tracking', 100)

    # Clean up intermediate files (keep only the final output)
    for intermediate in [stabilized_path, detected_path]:
        try:
            os.remove(intermediate)
        except OSError:
            pass

    return final_path
