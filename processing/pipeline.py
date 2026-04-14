"""
Processing Pipeline – orchestrates stabilization → detection → tracking.
"""

import os
from typing import Callable, Optional

import config
from loguru import logger

from processing.csv_postprocess import process_trajectory_file
from processing.stabilize import stabilize_video
from processing.track import track_and_ouput_csv


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

    def log(stage: str, pct: int = 0):
        if on_progress:
            on_progress(stage, pct)
        logger.info(f"[PIPELINE] {job_id} | {stage} ({pct}%)")

    ext = os.path.splitext(input_path)[1] or ".mp4"

    # ── Stage 1: Video Stabilization ──
    stabilized_path = os.path.join(
        output_dir, f"{input_path.split('/')[-1].split('.')[0]}_stabilized{ext}"
    )
    log("stabilizing", 0)
    stabilize_video(input_path, stabilized_path, "ecc", (1920, 1080))  # lk, ecc
    log("stabilizing", 100)

    # ── Stage 2: Object Detect & Tracking ──
    tracked_path = os.path.join(
        output_dir, f"{input_path.split('/')[-1].split('.')[0]}_detected{ext}"
    )
    raw_csv = os.path.join(output_dir, "raw.csv")
    log("dracking object and output csv", 0)
    track_and_ouput_csv(
        stabilized_path,
        tracked_path,
        config.MODEL_PATH,
        os.path.join(output_dir, raw_csv),
    )
    log("dracking object and output csv", 100)

    # ── Stage 3: CSV file fixing ──
    processed_csv = os.path.join(output_dir, "processed.csv")
    log("fixing csv file", 0)
    process_trajectory_file(raw_csv, processed_csv)
    log("fixing csv file", 100)

    # Clean up intermediate files (keep only the final output)
    # for intermediate in [stabilized_path, raw_csv]:
    for intermediate in [stabilized_path]:
        try:
            os.remove(intermediate)
        except OSError:
            pass

    return tracked_path
