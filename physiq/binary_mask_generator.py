# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse
import os

import cv2
import numpy as np


def generate_mask(
    in_path: str, out_video_path: str, is_real: bool, threshold_value: int = 10
) -> None:
    """
    Processes a video to generate a binary mask for ROI detection.

    Args:
        in_path: Path to the input video file.
        out_video_path: Path to save the generated binary mask video.
        is_real: Boolean indicating whether the video is real.
        threshold_value: Threshold value for binary segmentation.
    """

    try:
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {in_path}")
            return

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure dimensions are even
        width, height = width - width % 2, height - height % 2

        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        name_parts = out_video_path.split("/")
        name_parts_last = name_parts[-1].split("_")

        # Account for name difference between real and generated videos and save the mask accordingly
        if is_real:
            replace_last_part = (
                name_parts_last[0]
                + "_video-masks_"
                + f"{int(fps)}FPS_"
                + name_parts_last[3]
                + "_"
                + name_parts_last[4]
                + "_"
                + name_parts_last[5]
            )
        else:
            replace_last_part = (
                name_parts_last[0]
                + "_video-masks_"
                + f"{int(fps)}FPS_"
                + name_parts_last[1]
                + "_take-1_"
                + name_parts_last[2]
            )

        name_parts[-1] = replace_last_part
        out_video_path = "/".join(name_parts)

        out = cv2.VideoWriter(
            out_video_path, fourcc, fps, (width, height), isColor=False
        )

        # Read and initialize background model with first real frame
        ret, prev_frame = cap.read()
        if not ret:
            print(f"Error: Could not read the first frame from {in_path}")
            return

        # Process the first frame immediately
        gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        avg_frame = gray_frame.astype("float")

        # Write the first frame's mask
        first_mask = np.zeros_like(gray_frame, dtype=np.uint8)
        out.write(first_mask)
        generated_frame_count = 1  # Account for the first written frame

        kernel = np.ones((5, 5), np.uint8)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

            cv2.accumulateWeighted(gray_frame, avg_frame, 0.3)
            avg_gray_frame = cv2.convertScaleAbs(avg_frame)

            frame_diff = cv2.absdiff(gray_frame, avg_gray_frame)

            _, binary_frame = cv2.threshold(
                frame_diff, threshold_value, 255, cv2.THRESH_BINARY
            )

            binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
            binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
            out.write(binary_frame)
            generated_frame_count += 1

        cap.release()
        out.release()
        print(f"Processed video saved at {out_video_path}")
        print(f"Input video frame count: {input_frame_count}")
        print(f"Number of generated mask frames: {generated_frame_count}")

    except cv2.error as e:
        print(f"Error processing video {in_path}: {e}")


def generate_binary_masks(
    input_parent_directory: str,
    output_parent_directory: str,
    is_real: bool,
    threshold_value: int = 10,
) -> None:
    """
    Generates binary masks for all videos in the input directory.

    Args:
        input_parent_directory: Path to the parent directory containing input videos.
        output_parent_directory: Path to the parent directory for saving output videos.
        is_real: Boolean indicating whether the video is real.
        threshold_value: Threshold value for binary segmentation.
    """

    if not os.path.exists(output_parent_directory):
        os.makedirs(output_parent_directory)

    for root, _, files in os.walk(input_parent_directory):
        relative_path = os.path.relpath(root, input_parent_directory)
        output_directory = os.path.join(output_parent_directory, relative_path)
        os.makedirs(output_directory, exist_ok=True)

        for filename in files:
            if filename.endswith(".mp4"):
                input_path = os.path.join(root, filename)
                output_video_path = os.path.join(output_directory, filename)
                print(f"Generating mask for: {filename}")
                generate_mask(input_path, output_video_path, is_real, threshold_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos for binary segmentation."
    )
    parser.add_argument(
        "--input_parent_directory",
        type=str,
        required=True,
        help="Path to the parent directory containing input videos.",
    )
    parser.add_argument(
        "--output_parent_directory",
        type=str,
        required=True,
        help="Path to the parent directory for saving output videos.",
    )
    parser.add_argument(
        "--threshold_value",
        type=int,
        default=10,
        help="Threshold value for binary segmentation.",
    )
    args = parser.parse_args()

    generate_binary_masks(
        args.input_parent_directory,
        args.output_parent_directory,
        True,
        args.threshold_value,
    )
