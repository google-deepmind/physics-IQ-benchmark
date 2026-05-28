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


def trim_videos(input_folder: str, output_folder: str) -> None:
    """
    Trims all MP4 videos in input_folder to 5 seconds and saves them to output_folder.

    Args:
        input_folder: Path to the folder containing input videos.
        output_folder: Path to the folder where trimmed videos will be saved.
    """
    print(f"Trimming videos in {input_folder} to 5 seconds.")
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith(".mp4")]

    if not video_files:
        print(f"No MP4 files found in {input_folder}. Exiting.")
        return

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing: {video_file}")

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Ensure even dimensions (required by H.264 codec)
            width, height = width - width % 2, height - height % 2

            max_frames = int(5 * fps)
            output_path = os.path.join(output_folder, video_file)

            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"avc1"),
                fps,
                (width, height),
            )

            count = 0
            while count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                count += 1

            cap.release()
            out.release()
            print(
                f"Saved trimmed video ({count} frames at {fps} FPS) to: {output_path}"
            )

        except Exception as e:
            print(f"Error processing {video_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim videos to 5 seconds.")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing input videos.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder where trimmed videos will be saved.",
    )

    args = parser.parse_args()
    trim_videos(args.input_folder, args.output_folder)
