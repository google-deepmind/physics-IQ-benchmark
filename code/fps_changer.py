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

import os
import cv2
import numpy as np
from PIL import Image

def change_video_fps(input_folder: str, output_folder: str, fps_new: float) -> None:
    """
    Changes the FPS of videos in a folder, processes only the first 5 seconds, 
    and saves them in the output folder.

    Args:
        input_folder: Path to the folder containing input videos.
        output_folder: Path to the folder where output videos will be saved.
        fps_new: New frames per second (FPS) value.
    """
    print(f"Starting FPS change process for target FPS: {fps_new}")
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
            fps_original = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps_original

            # Ensure even dimensions
            width, height = width - width % 2, height - height % 2

            subclip_duration = min(5, duration)  # Limit to the first 5 seconds
            print(f"Original FPS: {fps_original}, Duration: {duration}s, Frames: {frame_count}")

            frames = []
            for _ in range(int(subclip_duration * fps_original)):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            cap.release()

            frame_count_original = len(frames)
            frame_count_new = int(subclip_duration * fps_new)

            print(f"Original frames: {frame_count_original}, New frames: {frame_count_new}")

            # Interpolate frames to match the new frame rate
            frames_new = []
            for j in range(frame_count_new):
                alpha = j * (frame_count_original - 1) / (frame_count_new - 1)
                idx = int(alpha)
                alpha -= idx

                frame1 = frames[idx]
                frame2 = frames[min(idx + 1, frame_count_original - 1)]

                frame1_np = np.array(frame1).astype(np.float32)
                frame2_np = np.array(frame2).astype(np.float32)

                frame_interp_np = (1 - alpha) * frame1_np + alpha * frame2_np
                frame_interp = Image.fromarray(frame_interp_np.astype(np.uint8))

                frames_new.append(frame_interp)

            # Save the new video
            new_file_name = video_file.replace(f"{int(fps_original)}FPS", f"{int(fps_new)}FPS")
            new_file_path = os.path.join(output_folder, new_file_name)

            print(f"Saving video with dimensions: {width}x{height}, Codec: H.264")
            out = cv2.VideoWriter(
                new_file_path,
                cv2.VideoWriter_fourcc(*"avc1"),  # Use H.264 codec
                fps_new,
                (width, height)
            )

            for frame in frames_new:
                out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

            out.release()
            print(f"Saved new video to: {new_file_path}")

        except Exception as e:
            print(f"Error processing {video_file}: {e}")
