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
import subprocess
import multiprocessing

from fps_changer import change_video_fps

multiprocessing.set_start_method("spawn", force=True)


def download_directory(remote_path: str, local_path: str):
    """Sync a remote directory with a local directory using gcloud storage rsync.

    Args:
      remote_path: Cloud path.
      local_path: Local path.
    """
    print(f"Syncing {remote_path} → {local_path} using gcloud storage rsync...")
    os.makedirs(local_path, exist_ok=True) 
    try:
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", remote_path, local_path], check=True)
        print(f"Sync complete for {remote_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to sync: {remote_path}. Error: {e}")
        raise


def download_physics_iq_data(fps: str):
    """Download the Physics-IQ dataset based on the specified FPS.

    Args:
      fps: Desired FPS as an integer in [1, 30]. 8, 16, 24, and 30 FPS are
        available pre-computed; any other value in range is downsampled
        locally from the 30 FPS data.
    """
    try:
        fps_int = int(fps)
    except (TypeError, ValueError):
        raise ValueError(f'FPS must be an integer in [1, 30], got {fps!r}')
    assert 1 <= fps_int <= 30, f'FPS must be in [1, 30], got {fps_int}'

    # Always download 30FPS data
    download_fps = ['30']

    # Additionally download pre-computed non-30 FPS data if available
    if fps_int in (8, 16, 24):
        download_fps.append(str(fps_int))
        
    base_url = "gs://physics-iq-benchmark" 
    local_base_dir = "./physics-IQ-benchmark"  

    directories = {
        "full-videos/take-1": download_fps,
        "full-videos/take-2": download_fps,
        "split-videos/conditioning": download_fps,
        "split-videos/testing": download_fps,
        "switch-frames": None,
        "video-masks/real": download_fps
    }

    for directory, subdirs in directories.items():
        if subdirs:
            for fps_option in subdirs:
                remote_path = f"{base_url}/{directory}/{fps_option}FPS"
                local_path = os.path.join(local_base_dir, directory, f"{fps_option}FPS")
                download_directory(remote_path=remote_path, local_path=local_path)
        else:
            remote_path = f"{base_url}/{directory}"
            local_path = os.path.join(local_base_dir, directory)
            download_directory(remote_path=remote_path, local_path=local_path)

    # For FPS values without a pre-computed version, downsample locally from 30 FPS.
    # Skip video-masks/real: linear frame interpolation would produce non-binary
    # pixels; run_physics_iq.py::ensure_binary_mask_structure will regenerate
    # proper binary masks from the downsampled real videos at benchmark time.
    if fps_int not in (8, 16, 24, 30):
        print(f"Downsampling 30 FPS videos to {fps_int} FPS locally...")
        for directory, subdirs in directories.items():
            if subdirs is None or directory == "video-masks/real":
                continue
            input_folder = os.path.join(local_base_dir, directory, "30FPS")
            output_folder = os.path.join(local_base_dir, directory, f"{fps_int}FPS")
            change_video_fps(input_folder=input_folder, output_folder=output_folder, fps_new=fps_int)

    print("Download process complete.")


if __name__ == '__main__':
    user_fps = input("Enter your model's frames per second FPS as an integer in [1, 30]: ").strip()
    download_physics_iq_data(user_fps)
