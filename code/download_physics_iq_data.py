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

# Ensure 'spawn' is the default start method for multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def download_directory(remote_path: str, local_path: str):
    """Download a single directory using gsutil.

    Args:
      remote_path: Cloud path
      local_path: Local path
      dry_run: If True, only prints the commands without executing them.
    """
    os.makedirs(local_path, exist_ok=True)
    print(f"Preparing to download: {remote_path} to {local_path}")

    try:
        # Limit parallel downloads to avoid freezing
        subprocess.run(
            ["gsutil", "-m", "-o", "GSUtil:parallel_process_count=5", "cp", "-r", remote_path + "/*", local_path], check=True
        )
        print(f"Downloaded: {remote_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download: {remote_path} Error: {e}")
        raise


def download_physics_iq_data(fps: str):
    """Download the Physics-IQ dataset based on the specified FPS.

    Args:
      fps: Desired FPS (in ['8', '16', '24', '30']).
    """
    valid_fps = ['8', '16', '24', '30']
    assert fps in valid_fps, 'FPS needs to be in [8, 16, 24, 30]'

    download_fps = [fps]
    if fps != '30':
        download_fps.append('30')  # Always download 30FPS data

    base_url = "gs://physics-iq-benchmark"  # Base GCS URL
    local_base_dir = "./physics-IQ-benchmark"  # Local base directory

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

    print("Download process complete.")


if __name__ == '__main__':
    user_fps = input("Enter your model's frames per second FPS (e.g., 8, 16, 24, 30): ").strip()
    download_physics_iq_data(user_fps)
