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
import shutil

multiprocessing.set_start_method("spawn", force=True)

def require_executable(name: str, install_hint: str):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Required executable '{name}' not found in PATH.\n"
            f"Install hint: {install_hint}"
        )
    return path


def download_directory(remote_path: str, local_path: str):
    """Sync a remote directory with a local directory using gsutil rsync.

    Args:
      remote_path: Cloud path.
      local_path: Local path.
    """
    print(f"Syncing {remote_path} → {local_path} using gsutil rsync...")
    os.makedirs(local_path, exist_ok=True) 
    gsutil_path = require_executable(
    "gsutil",
    "Install via `pip install gsutil` or use system gsutil"
    )
    try:
        #
        subprocess.run(["gsutil_path", "-m", "rsync", "-r", remote_path, local_path], check=True)
        print(f"Sync complete for {remote_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to sync: {remote_path}. Error: {e}")
        raise


def download_physics_iq_data(fps: str):
    """Download the Physics-IQ dataset based on the specified FPS.

    Args:
      fps: Desired FPS (in ['8', '16', '24', '30', 'other']).
    """
    valid_fps = ['8', '16', '24', '30', 'other']
    assert fps in valid_fps, 'FPS needs to be in [8, 16, 24, 30, other]'

    # Always download 30FPS data
    download_fps = ['30']

    # Additionally download available non-30 FPS data if necessary
    if fps in ['8', '16', '24']:
        download_fps.append(fps)
        
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

    print("Download process complete.")


if __name__ == '__main__':
    user_fps = input("Enter your model's frames per second FPS (e.g., 8, 16, 24, 30, other): ").strip()
    download_physics_iq_data(user_fps)
