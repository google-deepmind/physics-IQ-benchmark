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


def list_remote_files(remote_path: str):
    """Lists files in the remote GCS path.

    Args:
      remote_path: Cloud path.

    Returns:
      A set of file paths relative to remote_path.
    """
    try:
        result = subprocess.run(
            ["gsutil", "ls", "-r", remote_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        remote_files = result.stdout.strip().split("\n")

        # Remove the base path to get relative paths
        remote_relative_paths = {
            file.replace(remote_path + "/", "").strip()
            for file in remote_files
            if file.startswith(remote_path) and "." in file  # Ensure it's a file, not a folder
        }
        return remote_relative_paths
    except subprocess.CalledProcessError as e:
        print(f"Error listing remote files for {remote_path}: {e.stderr}")
        return set()


def list_local_files(local_path: str):
    """Lists files in the local directory with relative paths.

    Args:
      local_path: Local directory path.

    Returns:
      A set of file paths relative to local_path.
    """
    local_files = set()
    for root, _, files in os.walk(local_path):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, local_path)  # Preserve directory structure
            local_files.add(relative_path)
    return local_files


def download_directory(remote_path: str, local_path: str):
    """Download missing files from the remote directory using gsutil.

    Args:
      remote_path: Cloud path.
      local_path: Local path.
    """
    if not os.path.exists(local_path):
        print(f"Directory {local_path} does not exist. Creating it and downloading everything using gsutil -m rsync -r...")
        os.makedirs(local_path, exist_ok=True)  # Ensure the directory exists

        try:
            subprocess.run(
                ["gsutil", "-m", "rsync", "-r", remote_path, local_path],  # Use rsync instead of cp -r
                check=True,
            )
            print(f"Downloaded full directory: {remote_path} → {local_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download: {remote_path} Error: {e}")
            raise
        return

    print(f"Checking missing files in {remote_path}...")

    remote_files = list_remote_files(remote_path)
    local_files = list_local_files(local_path)

    # Ensure we only check for missing files, avoiding directory paths
    missing_files = {file for file in remote_files if file and file not in local_files and "." in file}

    if not missing_files:
        print(f"All files already exist in {local_path}, skipping download.")
        return

    print(f"Downloading {len(missing_files)} missing files to {local_path}...")

    try:
        for missing_file in missing_files:
            remote_file_path = f"{remote_path}/{missing_file}"
            local_file_path = os.path.join(local_path, missing_file)

            # Ensure directory structure exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            print(f"Downloading {remote_file_path} → {local_file_path}")
            subprocess.run(
                ["gsutil", "-m", "cp", remote_file_path, local_file_path],
                check=True,
            )

        print(f"Downloaded missing files from {remote_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download: {remote_path} Error: {e}")
        raise


def download_physics_iq_data(fps: str):
    """Download the Physics-IQ dataset based on the specified FPS.

    Args:
      fps: Desired FPS (in ['8', '16', '24', '30', 'other']).
    """
    valid_fps = ['8', '16', '24', '30', 'other']
    assert fps in valid_fps, 'FPS needs to be in [8, 16, 24, 30, other]'

    if fps == 'other':
        download_fps = ['30']
    else:
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
    user_fps = input("Enter your model's frames per second FPS (e.g., 8, 16, 24, 30, other): ").strip()
    download_physics_iq_data(user_fps)


