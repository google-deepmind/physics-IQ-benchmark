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
import multiprocessing
import os
import subprocess

from physiq.fps_changer import change_video_fps

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
        subprocess.run(
            ["gcloud", "storage", "rsync", "--recursive", remote_path, local_path],
            check=True,
        )
        print(f"Sync complete for {remote_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to sync: {remote_path}. Error: {e}")
        raise


def download_physics_iq_data(
    fps: list[int] | tuple[int,], base_dir: str, verified: bool = True
):
    """Download the Physics-IQ dataset based on the specified FPS.

    Args:
      fps: Desired FPS as an integer in [1, 30]. 8, 16, 24, and 30 FPS are
        available pre-computed; any other value in range is downsampled
        locally from the 30 FPS data.
    """

    # Always download 30FPS data
    download_fps = [30]
    # Additionally download pre-computed non-30 FPS data if available
    for f in fps:
        assert 1 <= f <= 30, f"FPS must be in [1, 30], got {f}"
        if f in download_fps:
            continue
        download_fps.append(f)

    if not verified:
        local_base_dir = os.path.join(base_dir, "physics-IQ-benchmark")
        base_url = "gs://physics-iq-benchmark"
    else:
        local_base_dir = os.path.join(base_dir, "physics-IQ-benchmark-verified")
        raise NotImplementedError(
            "Currently there is no download url available for the verified dataset.\nPlease Download from the Link provided in the README.md"
        )
        # TODO: add download link for physics iq Verified dataset here
        base_url = "gs://physics-iq-benchmark"

    directories = {
        "full-videos/take-1": download_fps,
        "full-videos/take-2": download_fps,
        "split-videos/conditioning": download_fps,
        "split-videos/testing": download_fps,
        "switch-frames": None,
        "video-masks/real": download_fps,
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
    for f in fps:
        if f not in (8, 16, 24, 30):
            print(f"Downsampling 30 FPS videos to {f} FPS locally...")
            for directory, subdirs in directories.items():
                if subdirs is None or directory == "video-masks/real":
                    continue
                input_folder = os.path.join(local_base_dir, directory, "30FPS")
                output_folder = os.path.join(local_base_dir, directory, f"{f}FPS")
                change_video_fps(
                    input_folder=input_folder, output_folder=output_folder, fps_new=f
                )

    print("Download process complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Downloads Physics IQ Dataset or Physiqs IQ verified Dataset."
    )
    parser.add_argument(
        "--fps",
        nargs="+",
        default=[30],
        type=int,
        help="Select values in (8, 16, 24, 30) for download otherwise resample",
    )
    parser.add_argument(
        "--original_physics_iq",
        action="store_true",
        help="Whether to download verified (default) or orginal dataset",
    )
    parser.add_argument(
        "--benchmark_base_folder",
        type=str,
        default=".",
        help="Path to the folder within which physics-IQ-benchmark & physics-IQ-benchmark-verifed data are or will be located after download.",
    )
    args = parser.parse_args()
    download_physics_iq_data(
        args.fps, args.benchmark_base_folder, not args.original_physics_iq
    )


if __name__ == "__main__":
    main()
