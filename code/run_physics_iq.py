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
import sys
import pandas as pd
import cv2
import argparse
from fps_changer import change_video_fps
from calculate_and_write_metrics_to_csv import process_videos
from calculate_iq_score import process_directory
from binary_mask_generator import generate_binary_masks


def is_csv_complete(csv_file_path: str, expected_scenarios: set[str]) -> bool:
    """
    Checks if the CSV file exists and contains all required scenarios.

    Args:
        csv_file_path: Path to the CSV file.
        expected_scenarios: Set of expected scenario names with suffixes.

    Returns:
        bool: True if the CSV file is complete, False otherwise.
    """
    if not os.path.exists(csv_file_path):
        return False

    # Load the calculated CSV file
    df = pd.read_csv(csv_file_path)

    # Extract base scenario names from descriptions (strip `perspective-center_take-1`, `perspective-left_take-1`)
    base_expected_scenarios = {scenario.split("_")[-1] for scenario in expected_scenarios}
    # Extract scenario names from the calculated CSV file
    csv_scenarios = set(df["scenario"])

    # Check for missing scenarios
    missing_scenarios = base_expected_scenarios - csv_scenarios
    if missing_scenarios:
        print(f"CSV file {csv_file_path} is incomplete. Missing scenarios: {missing_scenarios}")
        return False

    print(f"CSV file {csv_file_path} is complete with all scenarios.")
    return True



def validate_directory_exists(directory: str, description: str) -> None:
  """
  Validates that a directory exists.

  Args:
    directory: Path to the directory.
    description: A description of the directory for error messages.

  Raises:
    FileNotFoundError: If the directory does not exist.
  """
  if not os.path.exists(directory):
    raise FileNotFoundError(f"{description} not found: {directory}")

def validate_folder_files_exist(
    folder: str, expected_files: set[str], description: str
) -> None:
  """
  Validates that a folder exists and contains all expected files.

  Args:
    folder: Path to the folder.
    expected_files: A set of filenames expected in the folder.
    description: Description of the folder for error messages.

  Raises:
    FileNotFoundError: If the folder is missing or does not contain all files.
  """
  if not os.path.exists(folder):
    raise FileNotFoundError(f"{description} folder does not exist: {folder}")

  actual_files = {f for f in sorted(os.listdir(folder)) if f.endswith(".mp4")}
  fps = int(description.split(" ")[-1])
  expected_files = {f.replace('30FPS', f'{fps}FPS') for f in actual_files}
  missing_files = expected_files - actual_files
  if missing_files:
    raise FileNotFoundError(
      f"{description} folder is missing files: {missing_files}"
    )

  print(f"{description} folder is valid with all required files.")

def validate_video_files(
    input_folder: str, descriptions_file: str, video_type: str, fps: int, include_take_2: bool = False
) -> set[str]:
  """
  Validates that the MP4 files in the input folder match the scenarios
  in the descriptions file.

  Args:
    input_folder: Path to the folder containing input videos.
    descriptions_file: Path to the descriptions CSV file.
    include_take_2: If True, include `take-2` filenames in the expected set.

  Returns:
    A set of expected filenames.

  Raises:
    FileNotFoundError: If required videos are missing.
  """
  if video_type not in ["video-masks", "testing-videos", ""]:
    raise ValueError("video type is neither video-mask nor testing-video")
  descriptions = pd.read_csv(descriptions_file)
  expected_files = set(descriptions["scenario"])  # Includes ".mp4"
  def modify_file_name(file_name):
    file_id, view, take, scenario = file_name.split("_")
    if video_type != "":
      new_name = f"{file_id}_{video_type}_{fps}FPS_{view}_{take}_{scenario}"
    else:
      new_name = f"{file_id}_{view}_{scenario}"
    return new_name

  if not include_take_2:
    expected_files = {f for f in expected_files if "take-2" not in f}
    
  modified_files = {modify_file_name(file_name) for file_name in expected_files}
  expected_files = modified_files

  # Filter out `take-2` files if `include_take_2` is False
  

  video_files = {
    f for f in sorted(os.listdir(input_folder)) if f.endswith(".mp4")
  }

  video_files = {
    f.replace('30FPS', f'{fps}FPS') for f in sorted(os.listdir(input_folder)) if f.endswith(".mp4")
  }


  missing_files = expected_files - video_files
  extra_files = video_files - expected_files

  if missing_files:
    raise FileNotFoundError(
      f"Validation failed: Missing required files in {input_folder}: "
      f"{missing_files}"
    )
  if extra_files:
    print(f"Warning: Extra files in {input_folder}: {extra_files}")

  print(f"Validation successful for {input_folder}.")
  return expected_files

def get_video_fps(input_folder: str) -> float:
  """
  Gets the FPS of videos in the input folder and ensures they are
  the same across all videos.

  Args:
    input_folder: Path to the folder containing input videos.

  Returns:
    The FPS of the videos.

  Raises:
    ValueError: If no MP4 files are found or FPS values are inconsistent.
  """
  video_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith(".mp4")]

  if not video_files:
      raise ValueError(f"No MP4 files found in {input_folder}.")

  fps_values = set()
  for video_file in video_files:
      video_path = os.path.join(input_folder, video_file)
      cap = cv2.VideoCapture(video_path)
      fps = cap.get(cv2.CAP_PROP_FPS)
      fps_values.add(fps)
      cap.release()

  if len(fps_values) > 1:
      raise ValueError(f"Inconsistent FPS values: {fps_values}")

  fps = fps_values.pop()
  print(f"All videos in {input_folder} have FPS: {fps}")
  return fps

def ensure_real_videos_at_fps(
    output_folder: str, target_fps: float, descriptions_file: str
) -> str:
  """
  Ensures that the physics-IQ-benchmark/split-videos/testing/{fps}FPS folder exists, generating it
  from physics-IQ-benchmark/split-videos/testing/30FPS if necessary.

  Args:
    output_folder: Path to the output folder.
    target_fps: Target FPS for the videos.
    descriptions_file: Path to the descriptions CSV file.

  Returns:
    Path to the folder containing the real videos at the target FPS.
  """
  testing_folder = "./physics-IQ-benchmark/split-videos/testing"
  thirty_fps_folder = os.path.join(testing_folder, "30FPS")
  target_fps_folder = os.path.join(testing_folder, f"{int(target_fps)}FPS")

  if os.path.exists(target_fps_folder):
    print(f"Validating real videos at FPS {int(target_fps)}...")
    expected_files = validate_video_files(thirty_fps_folder, descriptions_file, 'testing-videos', int(target_fps), include_take_2=True)
    try:
      validate_folder_files_exist(
        target_fps_folder, expected_files, f"Real videos for FPS {int(target_fps)}"
      )
      print(f"Real videos at FPS {int(target_fps)} are complete and ready at {target_fps_folder}.")
      return target_fps_folder
    except FileNotFoundError:
      print(f"Incomplete real videos at FPS {int(target_fps)}. Regenerating...")

  print(f"Generating real videos for FPS {int(target_fps)} from 30FPS...")
  validate_directory_exists(thirty_fps_folder, "30FPS folder")
  expected_files = validate_video_files(thirty_fps_folder, descriptions_file, 'testing-videos', int(target_fps), include_take_2=True)
  change_video_fps(thirty_fps_folder, target_fps_folder, target_fps)
  validate_folder_files_exist(
    target_fps_folder, expected_files, f"Real videos for FPS {int(target_fps)}"
  )

  print(f"Real videos at FPS {int(target_fps)} are ready at {target_fps_folder}.")
  return target_fps_folder

def ensure_binary_mask_structure(
    output_folder: str, input_folder: str, target_fps: float, expected_files: set[str], is_real: bool
) -> str:
  """
  Ensures binary masks for videos are generated and validated.

  Args:
    output_folder: Path to the base output folder.
    input_folder: Path to the folder containing input videos.
    target_fps: FPS of the videos.
    expected_files: Set of filenames to validate.
    is_real: Whether the masks are for real videos (True) or generated videos (False).

  Returns:
    Path to the binary masks folder.
  """
  folder_type = "real" if is_real else f"generated/{os.path.basename(os.path.normpath(input_folder))}"
  binary_mask_folder = f"./physics-IQ-benchmark/video-masks/{folder_type}/{int(target_fps)}FPS"

  if not os.path.exists(binary_mask_folder):
    print(f"Binary masks for {'real' if is_real else 'generated'} videos do not exist. Creating...")
    os.makedirs(binary_mask_folder, exist_ok=True)
    generate_binary_masks(input_folder, binary_mask_folder)

  validate_folder_files_exist(
    binary_mask_folder, expected_files, f"Binary masks for {'real' if is_real else 'generated'} videos at FPS {int(target_fps)}"
  )
  print(f"Binary masks for {'real' if is_real else 'generated'} videos are ready at {binary_mask_folder}.")
  return binary_mask_folder

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Video Processing Script.")
  parser.add_argument(
    "--input_folders",
    type=str,
    nargs="+",
    required=True,
    help="Paths to the folders containing input videos.",
  )
  parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Path to the folder for output videos.",
  )
  parser.add_argument(
    "--descriptions_file",
    type=str,
    required=True,
    help="Path to the descriptions CSV file (master file).",
  )

  args = parser.parse_args()

  csv_files_folder = os.path.join(args.output_folder, "results")
  os.makedirs(csv_files_folder, exist_ok=True)

  for input_folder in args.input_folders:
    print(f"\nProcessing folder: {input_folder}")

    validate_directory_exists(input_folder, "Input folder")
    validate_directory_exists(args.descriptions_file, "Descriptions file")

    fps = get_video_fps(input_folder)
      
    # Ensure real videos exist at the target FPS
    real_video_folder = ensure_real_videos_at_fps(args.output_folder, fps, args.descriptions_file)

    # Generate binary masks for real videos
    expected_real_files = validate_video_files(real_video_folder, args.descriptions_file, 'testing-videos', int(fps), include_take_2=True)
    expected_real_files = {f.replace("testing-videos", "mask-videos") for f in expected_real_files}
    binary_mask_real_folder = ensure_binary_mask_structure(
      output_folder=args.output_folder,
      input_folder=real_video_folder,
      target_fps=fps,
      expected_files=expected_real_files,
      is_real=True,
    )

    # Generate binary masks for generated videos
    expected_generated_files = validate_video_files(input_folder, args.descriptions_file, '', int(fps), include_take_2=False)
    #expected_generated_files = [f.replace("", "video-masks") for f in expected_real_files]
    binary_mask_generated_folder = ensure_binary_mask_structure(
      output_folder=args.output_folder,
      input_folder=input_folder,
      target_fps=fps,
      expected_files=expected_generated_files,
      is_real=False,
    )

    input_folder_name = os.path.basename(input_folder.rstrip("/"))
    csv_file_path = os.path.join(csv_files_folder, f"{input_folder_name}.csv")
    # Check if the CSV is complete
    # Validate all required real video scenarios
    expected_real_scenarios = validate_video_files(real_video_folder, args.descriptions_file, 'testing-videos', int(fps), include_take_2=True)
    # Check if the CSV is complete
    if is_csv_complete(csv_file_path, expected_real_scenarios):
      print(f"Skipping calculations for {csv_file_path} as it is already complete.")
    else:

      process_videos(
        real_folders=real_video_folder,
        generated_folders=input_folder,
        binary_real_folders=binary_mask_real_folder,
        binary_generated_folders=binary_mask_generated_folder,
        csv_file_path=csv_file_path,
        fps=fps,
        video_time_selection="first",
      )

  process_directory(csv_files_folder)
  print(f"\nCheck out {csv_files_folder} for saved plots and metrics.")
  print("Thank you for using the Physics-IQ benchmark!")

