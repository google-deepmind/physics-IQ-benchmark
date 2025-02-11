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

"""Calculate metrics and plot results from physics-IQ benchmark videos."""

import argparse
from multiprocessing import pool
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm


def get_video_frame_count(filepath):
  """Get the total number of frames in a video."""
  if not os.path.exists(filepath):
    return 0
  cap = cv2.VideoCapture(filepath)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cap.release()
  return total_frames


def process_videos(
    real_folders,
    generated_folders,
    binary_real_folders,
    binary_generated_folders,
    csv_file_path,
    fps,
    video_time_selection='first',
):
  """Goes through the videos and masks, and calculates metrics.

  This function processes a set of real and generated videos along with their
  corresponding binary masks. It calculates various metrics such as MSE,
  IOU, and others, and saves the results to a specified CSV file.

  Args:
      real_folders (list of str): A list of paths to folders containing real
        videos.
      generated_folders (list of str): A list of paths to folders containing
        generated videos.
      binary_real_folders (list of str): A list of paths to folders containing
        binary masks for real videos.
      binary_generated_folders (list of str): A list of paths to folders
        containing binary masks for generated videos.
      csv_file_path (str): The file path where the results will be saved as a
        CSV file.
      fps (int): frames per second (FPS) value for each
        video.
      video_time_selection (str): Specifies which part of the video to process
        (e.g., 'first', 'last').

  Returns:
      None: This function does not return any value but saves the results to a
      CSV file.
  """

  def spatial_binary_masks(mask_frames):
    """Collapse the time dimension of binary masks into a single frame."""
    if not mask_frames:
      print(
          'Warning: Received empty mask_frames for spatial binary mask '
          'calculation.'
      )
      return np.zeros(
          (1, 1), dtype=np.uint8
      )  # Return an empty mask to prevent errors
    spatial_mask = np.max(mask_frames, axis=0)
    return (spatial_mask > 0).astype(np.uint8) * 255

  def spatiotemporal_binary_mask(mask_frames):
    """Calculate an spatiotemporal presence map by counting the number of frames with activity."""
    spatiotemporal_mask = np.sum(mask_frames, axis=0)
    # Keep the count of active frames as pixel values
    return spatiotemporal_mask.astype(np.uint8)

  scenario_data = []
  processed_scenarios = set()
  gen_video_duration_frames = get_video_frame_count(
      os.path.join(generated_folders, sorted(os.listdir(generated_folders))[0])
  )

  def mse_per_frame(video1, video2):
    """Calculate MSE per frame for two videos.

    Args:
      video1:
      video2:

    Returns:

    """
    frame_mses = []
    for frame1, frame2 in zip(video1, video2):
      mse = np.mean((frame1.flatten() - frame2.flatten()) ** 2)
      frame_mses.append(mse)
    return frame_mses

  def load_and_resize_video(
      filepath, start_frame, end_frame, target_size=None, normalize=True
  ):
    """Load and resize a video.

    Args:
        filepath: Path to the video file.
        start_frame: Index of the first frame to load.
        end_frame: Index of the last frame to load.
        target_size: Desired size of the frames (width, height).
        normalize: Whether to normalize the pixel values to the range [0, 1].

    Returns:
        A list of frames from the video.
    """
    assert os.path.exists(filepath), f'File not found: {filepath}'

    cap = cv2.VideoCapture(filepath)
    frames = []
    frame_idx = 0
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      if start_frame <= frame_idx < end_frame:
        if target_size:
          frame = cv2.resize(frame, target_size)
        if normalize:
          frame = frame / 255.0
        frames.append(frame)
      frame_idx += 1
    cap.release()
    return frames

  def iou_per_frame(mask1, mask2):
    """Calculate Intersection over Union (IoU) per frame for two binary masks."""
    iou_values = []
    for m1, m2 in zip(mask1, mask2):
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        if union == 0:
            iou = 1.0  # Both masks are empty
        else:
            iou = intersection / union
        iou_values.append(iou)
    return iou_values

  scenario_data = []
  processed_scenarios = set()
  gen_video_duration_frames = get_video_frame_count(
      os.path.join(generated_folders, sorted(os.listdir(generated_folders))[0])
  )

  consider_frames = fps * 5
  if video_time_selection == 'first':
    start_frame, end_frame = (0, consider_frames)
  else:
    start_frame, end_frame = (
        gen_video_duration_frames - (5 * fps),
        gen_video_duration_frames,
    )

  progress_bar = tqdm.tqdm(
      total=len(
          set(
              '_'.join(f.split('_')[:-2])
              for f in sorted(os.listdir(real_folders))
          )
      ) // 6,
      desc='Processing scenarios'
  )
  if not os.path.exists(real_folders):
    print(f'Folder not found: {real_folders}')
    return


  def process_view(scenario_name, view, scenario_ID_take_1, scenario_ID_take_2, fps):
    """Process a single view of a scenario.

    Args:
      scenario_name: Name of the scenario.
      view: View identifier.

    Returns:
      A dictionary of calculated metrics.
    """
    print('## Processing scenario ', scenario_name)
    real_path_v1 = os.path.join(real_folders, f'{scenario_ID_take_1}_testing-videos_{fps}FPS_{view}_take-1_{scenario_name}')
    generated_path = os.path.join(
        generated_folders, f'{scenario_ID_take_1}_{view}_{scenario_name}'
    )
    real_path_v2 = os.path.join(real_folders, f'{scenario_ID_take_2}_testing-videos_{fps}FPS_{view}_take-2_{scenario_name}')
    binary_path_v1 = os.path.join(
        binary_real_folders, f'{scenario_ID_take_1}_video-masks_{fps}FPS_{view}_take-1_{scenario_name}'
    )
    binary_path_v2 = os.path.join(
        binary_real_folders, f'{scenario_ID_take_2}_video-masks_{fps}FPS_{view}_take-2_{scenario_name}'
    )
    binary_generated_path = os.path.join(
        binary_generated_folders, f'{scenario_ID_take_1}_video-masks_{fps}FPS_{view}_take-1_{scenario_name}'
    )

    # Load the first frame to determine the original target size
    real_v1_sample = load_and_resize_video(
        real_path_v1, 0, 1, normalize=False
    )
    if not real_v1_sample:
      return None  # Early exit if no frames are loaded
    target_size = (
        real_v1_sample[0].shape[1] // 4,
        real_v1_sample[0].shape[0] // 4,
    )

    real_v1_frames = load_and_resize_video(
        real_path_v1,
        0,
        consider_frames,
        target_size,
    )
    generated_frames = load_and_resize_video(
        generated_path, start_frame, end_frame, target_size
    )
    real_v2_frames = load_and_resize_video(
        real_path_v2,
        0,
        consider_frames,
        target_size,
    )
    binary_v1_frames = load_and_resize_video(
        binary_path_v1,
        3 * fps,
        8 * fps,
        target_size,
        normalize=False,
    )
    binary_v2_frames = load_and_resize_video(
        binary_path_v2,
        3 * fps,
        8 * fps,
        target_size,
        normalize=False,
    )
    binary_generated_frames = load_and_resize_video(
        binary_generated_path,
        0,
        consider_frames,
        target_size,
        normalize=False,
    )
    # Ensure binary masks are binary (0 or 1 pixel values)
    binary_v1_frames = [
        (mask > 127).astype(np.uint8) * 255 for mask in binary_v1_frames
    ]
    binary_v2_frames = [
        (mask > 127).astype(np.uint8) * 255 for mask in binary_v2_frames
    ]
    binary_generated_frames = [
        (mask > 127).astype(np.uint8) * 255 for mask in binary_generated_frames
    ]

    # Check if frames are loaded properly
    if (
        not generated_frames
        or not real_v2_frames
        or not real_v1_frames
        or not binary_v1_frames
        or not binary_v2_frames
    ):
      raise ValueError(f"Scenario {scenario_name}_{view} has missing frames.")

    # Create subdirectories for each model in spatial and spatiotemporal maps
    # folders
    model_name = generated_folders.split('/')[-1]


    # Calculate IOU for v1 and v2 masks
    iou_v1 = iou_per_frame(binary_v1_frames, binary_generated_frames)
    iou_v2 = iou_per_frame(binary_v2_frames, binary_generated_frames)

    # Collapse binary masks over time to calculate spatial IOU
    spatial_v1 = spatial_binary_masks(binary_v1_frames)
    spatial_v2 = spatial_binary_masks(binary_v2_frames)

    # Calculate spatial IOU for v1 and v2
    spatial_generated = spatial_binary_masks(binary_generated_frames)

    iou_v1_spatial = iou_per_frame([spatial_v1], [spatial_generated])[0]
    iou_v2_spatial = iou_per_frame([spatial_v2], [spatial_generated])[0]

    # Calculate spatial IOU
    spatial_iou = iou_per_frame([spatial_v1], [spatial_v2])[0]  # Calculate IOU between collapsed spatial masks


    # Calculate spatiotemporal presence IOU for v1 and v2 masks
    spatiotemporal_v1 = spatiotemporal_binary_mask(binary_v1_frames)
    spatiotemporal_v2 = spatiotemporal_binary_mask(binary_v2_frames)

    # Calculate spatiotemporal IOU
    max_spatiotemporal_v1 = np.sum(
        np.maximum(spatiotemporal_v1, binary_generated_frames)
    )
    if max_spatiotemporal_v1 != 0:
      iou_v1_spatiotemporal = np.sum(
          np.minimum(spatiotemporal_v1, binary_generated_frames)
      ) / max_spatiotemporal_v1
    else:
      iou_v1_spatiotemporal = 0
    max_spatiotemporal_v2 = np.sum(
        np.maximum(spatiotemporal_v2, binary_generated_frames)
    )
    if max_spatiotemporal_v2 != 0:
      iou_v2_spatiotemporal = np.sum(
          np.minimum(spatiotemporal_v2, binary_generated_frames)
      ) / max_spatiotemporal_v2
    else:
      iou_v2_spatiotemporal = 0
    # Check for empty spatiotemporal masks
    if spatiotemporal_v1.size == 0 or spatiotemporal_v2.size == 0:
      print(f'Empty spatiotemporal masks for scenario {scenario_name}_{view}')
      return None
    # Calculate variance for spatial and spatiotemporal IOUs
    variance_spatial = iou_per_frame(
        [spatial_v1], [spatial_v2]
    )  # Calculate per-frame spatial IOU between v1 and v2 masks
    spatiotemporal_v1_max = np.sum(
        np.maximum(spatiotemporal_v1, spatiotemporal_v2)
    )
    if spatiotemporal_v1_max != 0:
      variance_spatiotemporal = np.sum(
          np.minimum(spatiotemporal_v1, spatiotemporal_v2)
      ) / spatiotemporal_v1_max
    else:
      variance_spatiotemporal = 0
    # Calculate variance IOU
    variance_iou = iou_per_frame(
        binary_v1_frames, binary_v2_frames
    )  # Calculate per-frame IOU between v1 and v2 masks

    # Calculate variance MSE
    variance_mse = mse_per_frame(
        real_v1_frames, real_v2_frames
    )  # Calculate MSE between v1 and v2 frames
    # Print average variances for this scenario
    #print(f"Average Variances for Scenario '{scenario_name}_{view}':")
    #print(f' - Variance Spatial: {np.mean(variance_spatial):.4f}')
    #print(f' - Variance Spatiotemporal: {variance_spatiotemporal:.4f}')
    #print(f' - Variance IOU: {np.mean(variance_iou):.4f}')
    #print(f' - Variance MSE: {np.mean(variance_mse):.4f}')

    # Return the result for this view
    return {
        f'v1_mse_{view}': mse_per_frame(real_v1_frames, generated_frames),
        f'v2_mse_{view}': mse_per_frame(real_v2_frames, generated_frames),
        f'spatial_iou_{view}': spatial_iou,  # Spatial IOU
        f'variance_spatial_{view}': np.mean(
            variance_spatial
        ),  # Variance for spatial IOU
        f'variance_spatiotemporal_{view}': (
            variance_spatiotemporal
        ),  # Variance for spatiotemporal IOU
        f'variance_iou_{view}': variance_iou,  # Variance IOU metric
        f'variance_mse_{view}': variance_mse,  # Variance MSE metric
        f'upper_bound_{view}': mse_per_frame(
            real_v1_frames,
            real_v2_frames,
        ),
        f'iou_v1_{view}': iou_v1,  # IOU for v1
        f'iou_v2_{view}': iou_v2,  # IOU for v2
        'spatial_iou_v1': iou_v1_spatial,  # Spatial IOU for v1
        'spatial_iou_v2': iou_v2_spatial,  # Spatial IOU for v2
        'spatiotemporal_iou_v1': (
            iou_v1_spatiotemporal
        ),  # Spatiotemporal IOU for v1
        'spatiotemporal_iou_v2': (
            iou_v2_spatiotemporal
        ),  # Spatiotemporal IOU for v2
    }
  # Iterate over all scenarios
  # Dictionary to track processed scenarios and their IDs
  scenario_info = {}
  processed_scenarios = set()
  scenario_data = []

  # First pass: Populate scenario_info with both take-1 and take-2 IDs per view
  for real_file in sorted(os.listdir(real_folders)):
      if real_file.endswith('.mp4'):
          parts = real_file.split('_')
          if len(parts) < 6:
              print(f'Unexpected filename format: {real_file}')
              continue

          # Extract attributes
          scenario_name = parts[5]
          file_id = parts[0]
          view = parts[3]  # e.g., perspective-left
          version = parts[4]  # e.g., 30FPS

          # Initialize scenario entry with views if not present
          if scenario_name not in scenario_info:
              scenario_info[scenario_name] = {"take-1": {}, "take-2": {}}

          # Update the IDs for the specific view
          if 'take-1' in real_file:
              scenario_info[scenario_name]["take-1"][view] = file_id
          elif 'take-2' in real_file:
              scenario_info[scenario_name]["take-2"][view] = file_id
          else:
              raise ValueError('File must contain either take-1 or take-2')

  # Second pass: Process each scenario
  for scenario_name, ids in scenario_info.items():
      take_1_views = ids["take-1"]  # Dictionary of views and IDs for take-1
      take_2_views = ids["take-2"]  # Dictionary of views and IDs for take-2

      # Ensure all views have both take-1 and take-2 IDs
      for view in ['perspective-left', 'perspective-center', 'perspective-right']:
          if view not in take_1_views or view not in take_2_views:
              raise ValueError(f"Missing IDs for scenario {scenario_name}, view {view}: "
                               f"take-1={take_1_views.get(view)}, take-2={take_2_views.get(view)}")

      processed_scenarios.add(scenario_name)
      progress_bar.update(1)

      scenario_result = {'scenario': scenario_name}

      # Process each view in parallel
      with pool.ThreadPool() as executor:
          futures = [
              executor.apply_async(process_view, 
                                  (scenario_name, view, take_1_views[view], take_2_views[view], int(fps)))
              for view in ['perspective-left', 'perspective-center', 'perspective-right']
          ]
          for future in futures:
              view_result = future.get()
              if view_result:
                  for key, value in view_result.items():
                      if isinstance(value, list):
                          scenario_result[key] = [float(v) for v in value]
                      else:
                          scenario_result[key] = value
      scenario_data.append(scenario_result)

  # Convert list to DataFrame and write to CSV
  if scenario_data:
      df = pd.DataFrame(scenario_data)
      df.to_csv(csv_file_path, index=False)
  else:
      print('No data to write to CSV')





if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process videos and plot MSE.')
  parser.add_argument(
      '--real_folder',
      type=str,
      nargs='+',
      help='Paths to the folders containing real videos.',
  )
  parser.add_argument(
      '--generated_folder',
      type=str,
      nargs='+',
      help='Paths to the folders containing generated videos.',
  )
  parser.add_argument(
      '--binary_real_folder',
      type=str,
      nargs='+',
      help='Paths to the folders containing binary real videos.',
  )
  parser.add_argument(
      '--binary_generated_folders',
      type=str,
      nargs='+',
      help='Paths to the folders containing binary generated videos.',
  )
  parser.add_argument(
      '--csv_files',
      type=str,
      nargs='+',
      help='Paths to the CSV files containing scenarios and MSE values.',
  )
  parser.add_argument(
      '--fps',
      type=float,
      nargs='+',
      required=True,
      help='FPS for each CSV file. Must match the number of CSV files.',
  )
  parser.add_argument(
      '--video_time',
      type=str,
      nargs='+',
      choices=['first', 'last'],
      default=['last'],
      help=(
          'Choose whether to pick the first or last 5 seconds '
          'of the video. Must match the number of CSV files.'
      ),
  )

  args = parser.parse_args()
  print(args.real_folder, args.generated_folder)
  if len(args.real_folder) != len(args.generated_folder):
    raise ValueError(
        'The number of real folders must match the number of generated folders.'
    )
  if len(args.real_folder) != len(args.csv_files):
    raise ValueError(
        'The number of real folders must match the number of CSV files.'
    )
  if len(args.real_folder) != len(args.fps):
    raise ValueError(
        'The number of real folders must match the number of FPS values.'
    )
  if len(args.video_time) != len(args.csv_files):
    raise ValueError(
        'The number of video time options must match the number of CSV files.'
    )

  # Process the videos and update the CSV files
  for (
      real_folder,
      generated_folder,
      binary_real_folder,
      binary_generated_folder,
      csv_file,
      fps,
      video_time,
  ) in zip(
      args.real_folder,
      args.generated_folder,
      args.binary_real_folder,
      args.binary_generated_folders,
      args.csv_files,
      args.fps,
      args.video_time,
  ):
    print(
        f"Processing {csv_file} with FPS {fps}, video time '{video_time}' using"
        f" real folder '{real_folder}' and generated folder"
        f" '{generated_folder}'."
    )

    csv_data = pd.read_csv(csv_file)


    # Proceed with processing if the row count is correct
    process_videos(
        real_folder,
        generated_folder,
        binary_real_folder,
        binary_generated_folder,
        csv_file,
        fps,
        video_time,
    )

  # Plot MSE values
  plot_metrics(args.csv_files, './plots', args.fps)

