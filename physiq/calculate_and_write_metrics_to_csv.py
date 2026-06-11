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

import concurrent.futures
import gc
import os
from collections import defaultdict
from dataclasses import dataclass

import cv2
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


@dataclass
class ViewPaths:
    """Resolved file paths for one (scenario, view) combination."""

    real_v1: str
    real_v2: str
    generated: str
    mask_v1: str
    mask_v2: str
    mask_generated: str


@dataclass
class ViewFrames:
    """Loaded frames for one (scenario, view) combination.

    Each field is a numpy array of shape (n_frames, h, w, c).
    RGB arrays are float64 in [0, 1]; mask arrays are bool.
    """

    real_v1: np.ndarray  # float in [0,1] shape (n_frames, h, w, c)
    real_v2: np.ndarray  # float in [0,1] shape (n_frames, h, w, c)
    generated: np.ndarray  # float in [0,1] shape (n_frames, h, w, c)
    mask_v1: np.ndarray  # bool in [0,1] shape (n_frames, h, w)
    mask_v2: np.ndarray  # bool in [0,1] shape (n_frames, h, w)
    mask_generated: np.ndarray  # bool in [0,1] shape (n_frames, h, w)


def load_and_resize_video(
    filepath, start_frame, end_frame, target_size=None, normalize=True
) -> np.ndarray:
    """Load and resize a video.

    Args:
        filepath: Path to the video file.
        start_frame: Index of the first frame to load.
        end_frame: Index of the last frame to load.
        target_size: Desired size of the frames (width, height).
        normalize: Whether to normalize the pixel values to the range [0, 1].

    Returns:
        Array of shape: F (frames) x target_size[1] (height) x target_size[0] (width) x 3
    """
    assert os.path.exists(filepath), f"File not found: {filepath}"

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
    return np.stack(frames)


def spatial_binary_masks(mask_frames: np.ndarray) -> np.ndarray:
    """Collapse the time dimension: True where any frame has a mask pixel."""
    if mask_frames.shape[0] == 0:
        print(
            "Warning: Received empty mask_frames for spatial binary mask calculation."
        )
        return np.zeros((1, 1), dtype=bool)
    return mask_frames.any(axis=0)


def weighted_spatial_mask(mask_frames: np.ndarray) -> np.ndarray:
    """Return per-pixel fraction of frames that have a mask pixel."""
    return np.sum(mask_frames, axis=0, dtype=np.uint16) / len(mask_frames)


def mse_per_frame(video1: np.ndarray, video2: np.ndarray) -> list[float]:
    """Calculate MSE per frame for two (n_frames, h, w, c) video arrays."""
    if video1.shape != video2.shape:
        raise ValueError("Videos must have the same shape.")
    diff = video1.astype(np.float32) - video2.astype(np.float32)
    return (diff**2).mean(axis=(1, 2, 3)).tolist()


def spatiotemporal_iou_per_frame(mask1: np.ndarray, mask2: np.ndarray) -> list[float]:
    """Calculate IOU per frame for two (n_frames, h, w) bool mask arrays."""
    spatial_axes = tuple(range(1, mask1.ndim))
    intersection = np.logical_and(mask1, mask2).sum(axis=spatial_axes)
    union = np.logical_or(mask1, mask2).sum(axis=spatial_axes)
    # have minimum union value of 1 to ensure that no divide by zero errors start
    iou = np.where(union == 0, 1.0, intersection / np.maximum(union, 1))
    return iou.tolist()


def compute_weighted_spatial_iou(weighted_spatial_1, weighted_spatial_2):
    """Compute IOU between two weighted spatial masks."""
    intersection = np.minimum(weighted_spatial_1, weighted_spatial_2)
    union = np.maximum(weighted_spatial_1, weighted_spatial_2)
    valid_pixels = union > 0
    if np.sum(valid_pixels) == 0:
        return 1.0
    return np.sum(intersection[valid_pixels]) / np.sum(union[valid_pixels])


def load_view(
    paths: ViewPaths, start_frame: int, end_frame: int, consider_frames: int
) -> ViewFrames | None:
    """Load and prepare all frames for one view from disk.

    Returns None if real_v1 has no frames (signals a missing video).
    Raises ValueError if any other required video is empty.
    """
    real_v1_sample = load_and_resize_video(paths.real_v1, 0, 1, normalize=False)
    if len(real_v1_sample) == 0:
        return None
    target_size = (real_v1_sample[0].shape[1] // 4, real_v1_sample[0].shape[0] // 4)

    real_v1 = load_and_resize_video(paths.real_v1, 0, consider_frames, target_size)
    real_v2 = load_and_resize_video(paths.real_v2, 0, consider_frames, target_size)
    generated = load_and_resize_video(
        paths.generated, start_frame, end_frame, target_size
    )
    mask_v1 = (
        load_and_resize_video(
            paths.mask_v1, 0, consider_frames, target_size, normalize=False
        )[:, :, :, 0]
        > 127
    )
    mask_v2 = (
        load_and_resize_video(
            paths.mask_v2, 0, consider_frames, target_size, normalize=False
        )[:, :, :, 0]
        > 127
    )
    mask_generated = (
        load_and_resize_video(
            paths.mask_generated, 0, consider_frames, target_size, normalize=False
        )[:, :, :, 0]
        > 127
    )

    vid_shape = real_v1.shape
    mask_shape = mask_v1.shape
    if (
        vid_shape != real_v2.shape
        or vid_shape != generated.shape
        or mask_shape != mask_v2.shape
        or mask_shape != mask_generated.shape
    ):
        raise ValueError(
            f"Frames are or shapes are inconistent across generated videos"
        )

    return ViewFrames(
        real_v1=real_v1,
        real_v2=real_v2,
        generated=generated,
        mask_v1=mask_v1,
        mask_v2=mask_v2,
        mask_generated=mask_generated,
    )


def compute_view_metrics(frames: ViewFrames) -> dict:
    """Compute all metrics for one view from loaded frames.

    Returns a dict with neutral keys (no view suffix).
    """
    spatiotemporal_iou_v1 = spatiotemporal_iou_per_frame(
        frames.mask_v1, frames.mask_generated
    )
    spatiotemporal_iou_v2 = spatiotemporal_iou_per_frame(
        frames.mask_v2, frames.mask_generated
    )
    variance_spatiotemporal_iou = spatiotemporal_iou_per_frame(
        frames.mask_v1, frames.mask_v2
    )

    spatial_v1 = spatial_binary_masks(frames.mask_v1)
    spatial_v2 = spatial_binary_masks(frames.mask_v2)
    spatial_generated = spatial_binary_masks(frames.mask_generated)

    iou_v1_spatial = spatiotemporal_iou_per_frame(
        spatial_v1[np.newaxis], spatial_generated[np.newaxis]
    )[0]
    iou_v2_spatial = spatiotemporal_iou_per_frame(
        spatial_v2[np.newaxis], spatial_generated[np.newaxis]
    )[0]
    variance_spatial = spatiotemporal_iou_per_frame(
        spatial_v1[np.newaxis], spatial_v2[np.newaxis]
    )[0]

    weighted_spatial_v1 = weighted_spatial_mask(frames.mask_v1)
    weighted_spatial_v2 = weighted_spatial_mask(frames.mask_v2)
    weighted_spatial_generated = weighted_spatial_mask(frames.mask_generated)

    iou_v1_weighted_spatial = compute_weighted_spatial_iou(
        weighted_spatial_v1, weighted_spatial_generated
    )
    iou_v2_weighted_spatial = compute_weighted_spatial_iou(
        weighted_spatial_v2, weighted_spatial_generated
    )
    variance_weighted_spatial = compute_weighted_spatial_iou(
        weighted_spatial_v1, weighted_spatial_v2
    )

    return {
        "spatiotemporal_iou_v1": spatiotemporal_iou_v1,
        "spatiotemporal_iou_v2": spatiotemporal_iou_v2,
        "spatial_iou_v1": iou_v1_spatial,
        "spatial_iou_v2": iou_v2_spatial,
        "weighted_spatial_iou_v1": iou_v1_weighted_spatial,
        "weighted_spatial_iou_v2": iou_v2_weighted_spatial,
        "v1_mse": mse_per_frame(frames.real_v1, frames.generated),
        "v2_mse": mse_per_frame(frames.real_v2, frames.generated),
        "variance_spatial": variance_spatial,
        "variance_weighted_spatial": variance_weighted_spatial,
        "variance_spatiotemporal_iou": variance_spatiotemporal_iou,
        "variance_mse": mse_per_frame(frames.real_v1, frames.real_v2),
    }


def process_view(
    paths: ViewPaths, view: str, start_frame: int, end_frame: int, consider_frames: int
) -> dict | None:
    """Load frames and compute metrics for one view; keys are suffixed with the view name."""
    frames = load_view(paths, start_frame, end_frame, consider_frames)
    if frames is None:
        return None
    metrics = compute_view_metrics(frames)
    del frames
    gc.collect()
    return {f"{k}_{view}": v for k, v in metrics.items()}


def _build_view_paths(
    scenario_name: str,
    view: str,
    id_take1: str,
    id_take2: str,
    fps: int,
    real_folder: str,
    generated_folder: str,
    binary_real_folder: str,
    binary_generated_folder: str,
) -> ViewPaths:
    return ViewPaths(
        real_v1=os.path.join(
            real_folder,
            f"{id_take1}_testing-videos_{fps}FPS_{view}_take-1_{scenario_name}",
        ),
        real_v2=os.path.join(
            real_folder,
            f"{id_take2}_testing-videos_{fps}FPS_{view}_take-2_{scenario_name}",
        ),
        generated=os.path.join(generated_folder, f"{id_take1}_{view}_{scenario_name}"),
        mask_v1=os.path.join(
            binary_real_folder,
            f"{id_take1}_video-masks_{fps}FPS_{view}_take-1_{scenario_name}",
        ),
        mask_v2=os.path.join(
            binary_real_folder,
            f"{id_take2}_video-masks_{fps}FPS_{view}_take-2_{scenario_name}",
        ),
        mask_generated=os.path.join(
            binary_generated_folder,
            f"{id_take1}_video-masks_{fps}FPS_{view}_take-1_{scenario_name}",
        ),
    )


def process_videos(
    real_folder: str,
    generated_folder: str,
    binary_real_folder: str,
    binary_generated_folder: str,
    csv_file_path: str,
    fps: int,
    video_time_selection="first",
    n_processes: int = 2,
):
    """Goes through the videos and masks, and calculates metrics.

    This function processes a set of real and generated videos along with their
    corresponding binary masks. It calculates various metrics such as MSE,
    IOU, and others, and saves the results to a specified CSV file.

    Args:
        real_folders (str): A path to folder containing real
          videos.
        generated_folders (str): A paths to folder containing
          generated videos.
        binary_real_folders (str): A path to folder containing
          binary masks for real videos.
        binary_generated_folder (str): A path to folders
          containing binary masks for generated videos.
        csv_file_path (str): The file path where the results will be saved as a
          CSV file.
        fps (int): frames per second (FPS) value for each
          video.
        video_time_selection (str): Specifies which part of the video to process
          (e.g., 'first', 'last').
        n_processes (int): Number of worker processes. 0 runs everything
          serially in the main process (useful for debugging).

    Returns:
        None: This function does not return any value but saves the results to a
        CSV file.
    """
    if not os.path.exists(real_folder):
        print(f"Folder not found: {real_folder}")
        return

    gen_video_duration_frames = get_video_frame_count(
        os.path.join(generated_folder, sorted(os.listdir(generated_folder))[0])
    )

    consider_frames = fps * 5
    if video_time_selection == "first":
        start_frame, end_frame = 0, consider_frames
    else:
        start_frame, end_frame = (
            gen_video_duration_frames - (5 * fps),
            gen_video_duration_frames,
        )

    # First pass: collect IDs for each (scenario, view, take) from real filenames
    scenario_info = {}
    for real_file in sorted(os.listdir(real_folder)):
        if not real_file.endswith(".mp4"):
            continue
        parts = real_file.split("_")
        if len(parts) < 6:
            print(f"Unexpected filename format: {real_file}")
            continue

        scenario_name = parts[5]
        file_id = parts[0]
        view = parts[3]

        if scenario_name not in scenario_info:
            scenario_info[scenario_name] = {"take-1": {}, "take-2": {}}

        if "take-1" in real_file:
            scenario_info[scenario_name]["take-1"][view] = file_id
        elif "take-2" in real_file:
            scenario_info[scenario_name]["take-2"][view] = file_id
        else:
            raise ValueError("File must contain either take-1 or take-2")

    progress_bar = tqdm.tqdm(total=len(scenario_info), desc="Processing scenarios")

    # Second pass: build flat task list (validates IDs and resolves paths)
    tasks = []
    for scenario_name, ids in scenario_info.items():
        take_1_views = ids["take-1"]
        take_2_views = ids["take-2"]
        for view in ["perspective-left", "perspective-center", "perspective-right"]:
            if view not in take_1_views or view not in take_2_views:
                raise ValueError(
                    f"Missing IDs for scenario {scenario_name}, view {view}: "
                    f"take-1={take_1_views.get(view)}, take-2={take_2_views.get(view)}"
                )
            tasks.append(
                (
                    scenario_name,
                    view,
                    _build_view_paths(
                        scenario_name,
                        view,
                        take_1_views[view],
                        take_2_views[view],
                        int(fps),
                        real_folder,
                        generated_folder,
                        binary_real_folder,
                        binary_generated_folder,
                    ),
                )
            )

    # Shared result store
    view_results_by_scenario = defaultdict(dict)
    completed_views = defaultdict(int)

    def _store(scenario_name, view_result):
        if view_result:
            for key, value in view_result.items():
                view_results_by_scenario[scenario_name][key] = (
                    [float(v) for v in value] if isinstance(value, list) else value
                )
        completed_views[scenario_name] += 1
        if completed_views[scenario_name] == 3:
            progress_bar.update(1)

    # Third pass: execute — serially or in parallel
    scenario_data = []
    if n_processes == 0:
        for scenario_name, view, paths in tasks:
            _store(
                scenario_name,
                process_view(paths, view, start_frame, end_frame, consider_frames),
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_processes
        ) as executor:
            future_to_scenario = {
                executor.submit(
                    process_view, paths, view, start_frame, end_frame, consider_frames
                ): scenario_name
                for scenario_name, view, paths in tasks
            }
            for future in concurrent.futures.as_completed(future_to_scenario):
                _store(future_to_scenario[future], future.result())

    for scenario_name in scenario_info:
        scenario_data.append(
            {"scenario": scenario_name, **view_results_by_scenario[scenario_name]}
        )

    if scenario_data:
        df = pd.DataFrame(scenario_data)
        df.to_csv(csv_file_path, index=False)
    else:
        print("No data to write to CSV")
