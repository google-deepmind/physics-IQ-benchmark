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
import json
import math
import os
import subprocess
import warnings

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from physiq.binary_mask_generator import generate_binary_masks
from physiq.calculate_and_write_metrics_to_csv import process_videos
from physiq.calculate_iq_score import calculate_iq_score
from physiq.calculate_iq_score_stable import IQTable
from physiq.fps_changer import change_video_fps
from physiq.plot_settings import model_to_color, model_to_plotting_name


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
    base_expected_scenarios = {
        scenario.split("_")[-1] for scenario in expected_scenarios
    }
    csv_scenarios = set(df["scenario"])

    # Check for missing scenarios
    missing_scenarios = base_expected_scenarios - csv_scenarios
    if missing_scenarios:
        print(
            f"CSV file {csv_file_path} is incomplete. Missing scenarios: {missing_scenarios}"
        )
        return False

    print(f"CSV file {csv_file_path} is complete with all scenarios.")
    return True


def rename_generated_videos(
    generated_video_directory: str, real_video_directory: str
) -> None:
    """Renames .mp4 files in the given directory for consistency.

    Args:
      generated_video_directory: The directory containing the generated .mp4 files.
      real_video_directory: The directory containing the corresponding real .mp4 files.
    """

    validate_generations(generated_video_directory)

    name_mapping = {}
    prefix_length = 4
    for realfile in sorted(os.listdir(real_video_directory)):
        parts = realfile.split("_")
        name_mapping[realfile[:prefix_length]] = "_".join(
            [
                parts[0],  # Keep the first part (e.g., 0092)
                parts[3],  # Keep the perspective (e.g., perspective-center)
                parts[5],  # Keep the scenario name (e.g., trimmed-lit-candle.mp4)
            ]
        )

    for genfile in sorted(os.listdir(generated_video_directory)):
        if genfile.endswith(".mp4"):
            new_filename = name_mapping[genfile[:prefix_length]]
            old_path = os.path.join(generated_video_directory, genfile)
            new_path = os.path.join(generated_video_directory, new_filename)
            os.rename(old_path, new_path)
        else:
            raise ValueError("Only .mp4 files are supported.")


def validate_path_exists(directory: str, description: str) -> None:
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
    expected_files = {f.replace("30FPS", f"{fps}FPS") for f in actual_files}
    missing_files = expected_files - actual_files
    if missing_files:
        raise FileNotFoundError(
            f"{description} folder is missing files: {missing_files}"
        )

    print(f"{description} folder is valid with all required files.")


def get_video_duration(video_path):
    """Return the duration of a video in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


def validate_generations(
    input_folder: str, num_videos: int = 198, video_duration_s: float = 5
):
    """Check that num_videos .mp4 videos exist, each video_duration_s seconds with an ID prefix from 0001_ to 0198_."""

    assert os.path.exists(input_folder)
    assert os.path.isdir(input_folder)

    files_in_folder = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    )

    length_error_msg = f"found {len(files_in_folder)} videos but expected {num_videos}"
    assert len(files_in_folder) == num_videos, length_error_msg

    counter = 1
    for f in files_in_folder:
        expected_prefix = "{:04d}_".format(counter)
        assert f.startswith(
            expected_prefix
        ), "Video {f} does not start with expected video ID {expected_prefix}"

        video_path = os.path.join(input_folder, f)
        video_duration = get_video_duration(video_path)

        duration_error_msg = (
            f"Video {video_path} is {video_duration} seconds long but needs to be 5s. "
            + "Please ensure that all generated videos are exactly 5 seconds long."
        )
        assert math.isclose(
            video_duration, video_duration_s, abs_tol=0.001
        ), duration_error_msg
        counter += 1


def validate_video_files(
    input_folder: str,
    descriptions_file: str,
    video_type: str,
    fps: int,
    include_take_2: bool = False,
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

    video_files = {f for f in sorted(os.listdir(input_folder)) if f.endswith(".mp4")}

    video_files = {
        f.replace("30FPS", f"{fps}FPS")
        for f in sorted(os.listdir(input_folder))
        if f.endswith(".mp4")
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
    target_fps: float, descriptions_file: str, gt_folder: str
) -> str:
    """
    Ensures that the gt_folder/split-videos/testing/{fps}FPS folder exists, generating it
    from gt_folder/split-videos/testing/30FPS if necessary.

    Args:
      output_folder: Path to the output folder.
      target_fps: Target FPS for the videos.
      descriptions_file: Path to the descriptions CSV file.
      gt_folder: path to benchmark e.g.  ./physics-IQ-benchmark or ./physics-IQ-benchmark-verifed

    Returns:
      Path to the folder containing the real videos at the target FPS.
    """
    testing_folder = os.path.join(gt_folder, "split-videos", "testing")
    thirty_fps_folder = os.path.join(testing_folder, "30FPS")
    target_fps_folder = os.path.join(testing_folder, f"{int(target_fps)}FPS")

    if os.path.exists(target_fps_folder):
        print(f"Validating real videos at FPS {int(target_fps)}...")
        expected_files = validate_video_files(
            thirty_fps_folder,
            descriptions_file,
            "testing-videos",
            int(target_fps),
            include_take_2=True,
        )
        try:
            validate_folder_files_exist(
                target_fps_folder,
                expected_files,
                f"Real videos for FPS {int(target_fps)}",
            )
            print(
                f"Real videos at FPS {int(target_fps)} are complete and ready at {target_fps_folder}."
            )
            return target_fps_folder
        except FileNotFoundError:
            print(f"Incomplete real videos at FPS {int(target_fps)}. Regenerating...")

    print(f"Generating real videos for FPS {int(target_fps)} from 30FPS...")
    validate_path_exists(thirty_fps_folder, "30FPS folder")
    expected_files = validate_video_files(
        thirty_fps_folder,
        descriptions_file,
        "testing-videos",
        int(target_fps),
        include_take_2=True,
    )
    change_video_fps(thirty_fps_folder, target_fps_folder, target_fps)
    validate_folder_files_exist(
        target_fps_folder, expected_files, f"Real videos for FPS {int(target_fps)}"
    )

    print(f"Real videos at FPS {int(target_fps)} are ready at {target_fps_folder}.")
    return target_fps_folder


def ensure_binary_mask_structure(
    input_folder: str,
    target_fps: float,
    expected_files: set[str],
    is_real: bool,
    gt_folder: str,
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
    folder_type = (
        "real"
        if is_real
        else os.path.join("generated", os.path.basename(os.path.normpath(input_folder)))
    )
    binary_mask_folder = os.path.join(
        gt_folder, "video-masks", folder_type, f"{int(target_fps)}FPS"
    )

    if not os.path.exists(binary_mask_folder):
        print(
            f"Binary masks for {'real' if is_real else 'generated'} videos do not exist. Creating..."
        )
        os.makedirs(binary_mask_folder, exist_ok=True)
        generate_binary_masks(input_folder, binary_mask_folder, is_real)

    validate_folder_files_exist(
        binary_mask_folder,
        expected_files,
        f"Binary masks for {'real' if is_real else 'generated'} videos at FPS {int(target_fps)}",
    )
    print(
        f"Binary masks for {'real' if is_real else 'generated'} videos are ready at {binary_mask_folder}."
    )
    return binary_mask_folder


def process_directory(directory_path: str, lazy_integrity: bool = False) -> None:
    """
    Process all CSV files in a directory to compute Physics IQ scores
    and generate a bar plot.

    Args:
      directory_path: Path to the directory containing CSV files.
      lazy_integrity: Skip the DataFrame mutation guard on IQTable. Disables
        the check that detects accidental post-construction mutations; only use
        this when the integrity check is known to produce false positives (e.g.
        on pandas versions < 2.0).

    Returns:
      None
    """
    if lazy_integrity:
        warnings.warn(
            "lazy_integrity=True: the IQTable mutation guard is disabled. "
            "Cached scores will not be validated against the underlying DataFrame. "
            "Upgrade to pandas>=2.0 to avoid needing this flag.",
            UserWarning,
            stacklevel=2,
        )

    score_names = ["Original", "Verified"]
    model_scores_orig = {}
    model_scores_verified = {}
    csv_files = [f for f in sorted(os.listdir(directory_path)) if f.endswith(".csv")]

    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        print(f"Processing {csv_file}...")

        model_name = os.path.splitext(csv_file)[0]
        iqtable = IQTable.from_csv(file_path, lazy_integrity=lazy_integrity)
        values = iqtable.get_output_dict()
        values["final_score_origround"] = calculate_iq_score(file_path)[0]
        with open(os.path.join(directory_path, f"{model_name}_metrics.json"), "w") as f:
            json.dump(values, f, indent=4)
        final_score_orig = round(values["final_score_orig"], 4) * 100
        final_score_verified = round(values["final_score_view"], 4) * 100

        print(f"Physics-IQ score (original) for {model_name}: {final_score_orig:.2f}")
        print(
            f"Physics-IQ score (verified) for {model_name}: {final_score_verified:.2f}"
        )
        print("-" * 50)

        model_scores_orig[model_name] = final_score_orig
        model_scores_verified[model_name] = final_score_verified

    for score_name, model_scores in zip(
        score_names, [model_scores_orig, model_scores_verified]
    ):
        sorted_items = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        model_names = [m[0] for m in sorted_items]
        values = [item[1] for item in sorted_items]

        plt.figure(figsize=(10, 6))

        plot_colors = [model_to_color(m) for m in model_names]
        plot_names = [model_to_plotting_name(m) for m in model_names]

        bars = plt.bar(plot_names, values, color=plot_colors)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.axhline(y=100, color="darkgrey", linestyle="--", linewidth=2)

        midpoint = (len(model_names) - 1) / 2.0
        plt.text(
            midpoint,
            102,
            "Physical Variance",
            ha="center",
            va="bottom",
            color="black",
            fontweight="bold",
        )

        plt.xticks(rotation=45, ha="right")

        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.xlabel("")
        plt.ylabel(f"{score_name} Physics-IQ score \n(higher=better)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
        plt.tight_layout()
        plt.savefig(
            os.path.join(directory_path, f"physics_IQ_score_{score_name}_barplot.pdf")
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Video Processing Script. During evaluation intermediate results (videos) are saved in the benchmark_base_folder."
    )
    parser.add_argument(
        "--input_folders",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the folders containing generated videos.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder for outputs.",
    )
    parser.add_argument(
        "--descriptions_file",
        type=str,
        required=True,
        help="Path to the descriptions CSV file (master file).",
    )
    parser.add_argument(
        "--original_physics_iq",
        action="store_true",
        help="Switch from verified GT (default) to original GT.",
    )
    parser.add_argument(
        "--benchmark_base_folder",
        type=str,
        default=".",
        help="Path to the folder within which physics-IQ-benchmark & physics-IQ-benchmark-verifed data are located.",
    )
    parser.add_argument(
        "--n_process",
        type=int,
        default=0,
        help="Number of processes used for computation of scores based on images."
        "Be cerafule here, one process can easily take up 9GB.",
    )
    parser.add_argument(
        "--lazy_integrity",
        action="store_true",
        help=(
            "Disable the IQTable DataFrame mutation guard. "
            "Only use this if the integrity check raises a false positive on your pandas version. "
            "Upgrading to pandas>=2.0 is the recommended fix or use uv for handling your python version."
        ),
    )

    args = parser.parse_args()

    # Setting which Ground truth is being used for evaluation.
    if args.original_physics_iq:
        benchmark_name = "physics-IQ-benchmark"
    else:
        benchmark_name = "physics-IQ-benchmark-verified"

    benchmark_data_path = os.path.join(args.benchmark_base_folder, benchmark_name)
    print(
        "Using Data {} from: {}".format(
            "verified" if not args.original_physics_iq else "original",
            benchmark_data_path,
        )
    )

    csv_files_folder = os.path.join(args.output_folder, benchmark_name, "results")
    os.makedirs(csv_files_folder, exist_ok=True)

    for input_folder in args.input_folders:
        print(f"\nProcessing folder: {input_folder}")

        validate_path_exists(input_folder, "Input folder")
        validate_path_exists(args.descriptions_file, "Descriptions file")

        fps = get_video_fps(input_folder)

        # Ensure real videos exist at the target FPS
        real_video_folder = ensure_real_videos_at_fps(
            fps, args.descriptions_file, benchmark_data_path
        )

        # Generate binary masks for real videos
        expected_real_files = validate_video_files(
            real_video_folder,
            args.descriptions_file,
            "testing-videos",
            int(fps),
            include_take_2=True,
        )
        expected_real_files = {
            f.replace("testing-videos", "mask-videos") for f in expected_real_files
        }
        binary_mask_real_folder = ensure_binary_mask_structure(
            input_folder=real_video_folder,
            target_fps=fps,
            expected_files=expected_real_files,
            is_real=True,
            gt_folder=benchmark_data_path,
        )

        # Generate binary masks for generated videos
        validate_generations(input_folder=input_folder)
        rename_generated_videos(
            generated_video_directory=input_folder,
            real_video_directory=real_video_folder,
        )
        expected_generated_files = validate_video_files(
            input_folder, args.descriptions_file, "", int(fps), include_take_2=False
        )

        binary_mask_generated_folder = ensure_binary_mask_structure(
            input_folder=input_folder,
            target_fps=fps,
            expected_files=expected_generated_files,
            is_real=False,
            gt_folder=benchmark_data_path,
        )

        input_folder_name = os.path.basename(input_folder.rstrip("/"))
        csv_file_path = os.path.join(csv_files_folder, f"{input_folder_name}.csv")
        # Check if the CSV is complete
        # Validate all required real video scenarios
        expected_real_scenarios = validate_video_files(
            real_video_folder,
            args.descriptions_file,
            "testing-videos",
            int(fps),
            include_take_2=True,
        )
        # Check if the CSV is complete
        if is_csv_complete(csv_file_path, expected_real_scenarios):
            print(
                f"Skipping calculations for {csv_file_path} as it is already complete."
            )
        else:
            process_videos(
                real_folder=real_video_folder,
                generated_folder=input_folder,
                binary_real_folder=binary_mask_real_folder,
                binary_generated_folder=binary_mask_generated_folder,
                csv_file_path=csv_file_path,
                fps=fps,
                video_time_selection="first",
                n_processes=args.n_process,
            )

    process_directory(csv_files_folder, lazy_integrity=args.lazy_integrity)
    print(f"\nCheck out {csv_files_folder} for saved plots and metrics.")

    print("Thank you for using the Physics-IQ benchmark!")
