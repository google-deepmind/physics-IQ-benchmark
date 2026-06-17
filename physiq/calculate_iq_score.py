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

import re

import numpy as np
import pandas as pd

VIEWS = ["perspective-left", "perspective-center", "perspective-right"]


def parse_list_of_floats(value):
    """
    Parse a string or list representing a list of floats and round each number to 4 decimal places.
    Raises ValueError if no valid floats are found.
    """

    if isinstance(value, str):
        if not (value.startswith("[") and value.endswith("]")):
            raise ValueError("Invalid string format for list of floats.")
        try:
            # Extract numbers from the string and convert to floats
            parsed_floats = [
                round(float(x), 4)
                for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value)
            ]
            if not parsed_floats:
                raise ValueError("No valid floats found in the input string.")
            return parsed_floats
        except ValueError:
            raise ValueError("Failed to parse floats from the input string.")

    elif isinstance(value, list):
        if all(isinstance(x, (int, float)) for x in value):
            parsed_floats = [round(float(x), 4) for x in value]
            if not parsed_floats:
                raise ValueError(
                    "Input list is empty or contains no valid numeric values."
                )
            return parsed_floats
        else:
            raise ValueError("List contains non-numeric values.")

    raise TypeError(
        "Input must be a string representing a list of floats or a list of numbers."
    )


def calculate_iq_score(file_path: str) -> tuple[float, float]:
    """
    Calculate the Physics IQ score and physical variance for a given CSV file.

    Args:
      file_path: Path to the CSV file containing metrics.

    Returns:
      A tuple containing the final score and physical variance (both rounded to 4 decimal places).
    """

    df = pd.read_csv(file_path)

    list_columns = [f"v1_mse_{view}" for view in VIEWS] + [
        f"spatiotemporal_iou_v1_{view}" for view in VIEWS
    ]

    for col in list_columns:
        df[col] = df[col].apply(parse_list_of_floats)

    # Calculate sum across views for MSE and IOU
    df["sum_v1_mse"] = df[[f"v1_mse_{view}" for view in VIEWS]].apply(
        lambda x: round(sum(sum(val) for val in x), 4), axis=1
    )

    df["sum_spatiotemporal_iou_v1"] = df[
        [f"spatiotemporal_iou_v1_{view}" for view in VIEWS]
    ].apply(lambda x: round(sum(sum(val) for val in x), 4), axis=1)

    total_sum_v1_mse = round(df["sum_v1_mse"].sum(), 4)
    total_sum_spatiotemporal_iou_v1 = round(df["sum_spatiotemporal_iou_v1"].sum(), 4)

    list_length = len(df[f"v1_mse_perspective-left"].iloc[0]) if len(df) > 0 else 0

    total_sum_v1_mse = df.apply(
        lambda row: np.mean(np.concatenate([row[f"v1_mse_{view}"] for view in VIEWS])),
        axis=1,
    ).mean()

    total_sum_spatiotemporal_iou_v1 = df.apply(
        lambda row: np.mean(
            np.concatenate([row[f"spatiotemporal_iou_v1_{view}"] for view in VIEWS])
        ),
        axis=1,
    ).mean()

    # Aggregate across views for spatial and weighted_spatial IOU
    total_sum_spatial_iou = (
        df[[f"spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()
    )

    total_sum_weighted_spatial_iou = (
        df[[f"weighted_spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()
    )

    final_score = round(
        total_sum_spatial_iou
        + total_sum_weighted_spatial_iou
        + total_sum_spatiotemporal_iou_v1
        - total_sum_v1_mse,
        4,
    )

    # Compute variance across views
    physical_variance_mse = round(
        np.mean(
            [
                df[f"variance_mse_{view}"].apply(parse_list_of_floats).explode().mean()
                for view in VIEWS
                if f"variance_mse_{view}" in df.columns
            ]
        ),
        4,
    )

    physical_variance_spatiotemporal_iou = round(
        np.mean(
            [
                df[f"variance_spatiotemporal_iou_{view}"]
                .apply(parse_list_of_floats)
                .explode()
                .mean()
                for view in VIEWS
                if f"variance_spatiotemporal_iou_{view}" in df.columns
            ]
        ),
        4,
    )

    physical_variance_spatial = round(
        np.mean(
            [
                df[f"variance_spatial_{view}"].mean()
                for view in VIEWS
                if f"variance_spatial_{view}" in df.columns
            ]
        ),
        5,
    )

    physical_variance_weighted_spatial = round(
        np.mean(
            [
                df[f"variance_weighted_spatial_{view}"].mean()
                for view in VIEWS
                if f"variance_weighted_spatial_{view}" in df.columns
            ]
        ),
        4,
    )

    physical_variance_all_metrics = round(
        physical_variance_spatiotemporal_iou
        + physical_variance_spatial
        + physical_variance_weighted_spatial
        - physical_variance_mse,
        4,
    )
    final_score = (
        (
            (total_sum_spatiotemporal_iou_v1 / physical_variance_spatiotemporal_iou)
            + (total_sum_spatial_iou / physical_variance_spatial)
            + (total_sum_weighted_spatial_iou / physical_variance_weighted_spatial)
        )
        / 3
    ) - (total_sum_v1_mse - physical_variance_mse)

    final_score *= 100
    final_score = round(max(min(final_score, 100.0), 0.0), 2)

    return final_score, physical_variance_all_metrics
