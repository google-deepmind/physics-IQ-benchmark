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
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_settings import model_to_plotting_name, model_to_color

def parse_list_of_floats(
    value: str | list[float] | list[int]
) -> list[float]:
  """
  Parse a string or list representing a list of floats.

  Args:
    value: A string or list representing a list of floats.

  Returns:
    A list of floats.
  """
  try:
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
      return [
        float(x)
        for x in re.findall(
          r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value
        )
      ]
    elif isinstance(value, list):
      return [float(x) for x in value if isinstance(x, (int, float))]
    return []
  except (ValueError, TypeError):
    return []


def calculate_iq_score(file_path: str) -> tuple[float, float]:
  """
  Calculate the Physics IQ score and physical variance for a given CSV file.

  Args:
    file_path: Path to the CSV file containing metrics.

  Returns:
    A tuple containing the final score and physical variance.
  """
  df = pd.read_csv(file_path)

  list_columns = [
    "v1_mse_perspective-left", "v1_mse_perspective-center", "v1_mse_perspective-right",
    "iou_v1_perspective-left", "iou_v1_perspective-center", "iou_v1_perspective-right"
  ]
  for col in list_columns:
    df[col] = df[col].apply(parse_list_of_floats)

  # Calculate sum across views for MSE and IOU
  df["sum_v1_mse"] = df[
    ["v1_mse_perspective-left", "v1_mse_perspective-center", "v1_mse_perspective-right"]
  ].apply(lambda x: sum(sum(val) for val in x), axis=1)
  df["sum_iou_v1"] = df[
    ["iou_v1_perspective-left", "iou_v1_perspective-center", "iou_v1_perspective-right"]
  ].apply(lambda x: sum(sum(val) for val in x), axis=1)

  total_sum_v1_mse = df["sum_v1_mse"].sum()
  total_sum_iou_v1 = df["sum_iou_v1"].sum()

  list_length = (
    len(df["v1_mse_perspective-left"].iloc[0]) if len(df) > 0 else 0
  )  # Assuming uniform list length

  total_sum_v1_mse /= (3 * 66 * list_length)
  total_sum_iou_v1 /= (3 * 66 * list_length)

  total_sum_spatial_iou = df["spatial_iou_v1"].sum() / 66
  total_sum_spatiotemporal_iou = df["spatiotemporal_iou_v1"].sum() / 66

  final_score = (
    total_sum_spatial_iou + total_sum_spatiotemporal_iou +
    total_sum_iou_v1 - total_sum_v1_mse
  )

  physical_variance_mse = np.mean([
    df[f"variance_mse_{view}"].apply(parse_list_of_floats).explode().mean()
    for view in ["perspective-left", "perspective-center", "perspective-right"] if f"variance_mse_{view}" in df.columns
  ])
  physical_variance_iou = np.mean([
    df[f"variance_iou_{view}"].apply(parse_list_of_floats).explode().mean()
    for view in ["perspective-left", "perspective-center", "perspective-right"] if f"variance_iou_{view}" in df.columns
  ])
  physical_variance_spatial = np.mean([
    df[f"variance_spatial_{view}"].mean()
    for view in ["perspective-left", "perspective-center", "perspective-right"] if f"variance_spatial_{view}" in df.columns
  ])
  physical_variance_spatiotemporal = np.mean([
    df[f"variance_spatiotemporal_{view}"].mean()
    for view in ["perspective-left", "perspective-center", "perspective-right"]
    if f"variance_spatiotemporal_{view}" in df.columns
  ])

  physical_variance_all_metrics = (
    physical_variance_iou + physical_variance_spatial +
    physical_variance_spatiotemporal - physical_variance_mse
  )

  final_score = (
    (
        (total_sum_iou_v1 / physical_variance_iou) +
        (total_sum_spatial_iou / physical_variance_spatial) +
        (total_sum_spatiotemporal_iou / physical_variance_spatiotemporal)
    ) / 3
  ) - (total_sum_v1_mse - physical_variance_mse)

  final_score *= 100

  # Make sure result is within [0, 100]
  final_score = max(final_score, 0.0)
  final_score = min(final_score, 100.0)

  return round(final_score, 2), physical_variance_all_metrics


def process_directory(directory_path: str) -> None:
  """
  Process all CSV files in a directory to compute Physics IQ scores
  and generate a bar plot.

  Args:
    directory_path: Path to the directory containing CSV files.

  Returns:
    None
  """
  model_scores = {}
  csv_files = [
    f for f in sorted(os.listdir(directory_path)) if f.endswith(".csv")
  ]

  for csv_file in csv_files:
    file_path = os.path.join(directory_path, csv_file)
    print(f"Processing {csv_file}...")

    model_name = os.path.splitext(csv_file)[0]
    final_score, physical_variance = calculate_iq_score(file_path)

    print(f"Physics-IQ score for {csv_file.replace('.csv', '')}: {final_score}")
    print("-" * 50)

    model_scores[model_name] = final_score

  sorted_items = sorted(
    model_scores.items(), key=lambda x: x[1], reverse=True
  )
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
      fontsize=10
    )

  plt.axhline(y=100, color="darkgrey", linestyle="--", linewidth=2)

  midpoint = (len(model_names) - 1) / 2.0
  plt.text(
    midpoint, 102, "Physical Variance",
    ha="center", va="bottom", color="black", fontweight="bold"
  )

  plt.xticks(rotation=45, ha="right")

  ax = plt.gca()
  ax.spines["right"].set_visible(False)
  ax.spines["top"].set_visible(False)

  plt.xlabel("")
  plt.ylabel("Physics-IQ score\n(higher=better)")
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
  plt.tight_layout()
  plt.savefig(os.path.join(directory_path, 'physics_IQ_score_barplot.pdf'))

