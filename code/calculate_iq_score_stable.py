import numpy as np
import pandas as pd

from .calculate_iq_score import parse_list_of_floats, VIEWS


def clip(value, min_value=0.0, max_value=1.0):
    """
    Clip a value to be within the specified range.
    """
    return max(min(value, max_value), min_value)


def calculate_iq_score_update(file_path: str) -> tuple[float, float]:
  """
  Calculate the Physics IQ score and physical variance for a given CSV file.

  Args:
    file_path: Path to the CSV file containing metrics.

  Returns:
    A tuple containing the final score and physical variance (both rounded to 4 decimal places).
  """

  df = pd.read_csv(file_path)

  list_columns = [
    f"v1_mse_{view}" for view in VIEWS
  ] + [
    f"spatiotemporal_iou_v1_{view}" for view in VIEWS
  ]

  for col in list_columns:
    df[col] = df[col].apply(parse_list_of_floats)


  total_sum_v1_mse = df.apply(
      lambda row: np.mean(np.concatenate([row[f"v1_mse_{view}"] for view in VIEWS])),
      axis=1
  ).mean()
  
  total_sum_spatiotemporal_iou_v1 = df.apply(
    lambda row: np.mean(np.concatenate([row[f"spatiotemporal_iou_v1_{view}"] for view in VIEWS])),
    axis=1
  ).mean()

  # Aggregate across views for spatial and weighted_spatial IOU
  total_sum_spatial_iou = df[[f"spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()

  total_sum_weighted_spatial_iou = df[[f"weighted_spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()


  # Compute variance across views
  physical_variance_mse = np.mean([
    df[f"variance_mse_{view}"].apply(parse_list_of_floats).explode().mean()
    for view in VIEWS
    if f"variance_mse_{view}" in df.columns
  ])
  
  physical_variance_spatiotemporal_iou = np.mean([
    df[f"variance_spatiotemporal_iou_{view}"].apply(parse_list_of_floats).explode().mean()
    for view in VIEWS
    if f"variance_spatiotemporal_iou_{view}" in df.columns
  ])
  
  physical_variance_spatial = np.mean([
    df[f"variance_spatial_{view}"].mean()
    for view in VIEWS
    if f"variance_spatial_{view}" in df.columns
  ])
  
  physical_variance_weighted_spatial = np.mean([
    df[f"variance_weighted_spatial_{view}"].mean()
    for view in VIEWS
    if f"variance_weighted_spatial_{view}" in df.columns
  ])

  physical_variance_all_metrics = \
    physical_variance_spatiotemporal_iou + physical_variance_spatial + \
    physical_variance_weighted_spatial - physical_variance_mse
  
  score_spatiotemporal = total_sum_spatiotemporal_iou_v1 / physical_variance_spatiotemporal_iou
  score_spatial = total_sum_spatial_iou / physical_variance_spatial
  score_weighted_spatial = total_sum_weighted_spatial_iou / physical_variance_weighted_spatial
  score_mse = total_sum_v1_mse - physical_variance_mse

  final_score_raw = (score_spatiotemporal + score_spatial + score_weighted_spatial) / 3 - score_mse
  final_score_stable = (clip(score_spatiotemporal), clip(score_spatial), clip(score_weighted_spatial)/3 - clip(score_mse))
  


  final_score_orig = clip(final_score_raw)

  out_dict = {
    "score_spatiotemporal": score_spatiotemporal,
    "score_spatial": score_spatial,
    "score_weighted_spatial": score_weighted_spatial,
    "score_mse": score_mse,
    "final_score_raw": final_score_raw,
    "final_score_stable": final_score_stable,
    "final_score_orig": final_score_orig,
    "physical_variance_mse": physical_variance_mse,
    "physical_variance_spatiotemporal_iou": physical_variance_spatiotemporal_iou,
    "physical_variance_spatial": physical_variance_spatial,
    "physical_variance_weighted_spatial": physical_variance_weighted_spatial,
    "physical_variance_all_metrics": physical_variance_all_metrics,
    "total_sum_v1_mse": total_sum_v1_mse,
    "total_sum_spatiotemporal_iou_v1": total_sum_spatiotemporal_iou_v1,
    "total_sum_spatial_iou": total_sum_spatial_iou,
    "total_sum_weighted_spatial_iou": total_sum_weighted_spatial_iou
  }

  

  return out_dict