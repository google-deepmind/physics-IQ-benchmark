import numpy as np
import pandas as pd

from calculate_iq_score import parse_list_of_floats, VIEWS

def clip(value, min_value=0.0, max_value=1.0):
    """
    Clip a value to be within the specified range.
    """
    return max(min(value, max_value), min_value)


ORIG_SCORE_KEY = "final_score_orig"
VERIFIED_SCORE_KEY = "final_score_stable"
METRIC_KEYS = ["spatial","spatiotemporal", "weighted_spatial", "mse"]
SCORES_LIST = [f"score_{metric}" for metric in METRIC_KEYS]
VARIANCE_KEYS = [f"physical_variance_{metric}" for metric in METRIC_KEYS]



class IQTable():
    spatial_iou_key = "spatial_iou_v1"
    weighted_spatial_iou_key = "weighted_spatial_iou_v1"
    spatiotemporal_iou_key = "spatiotemporal_iou_v1"
    mse_key = "v1_mse"
    variance_spatial_key = "variance_spatial"
    variance_weighted_spatial_key = "variance_weighted_spatial"
    variance_spatiotemporal_iou_key = "variance_spatiotemporal_iou"
    variance_mse_key = "variance_mse"
    views = VIEWS


    def __init__(self, df: pd.DataFrame, metadata: dict = None):
        self.df = df
        self.metadata = metadata or {}
        for col in self.get_list_keys():
            self.df[col] = self._get_list_column_mean(col)

        for col in self.get_scalar_keys():
            self.df[col] = self._get_scalar_column_mean(col)

    @property
    def spatial_iou_cols(self):
        return [f"{self.spatial_iou_key}_{view}" for view in self.views]
    
    @property
    def weighted_spatial_iou_cols(self):
        return [f"{self.weighted_spatial_iou_key}_{view}" for view in self.views]
    
    @property
    def spatiotemporal_iou_cols(self):
        return [f"{self.spatiotemporal_iou_key}_{view}" for view in self.views]
    
    @property
    def mse_cols(self):
        return [f"{self.mse_key}_{view}" for view in self.views]
    
    @property
    def variance_spatial_cols(self):
        return [f"{self.variance_spatial_key}_{view}" for view in self.views]
    
    @property
    def variance_weighted_spatial_cols(self):
        return [f"{self.variance_weighted_spatial_key}_{view}" for view in self.views]
    
    @property
    def variance_spatiotemporal_iou_cols(self):
        return [f"{self.variance_spatiotemporal_iou_key}_{view}" for view in self.views]
    
    @property
    def variance_mse_cols(self):
        return [f"{self.variance_mse_key}_{view}" for view in self.views]

    def __len__(self):
        return len(self.df)
    
    def _get_list_column_mean(self, metric_key):
        assert metric_key in self.get_list_keys(), f"Invalid metric key: {metric_key}"
        return self.df.apply(
          lambda row: np.mean(np.concatenate([row[f"{metric_key}_{view}"] for view in VIEWS])),
            axis=1
        )
    
    def get_full_df(self):
        out = self.df.copy()
        for m, k in self.metadata.items():
            out[m] = k
        return out
    
    def _get_scalar_column_mean(self, metric_key):
        assert metric_key not in self.get_list_keys(), f"Invalid metric key: {metric_key}"
        return self.df[[f"{metric_key}_{view}" for view in VIEWS]].mean(axis=1)
    
    def get_metric_mean(self, metric_key):
        return self.df[metric_key].mean()
    
    def get_score(self, metric_key):
        if metric_key == self.spatial_iou_key:
            return self.get_metric_mean(self.spatial_iou_key) / self.get_metric_mean(self.variance_spatial_key)
        elif metric_key == self.weighted_spatial_iou_key:
            return self.get_metric_mean(self.weighted_spatial_iou_key) / self.get_metric_mean(self.variance_weighted_spatial_key)
        elif metric_key == self.spatiotemporal_iou_key:
            return self.get_metric_mean(self.spatiotemporal_iou_key) / self.get_metric_mean(self.variance_spatiotemporal_iou_key)
        elif metric_key == self.mse_key:
            return self.get_metric_mean(self.mse_key) - self.get_metric_mean(self.variance_mse_key)
        else:
            raise ValueError(f"Invalid metric key: {metric_key}")
    
    def compute_final_score_orig_raw(self):
        score_spatiotemporal = self.get_score(self.spatiotemporal_iou_key)
        score_spatial = self.get_score(self.spatial_iou_key)
        score_weighted_spatial = self.get_score(self.weighted_spatial_iou_key)
        score_mse = self.get_score(self.mse_key)
        final_score_raw = (score_spatiotemporal + score_spatial + score_weighted_spatial) / 3 - score_mse
        return final_score_raw 
    
    def compute_final_score_orig(self):
        return clip(self.compute_final_score_orig_raw())
    
    def compute_final_score_stable(self):
        score_spatiotemporal = clip(self.get_score(self.spatiotemporal_iou_key))
        score_spatial = clip(self.get_score(self.spatial_iou_key))
        score_weighted_spatial = clip(self.get_score(self.weighted_spatial_iou_key))
        score_mse = clip(self.get_score(self.mse_key))
        final_score_stable = clip((score_spatiotemporal + score_spatial + score_weighted_spatial) / 3 - score_mse)
        return final_score_stable
    
    def get_output_dict(self):
        out_dict = {
            "score_spatiotemporal": self.get_score(self.spatiotemporal_iou_key),
            "score_spatial": self.get_score(self.spatial_iou_key),
            "score_weighted_spatial": self.get_score(self.weighted_spatial_iou_key),
            "score_mse": self.get_score(self.mse_key),
            "final_score_raw": self.compute_final_score_orig_raw(),
            "final_score_stable": self.compute_final_score_stable(),
            "final_score_orig": self.compute_final_score_orig(),
            "physical_variance_mse": self.get_metric_mean(self.variance_mse_key),
            "physical_variance_spatiotemporal_iou": self.get_metric_mean(self.variance_spatiotemporal_iou_key),
            "physical_variance_spatial": self.get_metric_mean(self.variance_spatial_key),
            "physical_variance_weighted_spatial": self.get_metric_mean(self.variance_weighted_spatial_key),
        }
        out_dict.update(self.metadata)

        return out_dict
    
    
    @classmethod
    def get_list_keys(cls):
        return [cls.spatiotemporal_iou_key, cls.mse_key, cls.variance_spatiotemporal_iou_key, cls.variance_mse_key]
    
    @classmethod
    def get_scalar_keys(cls):
        return [cls.spatial_iou_key, cls.weighted_spatial_iou_key, cls.variance_spatial_key, cls.variance_weighted_spatial_key]

    @classmethod
    def get_list_columns(cls):
        return [f"{metric}_{view}" for metric in cls.get_list_keys() for view in cls.views]
    
    @classmethod
    def from_csv(cls, file_path: str, *args, **kwargs):
        df = pd.read_csv(file_path)
        list_columns = cls.get_list_columns()
        for col in list_columns:
            df[col] = df[col].apply(parse_list_of_floats)
        return cls(df, *args, **kwargs)


def calculate_iq_score_update(file_path: str) -> dict:
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
    final_score_stable = (clip(score_spatiotemporal)+ clip(score_spatial) + clip(score_weighted_spatial))/3 - clip(score_mse)
  


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