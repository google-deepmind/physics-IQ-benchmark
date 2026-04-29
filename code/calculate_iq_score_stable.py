"""Scoring logic for the Physics-IQ benchmark.

Each scenario is evaluated across three camera perspectives. Raw per-perspective
metrics are aggregated into a single Physics-IQ score that normalises model
performance by the physical variance of each scenario — i.e. how much the
ground-truth outcome itself varies across repeated real-world trials.

``IQTable`` is the primary interface: it wraps a per-scenario metrics DataFrame
and exposes the score computations needed for both point-estimate evaluation and
bootstrap resampling.  ``calculate_iq_score_update`` is a deprecated thin wrapper
kept for backwards compatibility.
"""

import numpy as np
import pandas as pd

from calculate_iq_score import parse_list_of_floats, VIEWS


def clip(value, min_value=0.0, max_value=1.0):
    """Clamp *value* to [*min_value*, *max_value*]."""
    return max(min(value, max_value), min_value)


ORIG_SCORE_KEY = "final_score_orig"
VERIFIED_SCORE_KEY = "final_score_stable"
METRIC_KEYS = ["spatial","spatiotemporal", "weighted_spatial", "mse"]
SCORES_LIST = [f"score_{metric}" for metric in METRIC_KEYS]
VARIANCE_KEYS = [f"physical_variance_{metric}" for metric in METRIC_KEYS]



class IQTable():
    """Per-scenario metrics table with Physics-IQ score computation.

    Wraps a DataFrame where each row is one test scenario.  On construction the
    raw per-perspective columns are collapsed into cross-view means so that all
    subsequent score methods operate on a single value per row.

    Scoring formula
    ---------------
    IOU-based metrics (spatial, weighted-spatial, spatiotemporal) are divided by
    their physical variance; MSE is subtracted.  Physical variance is the empirical
    variance of the ground-truth outcome across repeated real-world trials of the
    same scenario, and serves as a difficulty-normalisation term.

    Two final-score variants
    ------------------------
    orig   — aggregate the three IOU scores and MSE score, then clip the total.
    stable — clip each component to [0, 1] before aggregating, so that a single
             extreme component cannot pull the total out of range.

    The ``stable`` variant is the primary reported score for the NeurIPS submission.
    """

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
        self.df = df.copy()  # own our data so callers can't mutate it under us
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
    
    @property
    def variance_keys(self) -> list[str]:
        return [self.variance_spatial_key, self.variance_weighted_spatial_key, self.variance_spatiotemporal_iou_key, self.variance_mse_key]

    def __len__(self):
        return len(self.df)
    
    def _get_list_column_mean(self, metric_key):
        # List columns hold per-frame sequences; concatenate across views before averaging
        # so that every frame contributes equally regardless of view.
        assert metric_key in self.get_list_keys(), f"Invalid metric key: {metric_key}"
        return self.df.apply(
          lambda row: np.mean(np.concatenate([row[f"{metric_key}_{view}"] for view in VIEWS])),
            axis=1
        )

    def get_full_df(self):
        """Return a copy of the internal DataFrame with metadata columns appended."""
        out = self.df.copy()
        for m, k in self.metadata.items():
            out[m] = k
        return out

    def _get_scalar_column_mean(self, metric_key):
        assert metric_key not in self.get_list_keys(), f"Invalid metric key: {metric_key}"
        return self.df[[f"{metric_key}_{view}" for view in VIEWS]].mean(axis=1)

    def get_metric_mean(self, metric_key):
        """Dataset-wide mean of a single (already aggregated) metric column."""
        return self.df[metric_key].mean()

    def get_score(self, metric_key):
        """Return the variance-normalised score for one metric.

        IOU metrics are divided by their physical variance (higher model IOU
        relative to the natural scene variance → higher score).  MSE is subtracted
        because a lower model MSE relative to the physical variance is better.
        """
        _score_map = {
            self.spatial_iou_key: (self.spatial_iou_key, self.variance_spatial_key, "divide"),
            self.weighted_spatial_iou_key: (self.weighted_spatial_iou_key, self.variance_weighted_spatial_key, "divide"),
            self.spatiotemporal_iou_key: (self.spatiotemporal_iou_key, self.variance_spatiotemporal_iou_key, "divide"),
            self.mse_key: (self.mse_key, self.variance_mse_key, "subtract"),
        }
        if metric_key not in _score_map:
            raise ValueError(f"Invalid metric key: {metric_key}")
        metric, variance, op = _score_map[metric_key]
        m, v = self.get_metric_mean(metric), self.get_metric_mean(variance)
        return m / v if op == "divide" else m - v
    
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
            "variance_mse": self.get_metric_mean(self.variance_mse_key),
            "variance_spatiotemporal_iou": self.get_metric_mean(self.variance_spatiotemporal_iou_key),
            "variance_spatial": self.get_metric_mean(self.variance_spatial_key),
            "variance_weighted_spatial": self.get_metric_mean(self.variance_weighted_spatial_key),
        }
        out_dict.update(self.metadata)

        return out_dict
    
    
    @classmethod
    def get_list_keys(cls):
        """Metric keys whose CSV columns contain per-frame lists (spatiotemporal IOU, MSE)."""
        return [cls.spatiotemporal_iou_key, cls.mse_key, cls.variance_spatiotemporal_iou_key, cls.variance_mse_key]

    @classmethod
    def get_scalar_keys(cls):
        """Metric keys whose CSV columns contain a single float per scenario (spatial IOU)."""
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
    import warnings
    warnings.warn(
        "calculate_iq_score_update() is deprecated; use IQTable.from_csv(path).get_output_dict() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return IQTable.from_csv(file_path).get_output_dict()