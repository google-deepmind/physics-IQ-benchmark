# Copyright 2026 DeepMind Technologies Limited
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
"""Scoring logic for the Physics-IQ benchmark.

Each scenario is evaluated across three camera perspectives. Raw per-perspective
metrics are aggregated into a single Physics-IQ score that normalises model
performance by the physical variance of each scenario — i.e. how much the
ground-truth outcome itself varies across repeated real-world trials.

``IQTable`` is the primary interface: it wraps a per-scenario metrics DataFrame
and exposes the score computations needed for both point-estimate evaluation and
bootstrap resampling.
"""

from functools import cached_property

import joblib
import numpy as np
import pandas as pd

from physiq.calculate_iq_score import VIEWS, parse_list_of_floats


def clip(value, min_value=0.0, max_value=1.0):
    """Clamp *value* to [*min_value*, *max_value*]."""
    return np.clip(value, min_value, max_value)


ORIG_SCORE_KEY = "final_score_orig"
VERIFIED_SCORE_KEY = "final_score_view"
METRIC_KEYS = ["spatial", "spatiotemporal", "weighted_spatial", "mse"]

SCORES_LIST = [f"score_{metric}" for metric in METRIC_KEYS]
VERIFIED_SCORES_LIST = [f"score_{metric}_view" for metric in METRIC_KEYS]
VARIANCE_KEYS = [f"physical_variance_{metric}" for metric in METRIC_KEYS]


class IQTable:
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
    ratio_eps = 1e-8  # small constant to prevent divide-by-zero in ratio computations

    def __init__(
        self, df: pd.DataFrame, metadata: dict = None, lazy_integrity: bool = False
    ):
        if len(df) == 0:
            raise ValueError(
                "IQTable requires at least one scenario row; the DataFrame is empty."
            )
        self.df = df.copy()  # own our data so callers can't mutate it under us
        self.metadata = metadata or {}

        # add mean values for each view self.df with names like "{col}_mean_{view}"
        for col in self.get_list_keys():
            df_temp = self._get_list_column_mean_by_view(col)
            df_temp = self._rename_list_column_mean_by_view(df_temp)
            for col in df_temp.columns:
                self.df[col] = df_temp[col]

        for col in self.get_list_keys():
            self.df[col] = self._get_list_column_mean(col)

        for col in self.get_scalar_keys():
            self.df[col] = self._get_scalar_column_mean(col)

        self._lazy_integrity = lazy_integrity
        if self._lazy_integrity:
            self._df_hash = None
        else:
            self._df_hash = joblib.hash(self.df)
        self._score_cache: dict[str, float] = {}

    def _verify_df_integrity(self) -> None:
        """Raise if self.df was mutated after construction."""
        if self._lazy_integrity:
            return None
        current = joblib.hash(self.df)
        if current != self._df_hash:
            raise RuntimeError(
                "IQTable.df was mutated after construction; cached scores are invalid."
            )

    @cached_property
    def spatial_iou_cols(self):
        return [f"{self.spatial_iou_key}_{view}" for view in self.views]

    @cached_property
    def weighted_spatial_iou_cols(self):
        return [f"{self.weighted_spatial_iou_key}_{view}" for view in self.views]

    @cached_property
    def spatiotemporal_iou_cols(self):
        return [f"{self.spatiotemporal_iou_key}_{view}" for view in self.views]

    @cached_property
    def mse_cols(self):
        return [f"{self.mse_key}_{view}" for view in self.views]

    @cached_property
    def variance_spatial_cols(self):
        return [f"{self.variance_spatial_key}_{view}" for view in self.views]

    @cached_property
    def variance_weighted_spatial_cols(self):
        return [f"{self.variance_weighted_spatial_key}_{view}" for view in self.views]

    @cached_property
    def variance_spatiotemporal_iou_cols(self):
        return [f"{self.variance_spatiotemporal_iou_key}_{view}" for view in self.views]

    @cached_property
    def variance_mse_cols(self):
        return [f"{self.variance_mse_key}_{view}" for view in self.views]

    @cached_property
    def variance_keys(self) -> list[str]:
        return [
            self.variance_spatial_key,
            self.variance_weighted_spatial_key,
            self.variance_spatiotemporal_iou_key,
            self.variance_mse_key,
        ]

    @cached_property
    def metric_keys(self) -> list[str]:
        return [
            self.spatial_iou_key,
            self.weighted_spatial_iou_key,
            self.spatiotemporal_iou_key,
            self.mse_key,
        ]

    @cached_property
    def list_column_mean_by_view_dict(self) -> dict[str, str]:
        metric_cols_map = {
            f"{metric_key}_{view}": f"{metric_key}_mean_{view}"
            for view in self.views
            for metric_key in self.get_list_keys()
        }
        return metric_cols_map

    def compute_metric_ratio(self, metric_key) -> pd.Series:
        _score_map = {
            self.spatial_iou_key: (self.spatial_iou_key, self.variance_spatial_key),
            self.weighted_spatial_iou_key: (
                self.weighted_spatial_iou_key,
                self.variance_weighted_spatial_key,
            ),
            self.spatiotemporal_iou_key: (
                self.spatiotemporal_iou_key,
                self.variance_spatiotemporal_iou_key,
            ),
            self.mse_key: (self.mse_key, self.variance_mse_key),
        }
        if metric_key not in _score_map:
            raise ValueError(f"Invalid metric key: {metric_key}")
        return self.df[_score_map[metric_key][0]] / (
            self.df[_score_map[metric_key][1]] + self.ratio_eps
        )

    def compute_metric_ratio_by_view(self, metric_key) -> np.ndarray:
        """Compute the metric ratio for each view separately, returning an array of shape (n_scenarios, n_views)."""
        get_mean_dict = self.list_column_mean_by_view_dict
        mapping_fct = lambda x: [get_mean_dict[view_col] for view_col in x]
        _score_map = {
            self.spatial_iou_key: (self.spatial_iou_cols, self.variance_spatial_cols),
            self.weighted_spatial_iou_key: (
                self.weighted_spatial_iou_cols,
                self.variance_weighted_spatial_cols,
            ),
            self.spatiotemporal_iou_key: (
                mapping_fct(self.spatiotemporal_iou_cols),
                mapping_fct(self.variance_spatiotemporal_iou_cols),
            ),
            self.mse_key: (
                mapping_fct(self.mse_cols),
                mapping_fct(self.variance_mse_cols),
            ),
        }
        if metric_key not in _score_map:
            raise ValueError(f"Invalid metric key: {metric_key}")

        metric_cols, variance_cols = _score_map[metric_key]
        return self.df[metric_cols].to_numpy() / (
            self.df[variance_cols].to_numpy() + self.ratio_eps
        )

    def compute_metric_scores_scenario(self, metric_key) -> pd.Series:
        """Compute the metric score for a single metric key, returning a Series of shape (n_scenarios,)."""
        ratio = self.compute_metric_ratio(metric_key)
        if metric_key == self.mse_key:
            return ratio**-1  # higher is better for MSE, so invert the ratio
        else:
            return ratio  # for IOU metrics, higher is already better

    def compute_scores_scenario(self):
        out = pd.DataFrame()
        metric_score_keys = [metric_key + "_score" for metric_key in self.metric_keys]

        for metric_key, metric_score_key in zip(self.metric_keys, metric_score_keys):
            out[metric_score_key] = self.compute_metric_scores_scenario(metric_key)
        self.compute_means_by_row_inplace(out, metric_score_keys)
        return out

    @classmethod
    def compute_means_by_row_inplace(
        cls, df: pd.DataFrame, cols: list[str]
    ) -> pd.DataFrame:
        df["final_arithmetic"] = clip(df[cols].to_numpy()).mean(axis=1)
        df["final_geometric"] = clip(df[cols].to_numpy()).prod(axis=1) ** (
            1 / len(cols)
        )
        df["final_harmonic"] = len(cols) / (
            1 / (clip(df[cols].to_numpy() + cls.ratio_eps))
        ).sum(axis=1)

    def compute_metric_scores_scenario_by_view(self, metric_key) -> np.ndarray:
        """Compute the metric score for a single metric key by view, returning an array of shape (n_scenarios, n_views)."""
        ratio_by_view = self.compute_metric_ratio_by_view(metric_key)
        if metric_key == self.mse_key:
            return ratio_by_view**-1  # higher is better for MSE, so invert the ratio
        else:
            return ratio_by_view  # for IOU metrics, higher is already better

    def compute_scores_scenario_by_view(self) -> pd.DataFrame:
        """Compute the metric scores for all metric keys by view, returning a DataFrame of shape (n_scenarios * n_views, n_metrics)."""
        out = pd.DataFrame()
        metric_score_keys = [metric_key + "_score" for metric_key in self.metric_keys]
        for metric_key, metric_score_key in zip(self.metric_keys, metric_score_keys):
            out[metric_score_key] = self.compute_metric_scores_scenario_by_view(
                metric_key
            ).flatten()
        self.compute_means_by_row_inplace(out, metric_score_keys)
        return out

    def __getitem__(self, index):
        return self.df.iloc[index]

    def __len__(self):
        return len(self.df)

    def _get_list_column_mean_by_view(self, metric_key):
        # For list columns, compute the mean per view first, then average across views.
        # This ensures that each frame contributes equally to the final score regardless of view.
        assert metric_key in self.get_list_keys(), f"Invalid metric key: {metric_key}"
        metric_cols = [f"{metric_key}_{view}" for view in self.views]
        return pd.DataFrame(
            {col: np.array(self.df[col].tolist()).mean(axis=1) for col in metric_cols},
            index=self.df.index,
        )

    def _rename_list_column_mean_by_view(self, df_rename: pd.DataFrame) -> pd.DataFrame:
        """Rename columns of the per-view mean DataFrame to have a consistent naming scheme."""
        # metric_cols_map = {
        #     f"{metric_key}_{view}": f"{metric_key}_mean_{view}" for view in self.views
        # }
        metric_cols_map = self.list_column_mean_by_view_dict
        df_rename = df_rename.rename(metric_cols_map, axis="columns")
        return df_rename

    def _get_list_column_mean(self, metric_key):
        """Compute the mean of a list column across all frames and views."""
        # List columns hold per-frame sequences; concatenate across views before averaging
        # so that every frame contributes equally regardless of view.
        assert metric_key in self.get_list_keys(), f"Invalid metric key: {metric_key}"
        arrays = [np.array(self.df[f"{metric_key}_{view}"].tolist()) for view in VIEWS]
        stacked = np.concatenate(arrays, axis=1)  # (n_scenarios, n_frames * n_views)
        return pd.Series(stacked.mean(axis=1), index=self.df.index)

    def get_full_df(self):
        """Return a copy of the internal DataFrame with metadata columns appended."""
        out = self.df.copy()
        for m, k in self.metadata.items():
            out[m] = k
        return out

    def _get_scalar_column_mean(self, metric_key):
        """Scalar columns already have one value per scenario; just average across views."""
        assert (
            metric_key not in self.get_list_keys()
        ), f"Invalid metric key: {metric_key}"
        return self.df[[f"{metric_key}_{view}" for view in VIEWS]].mean(axis=1)

    def get_metric_mean(self, metric_key):
        """Dataset-wide mean of a single (already aggregated) metric column."""
        return self.df[metric_key].mean()

    def get_score(self, metric_key):
        """Return the variance-normalised score for one metric.

        IOU metrics are divided by their physical variance (higher model IOU
        relative to the natural scene variance → higher score).  MSE is subtracted
        because a lower model MSE relative to the physical variance is better.

        Results are cached per instance. The first call verifies that self.df has
        not been mutated since construction.
        """
        if not self._score_cache:
            self._verify_df_integrity()
        if metric_key in self._score_cache:
            return self._score_cache[metric_key]
        _score_map = {
            self.spatial_iou_key: (
                self.spatial_iou_key,
                self.variance_spatial_key,
                "divide",
            ),
            self.weighted_spatial_iou_key: (
                self.weighted_spatial_iou_key,
                self.variance_weighted_spatial_key,
                "divide",
            ),
            self.spatiotemporal_iou_key: (
                self.spatiotemporal_iou_key,
                self.variance_spatiotemporal_iou_key,
                "divide",
            ),
            self.mse_key: (self.mse_key, self.variance_mse_key, "subtract"),
        }
        if metric_key not in _score_map:
            raise ValueError(f"Invalid metric key: {metric_key}")
        metric, variance, op = _score_map[metric_key]
        m, v = self.get_metric_mean(metric), self.get_metric_mean(variance)
        result = m / v if op == "divide" else m - v
        self._score_cache[metric_key] = result
        return result

    def compute_final_score_orig_raw(self) -> float:
        """Returns the raw original physics-iq score without clipping."""
        score_spatiotemporal = self.get_score(self.spatiotemporal_iou_key)
        score_spatial = self.get_score(self.spatial_iou_key)
        score_weighted_spatial = self.get_score(self.weighted_spatial_iou_key)
        score_mse = self.get_score(self.mse_key)
        final_score_raw = (
            score_spatiotemporal + score_spatial + score_weighted_spatial
        ) / 3 - score_mse
        return final_score_raw

    def compute_final_score_orig(self) -> float:
        """Returns the original physics-iq score with clipping."""
        return clip(self.compute_final_score_orig_raw())

    def compute_final_score_stable(self) -> float:
        """Returns the stable physics-iq score, where each component is clipped before aggregation."""
        score_spatiotemporal = clip(self.get_score(self.spatiotemporal_iou_key))
        score_spatial = clip(self.get_score(self.spatial_iou_key))
        score_weighted_spatial = clip(self.get_score(self.weighted_spatial_iou_key))
        score_mse = clip(self.get_score(self.mse_key))
        final_score_stable = clip(
            (score_spatiotemporal + score_spatial + score_weighted_spatial) / 3
            - score_mse
        )
        return final_score_stable

    def get_output_dict(self) -> dict[str, float]:
        """
        Return a dict of all relevant scores and metadata for this table.
        The original score is final_score_orig, but we advise to use final_score_stable for a more robust evaluation.
        The verified score is final_score_view. Verified subscores also include the _view suffix, e.g. score_spatial_view.
        """
        view_scenario_scores = clip(self.compute_scores_scenario_by_view())
        final_score_raw = self.compute_final_score_orig_raw()
        out_dict = {
            "score_spatiotemporal": self.get_score(self.spatiotemporal_iou_key),
            "score_spatial": self.get_score(self.spatial_iou_key),
            "score_weighted_spatial": self.get_score(self.weighted_spatial_iou_key),
            "score_mse": self.get_score(self.mse_key),
            "final_score_raw": final_score_raw,
            "final_score_stable": self.compute_final_score_stable(),
            "final_score_orig": clip(final_score_raw),
            "variance_mse": self.get_metric_mean(self.variance_mse_key),
            "variance_spatiotemporal_iou": self.get_metric_mean(
                self.variance_spatiotemporal_iou_key
            ),
            "variance_spatial": self.get_metric_mean(self.variance_spatial_key),
            "variance_weighted_spatial": self.get_metric_mean(
                self.variance_weighted_spatial_key
            ),
            "final_score_view": view_scenario_scores["final_arithmetic"].mean(),
            "score_spatiotemporal_view": view_scenario_scores[
                self.spatiotemporal_iou_key + "_score"
            ].mean(),
            "score_spatial_view": view_scenario_scores[
                self.spatial_iou_key + "_score"
            ].mean(),
            "score_weighted_spatial_view": view_scenario_scores[
                self.weighted_spatial_iou_key + "_score"
            ].mean(),
            "score_mse_view": view_scenario_scores[self.mse_key + "_score"].mean(),
        }
        out_dict.update(self.metadata)

        return out_dict

    @classmethod
    def get_list_keys(cls) -> list[str]:
        """Metric keys whose CSV columns contain per-frame lists (spatiotemporal IOU, MSE)."""
        return [
            cls.spatiotemporal_iou_key,
            cls.mse_key,
            cls.variance_spatiotemporal_iou_key,
            cls.variance_mse_key,
        ]

    @classmethod
    def get_scalar_keys(cls) -> list[str]:
        """Metric keys whose CSV columns contain a single float per scenario (spatial IOU)."""
        return [
            cls.spatial_iou_key,
            cls.weighted_spatial_iou_key,
            cls.variance_spatial_key,
            cls.variance_weighted_spatial_key,
        ]

    @classmethod
    def get_list_columns(cls) -> list[str]:
        return [
            f"{metric}_{view}" for metric in cls.get_list_keys() for view in cls.views
        ]

    @classmethod
    def from_csv(cls, file_path: str, *args, **kwargs):
        df = pd.read_csv(file_path)
        list_columns = cls.get_list_columns()
        for col in list_columns:
            df[col] = df[col].apply(parse_list_of_floats)
        return cls(df, *args, **kwargs)
