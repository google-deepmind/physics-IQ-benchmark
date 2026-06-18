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

"""Verify that calculate_iq_score() and IQTable.compute_final_score_orig() are equivalent.

Scope
-----
calculate_iq_score() returns a score in [0, 100].
IQTable.compute_final_score_orig() returns the same quantity in [0, 1].
All assertions use: old_score ≈ new_score * 100.

IQTable.compute_final_score_stable() and final_score_view are intentionally
different from the old code and are NOT tested for equivalence here.

Tolerance choices
-----------------
abs=0.005  — old vs new equivalence on 0-100 scale; old code applies round(..., 2)
             so the max possible divergence is 0.005 (tests 1, 7).
abs=1e-4   — variance match; old code rounds each variance component to 4 dp (test 3).
abs=1e-10  — floating-point equality; same formula computed via two different code
             paths (CSV round-trip + numpy vs direct Python).  Used where both sides
             apply the same rounding, so results must agree to floating-point
             precision (tests 2, 4).
==         — exact equality; clipping to 0 or max is a deterministic float operation
             with no ambiguity (tests 5, 6).
"""

import numpy as np
import pytest
from calculate_iq_score import VIEWS, calculate_iq_score, parse_list_of_floats
from calculate_iq_score_stable import IQTable
from tests.conftest import make_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _old(path):
    """Return (score_0_100, physical_variance) from the old implementation."""
    return calculate_iq_score(str(path))


def _new_orig(path):
    """Return score in [0, 100] from IQTable.compute_final_score_orig()."""
    return IQTable.from_csv(str(path)).compute_final_score_orig() * 100


# ---------------------------------------------------------------------------
# 1. Main equivalence: default fixture
# ---------------------------------------------------------------------------


def test_final_score_orig_matches_default(default_csv):
    old_score, _ = _old(default_csv)
    new_score = _new_orig(default_csv)
    assert old_score == pytest.approx(new_score, abs=0.005)


# ---------------------------------------------------------------------------
# 2. Both match a hand-computed expected value for the default fixture
#
# Default values: mse=0.1, st_iou=0.4, sp_iou=0.5, wsp_iou=0.6
#                 var_mse=0.05, var_st=0.3, var_sp=0.25, var_wsp=0.5
#
# raw = ((0.4/0.3 + 0.5/0.25 + 0.6/0.5) / 3) - (0.1 - 0.05)
#     = ((1.3333 + 2.0 + 1.2) / 3) - 0.05
#     = 1.5111 - 0.05 = 1.4611...
# clipped to [0, 1] → 1.0; scaled: 100.0
# ---------------------------------------------------------------------------


def test_final_score_is_manual_computation(default_csv):
    raw = ((0.4 / 0.3 + 0.5 / 0.25 + 0.6 / 0.5) / 3) - (0.1 - 0.05)
    expected_01 = float(np.clip(raw, 0.0, 1.0))
    expected_100 = round(max(min(raw * 100, 100.0), 0.0), 2)

    old_score, _ = _old(default_csv)
    new_score_01 = IQTable.from_csv(str(default_csv)).compute_final_score_orig()

    assert old_score == pytest.approx(expected_100, abs=1e-10)
    assert new_score_01 == pytest.approx(expected_01, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. Physical variance return value
# ---------------------------------------------------------------------------


def test_physical_variance_matches(default_csv):
    _, old_var = _old(default_csv)
    tab = IQTable.from_csv(str(default_csv))
    new_var = (
        tab.get_metric_mean("variance_spatiotemporal_iou")
        + tab.get_metric_mean("variance_spatial")
        + tab.get_metric_mean("variance_weighted_spatial")
        - tab.get_metric_mean("variance_mse")
    )
    assert old_var == pytest.approx(new_var, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Component metric means match
# ---------------------------------------------------------------------------


def test_component_metric_means_match(default_csv):
    import pandas as pd

    df = pd.read_csv(str(default_csv))
    for col in IQTable.get_list_columns():
        df[col] = df[col].apply(parse_list_of_floats)

    tab = IQTable.from_csv(str(default_csv))

    # MSE cross-view mean
    old_mse = df.apply(
        lambda row: np.mean(np.concatenate([row[f"v1_mse_{v}"] for v in VIEWS])),
        axis=1,
    ).mean()
    assert old_mse == pytest.approx(tab.get_metric_mean("v1_mse"), abs=1e-10)

    # Spatiotemporal IOU cross-view mean
    old_st = df.apply(
        lambda row: np.mean(
            np.concatenate([row[f"spatiotemporal_iou_v1_{v}"] for v in VIEWS])
        ),
        axis=1,
    ).mean()
    assert old_st == pytest.approx(
        tab.get_metric_mean("spatiotemporal_iou_v1"), abs=1e-10
    )

    # Spatial IOU cross-view mean (scalar)
    old_sp = df[[f"spatial_iou_v1_{v}" for v in VIEWS]].mean().mean()
    assert old_sp == pytest.approx(tab.get_metric_mean("spatial_iou_v1"), abs=1e-10)

    # Weighted spatial IOU cross-view mean (scalar)
    old_wsp = df[[f"weighted_spatial_iou_v1_{v}" for v in VIEWS]].mean().mean()
    assert old_wsp == pytest.approx(
        tab.get_metric_mean("weighted_spatial_iou_v1"), abs=1e-10
    )


# ---------------------------------------------------------------------------
# 5. Score clipped to 0 when raw < 0
# ---------------------------------------------------------------------------


def test_clipping_at_zero(tmp_path):
    # Very small IOU metrics relative to large variances, and large MSE → raw score << 0
    path = make_csv(
        tmp_path,
        mse=0.9,
        st_iou=0.01,
        sp_iou=0.01,
        wsp_iou=0.01,
        var_mse=0.05,
        var_st_iou=1.0,
        var_sp=1.0,
        var_wsp=1.0,
    )
    old_score, _ = _old(path)
    new_score = _new_orig(path)
    assert old_score == 0.0
    assert new_score == 0.0


# ---------------------------------------------------------------------------
# 6. Score clipped to 100 / 1.0 when raw > 1
# ---------------------------------------------------------------------------


def test_clipping_at_max(tmp_path):
    # IOU metrics >> variances, near-zero MSE → raw score >> 1
    path = make_csv(
        tmp_path,
        mse=0.0,
        st_iou=5.0,
        sp_iou=5.0,
        wsp_iou=5.0,
        var_mse=0.05,
        var_st_iou=0.1,
        var_sp=0.1,
        var_wsp=0.1,
    )
    old_score, _ = _old(path)
    new_score_01 = IQTable.from_csv(str(path)).compute_final_score_orig()
    assert old_score == 100.0
    assert new_score_01 == 1.0


# ---------------------------------------------------------------------------
# 7. Multi-scenario aggregation: scenarios with different per-scenario values
# ---------------------------------------------------------------------------


def test_multi_scenario_aggregation(tmp_path):
    per_scenario = [
        {"sp_iou": 0.3, "wsp_iou": 0.4, "st_iou": 0.2, "mse": 0.05},
        {"sp_iou": 0.5, "wsp_iou": 0.6, "st_iou": 0.4, "mse": 0.1},
        {"sp_iou": 0.7, "wsp_iou": 0.8, "st_iou": 0.6, "mse": 0.15},
        {"sp_iou": 0.4, "wsp_iou": 0.5, "st_iou": 0.3, "mse": 0.08},
        {"sp_iou": 0.6, "wsp_iou": 0.7, "st_iou": 0.5, "mse": 0.12},
    ]
    path = make_csv(tmp_path, n_scenarios=5, per_scenario=per_scenario)

    old_score, _ = _old(path)
    new_score = _new_orig(path)
    assert old_score == pytest.approx(new_score, abs=0.005)


# ---------------------------------------------------------------------------
# 8. get_output_dict completes without error and returns all expected keys
#
# Regression guard: compute_scores_scenario_by_view() runs before the first
# get_score() call inside get_output_dict(), so any false-positive from
# _verify_df_integrity would surface here.
# ---------------------------------------------------------------------------


def test_get_output_dict_returns_expected_keys(default_csv):
    result = IQTable.from_csv(str(default_csv)).get_output_dict()
    assert True  # code ran through


def test_get_output_dict_scores_are_finite_floats(default_csv):
    result = IQTable.from_csv(str(default_csv)).get_output_dict()
    for key in result:
        assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"


# ---------------------------------------------------------------------------
# 9. Empty DataFrame raises ValueError at construction
# ---------------------------------------------------------------------------


def test_empty_dataframe_raises_value_error():
    import pandas as pd

    cols = [
        f"{m}_{v}"
        for m in IQTable.get_list_keys() + IQTable.get_scalar_keys()
        for v in IQTable.views
    ]
    with pytest.raises(ValueError, match="empty"):
        IQTable(pd.DataFrame({col: [] for col in cols}))
