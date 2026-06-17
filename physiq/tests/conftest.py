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

import io

import pandas as pd
import pytest

VIEWS = ["perspective-left", "perspective-center", "perspective-right"]


def _list_str(value, n_frames):
    """Format a float as a per-frame list string, e.g. '[0.1, 0.1, 0.1, 0.1]'."""
    return "[" + ", ".join(str(value) for _ in range(n_frames)) + "]"


def make_csv(
    tmp_path,
    *,
    n_scenarios=3,
    n_frames=4,
    mse=0.1,
    st_iou=0.4,
    sp_iou=0.5,
    wsp_iou=0.6,
    var_mse=0.05,
    var_st_iou=0.3,
    var_sp=0.25,
    var_wsp=0.5,
    # per-scenario overrides: list of dicts with the same keys above (len == n_scenarios)
    per_scenario=None,
):
    """Write a minimal benchmark CSV to tmp_path and return its path.

    Columns satisfy both calculate_iq_score() and IQTable.from_csv():
      - list columns stored as "[v, v, ...]" strings
      - scalar columns stored as plain floats
    """
    rows = []
    for i in range(n_scenarios):
        overrides = (per_scenario or [{}])[i] if per_scenario else {}
        r_mse = overrides.get("mse", mse)
        r_st = overrides.get("st_iou", st_iou)
        r_sp = overrides.get("sp_iou", sp_iou)
        r_wsp = overrides.get("wsp_iou", wsp_iou)
        r_vm = overrides.get("var_mse", var_mse)
        r_vst = overrides.get("var_st_iou", var_st_iou)
        r_vsp = overrides.get("var_sp", var_sp)
        r_vwsp = overrides.get("var_wsp", var_wsp)

        row = {"scenario": f"scenario_{i}"}
        for view in VIEWS:
            row[f"v1_mse_{view}"] = _list_str(r_mse, n_frames)
            row[f"spatiotemporal_iou_v1_{view}"] = _list_str(r_st, n_frames)
            row[f"spatial_iou_v1_{view}"] = r_sp
            row[f"weighted_spatial_iou_v1_{view}"] = r_wsp
            row[f"variance_mse_{view}"] = _list_str(r_vm, n_frames)
            row[f"variance_spatiotemporal_iou_{view}"] = _list_str(r_vst, n_frames)
            row[f"variance_spatial_{view}"] = r_vsp
            row[f"variance_weighted_spatial_{view}"] = r_vwsp
        rows.append(row)

    df = pd.DataFrame(rows)
    path = tmp_path / "test_metrics.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def default_csv(tmp_path):
    """CSV with clean (≤4 dp) uniform values; score is well inside [0, 1]."""
    return make_csv(tmp_path)


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e", action="store_true", default=False, help="Run end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-e2e"):
        skip_e2e = pytest.mark.skip(reason="Pass --run-e2e to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
