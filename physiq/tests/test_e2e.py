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

import pytest

from physiq.calculate_and_write_metrics_to_csv import process_videos
from physiq.calculate_iq_score_stable import IQTable
from physiq.dataset import DESCRIPTIONS_PATH, Benchmark

REPO_ROOT = DESCRIPTIONS_PATH.parent.parent
BENCHMARK_DATA = REPO_ROOT / "physics-IQ-benchmark"
REAL_FOLDER = BENCHMARK_DATA / "split-videos" / "testing" / "30FPS"
BINARY_MASK_FOLDER = BENCHMARK_DATA / "video-masks" / "real" / "30FPS"
FPS = 30


@pytest.mark.e2e
def test_real2real_scores_are_one(tmp_path):
    """Evaluating ground-truth videos against themselves must produce score = 1."""
    if not REAL_FOLDER.exists():
        pytest.skip(f"Benchmark data not found at {REAL_FOLDER}")

    # Build a generated/ folder with symlinks named in the generated-video format.
    # Real files:      {id}_testing-videos_{fps}FPS_{view}_take-1_{scenario}.mp4
    # Generated files: {id}_{view}_{scenario}.mp4
    generated_folder = tmp_path / "generated"
    generated_folder.mkdir()
    benchmark = Benchmark.from_yaml(DESCRIPTIONS_PATH)
    benchmark_df = benchmark.to_dataframe()
    for _, row in benchmark_df[benchmark_df["take"] == 1].iterrows():
        src = REAL_FOLDER / Benchmark._infer_gt_fn_video(
            row["id"], row["perspective"], 1, row["scene"], FPS
        )
        dst = generated_folder / Benchmark._infer_generated(
            row["id"], row["perspective"], row["scene"]
        )
        dst.symlink_to(src)

    csv_path = tmp_path / "real2real.csv"
    process_videos(
        real_folder=str(REAL_FOLDER),
        generated_folder=str(generated_folder),
        binary_real_folder=str(BINARY_MASK_FOLDER),
        binary_generated_folder=str(BINARY_MASK_FOLDER),
        csv_file_path=str(csv_path),
        fps=FPS,
        n_processes=0,
    )

    iqtable = IQTable.from_csv(str(csv_path))
    scores = iqtable.get_output_dict()

    assert scores["final_score_orig"] == pytest.approx(1.0, abs=1e-6)
    assert scores["final_score_stable"] == pytest.approx(1.0, abs=1e-6)
    assert scores["final_score_view"] == pytest.approx(1.0, abs=1e-6)
