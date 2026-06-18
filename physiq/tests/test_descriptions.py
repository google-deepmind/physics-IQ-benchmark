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

import pandas as pd
from dataset import DESCRIPTIONS_PATH, Benchmark

_DESCRIPTIONS_CSV = DESCRIPTIONS_PATH.parent / "descriptions_original.csv"


def test_yaml_generates_matching_descriptions_csv():
    csv_gen = Benchmark.from_yaml(DESCRIPTIONS_PATH).build_original_descriptions()
    csv_disk = pd.read_csv(_DESCRIPTIONS_CSV)
    pd.testing.assert_frame_equal(
        csv_gen.reset_index(drop=True),
        csv_disk.reset_index(drop=True),
    )
