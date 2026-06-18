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

import argparse
from pathlib import Path

import pandas as pd
from dataset import DESCRIPTIONS_PATH, Benchmark

from physiq.templater import REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Generate a descriptions CSV using a named templater."
    )
    parser.add_argument("templater", choices=list(REGISTRY), help="Templater to use")
    parser.add_argument(
        "--output-dir",
        default=str(DESCRIPTIONS_PATH.parent / "best_practice"),
        help="Directory to write the output CSV (default: descriptions/best_practice)",
    )
    parser.add_argument(
        "--no-action-suffix",
        action="store_true",
        help="Disable subject-action postfix appending. Do not use this option for the verified evaluation.",
    )
    args = parser.parse_args()

    benchmark = Benchmark.from_yaml(DESCRIPTIONS_PATH)
    df = benchmark.to_dataframe()

    templater = REGISTRY[args.templater](
        df, use_action_suffix=not args.no_action_suffix
    )

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "description": templater.generate_prompt(row["scenario"]),
                "category": row["category"],
                "generated_video_name": row["generated_video_name"],
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(args.output_dir) / f"descriptions_{args.templater}.csv"
    if not out_path.parent.exists():
        print(f"Output path did not exist. creating {out_path.parent}")
        out_path.parent.mkdir(parents=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows → {out_path}")


if __name__ == "__main__":
    main()
