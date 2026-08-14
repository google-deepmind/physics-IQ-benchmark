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
"""
Generate the descriptions.csv and placeholder run_01 videos for the
submission/example/ walkthrough.

submission/example/ is a self-contained example a submitter can run
`physiq/submit.py --validate-only` against immediately. Rather than committing
a static descriptions.csv and ~200 binary MP4s to git, this script generates
both on demand: descriptions.csv is derived from the first
EXPECTED_VIDEO_COUNT rows of descriptions/descriptions_original.csv (with
generated_video_name remapped to the synthetic filenames below), and the
videos are one blank, correctly-specced clip per generated_video_name, with
fps/resolution/duration read from submission/example/submission.yaml so the
placeholders always match what that card declares.

Usage:
    uv run physiq/generate_example_submission.py
"""

import argparse
import csv
import subprocess
from pathlib import Path

import yaml

from physiq.submit import EXPECTED_VIDEO_COUNT

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARD = REPO_ROOT / "submission" / "example" / "submission.yaml"
DEFAULT_ORIGINAL_DESCRIPTIONS = REPO_ROOT / "descriptions" / "descriptions_original.csv"
DEFAULT_DESCRIPTIONS = REPO_ROOT / "submission" / "example" / "descriptions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "submission" / "example" / "run_01"

DURATION_S = 5.0


def _make_video(path: Path, *, fps: float, width: int, height: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={fps}",
            "-t", str(DURATION_S),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            "-y", str(path),
        ],
        check=True,
        capture_output=True,
    )


def _write_descriptions(original: Path, dest: Path) -> list[str]:
    """Write descriptions.csv covering EXPECTED_VIDEO_COUNT synthetic videos.

    Reuses the real scenario/description text from descriptions_original.csv
    so the example reads as realistic content, but points generated_video_name
    at the synthetic placeholder filenames this script actually generates.
    """
    with original.open(newline="", encoding="utf-8") as f:
        orig_rows = list(csv.DictReader(f))[:EXPECTED_VIDEO_COUNT]

    names = []
    out_rows = []
    for i, row in enumerate(orig_rows, start=1):
        name = f"{i:04d}_synthetic-example.mp4"
        names.append(name)
        out_rows.append({
            "scenario": row["scenario"],
            "description": row["description"],
            "generated_video_name": name,
        })

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "description", "generated_video_name"])
        writer.writeheader()
        writer.writerows(out_rows)

    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD, help="Path to submission.yaml")
    parser.add_argument(
        "--original-descriptions", type=Path, default=DEFAULT_ORIGINAL_DESCRIPTIONS,
        help="Source descriptions_original.csv to derive scenario/description text from",
    )
    parser.add_argument(
        "--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS,
        help="Path to write the generated descriptions.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Run directory to generate videos into",
    )
    args = parser.parse_args()

    card_data = yaml.safe_load(args.card.open())
    fps = float(card_data["public_info"]["fps"])
    width, height = (int(x) for x in card_data["public_info"]["resolution"].split("x"))

    print(f"Writing {args.descriptions} ...")
    names = _write_descriptions(args.original_descriptions, args.descriptions)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(names)} placeholder video(s) in {args.output}/ "
          f"({width}x{height} @ {fps}fps, {DURATION_S}s each) ...")
    for name in names:
        _make_video(args.output / name, fps=fps, width=width, height=height)
    print("Done.")


if __name__ == "__main__":
    main()
