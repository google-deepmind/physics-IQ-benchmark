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
Physics-IQ submission uploader.

Requires the "submission" extra:
    uv sync --extra submission
    (or: pip install '.[submission]')

S3 upload (default):
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_SESSION_TOKEN=...
    uv run physiq/submit.py \
        --run  <run_id>           \
        --card <submission.yaml>  \
        --descriptions <descriptions.csv> \
        --runs <run_dir> [<run_dir> ...]

Local package (--dest):
    uv run physiq/submit.py \
        --run  <run_id>           \
        --card <submission.yaml>  \
        --descriptions <descriptions.csv> \
        --runs <run_dir> [<run_dir> ...] \
        --dest <output_dir>

  Assembles all files into <output_dir>/ without uploading anything.
  The folder can then be zipped or shared via any file-transfer service.
  AWS credentials are not required in this mode, and boto3 is not required either.

Output layout (identical for both modes):
    submission.yaml
    descriptions.csv
    runs/<run_dir_name>/
        <all files recursively>
"""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

BUCKET = "anates-physics-iq-submissions"
REGION = "eu-central-1"
RUN_ID_RE = re.compile(r"^[a-z0-9]+__[a-z0-9.-]+__\d{4}-\d{2}-\d{2}$")
CURRENT_TERMS_VERSION = "1.0"

REQUIRED_CARD_FIELDS = [
    "organization",
    "model",
    "model_version",
    "input_type",
    "fps",
    "resolution",
    "bon_sampling",
    "prompt_upsampling",
    "descriptions",
    "availability",
    "model_source",
    "date",
    "generation_cost",
]

# Only meaningful when prompt upsampling was actually performed (see submission.yaml comments).
UPSAMPLING_REQUIRED_CARD_FIELDS = [
    "upsample_cost",
]

# Only meaningful when the submitter ran generation themselves (see submission.yaml comments).
NON_API_REQUIRED_CARD_FIELDS = [
    "generation_gpu",
    "generation_ngpu",
    "generation_time",
]

# Only meaningful when prompt upsampling was performed AND the submitter ran it themselves.
NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS = [
    "upsample_gpu",
    "upsample_ngpu",
    "upsample_time",
]

REQUIRED_REPORTED_SCORE_FIELDS = [
    "physiq_mean",
    "physiq_std",
    "spatial_iou_mean",
    "spatial_iou_std",
    "weighted_spatial_iou_mean",
    "weighted_spatial_iou_std",
    "spatiotemporal_iou_mean",
    "spatiotemporal_iou_std",
    "mse_mean",
    "mse_std",
]

WARRANTY_FIELDS = [
    "has_rights_to_submit",
    "no_ip_infringement",
    "permitted_by_provider_tos",
    "no_personal_data_in_content",
    "metadata_accurate",
]


def _validate_card(card_data: dict) -> list[str]:
    """Validate submission card fields. Returns list of error strings (empty = valid)."""
    errors = []

    public_info = card_data.get("public_info") or {}
    missing = [k for k in REQUIRED_CARD_FIELDS if public_info.get(k) in (None, "")]
    for k in missing:
        errors.append(f"required field 'public_info.{k}' is empty")

    if public_info.get("prompt_upsampling") is True:
        missing = [k for k in UPSAMPLING_REQUIRED_CARD_FIELDS if public_info.get(k) in (None, "")]
        for k in missing:
            errors.append(
                f"required field 'public_info.{k}' is empty (required when prompt_upsampling is true)"
            )

        if public_info.get("availability") != "api":
            missing = [
                k for k in NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS if public_info.get(k) in (None, "")
            ]
            for k in missing:
                errors.append(
                    f"required field 'public_info.{k}' is empty "
                    "(required for non-API models with prompt upsampling)"
                )

    if public_info.get("availability") != "api":
        missing = [k for k in NON_API_REQUIRED_CARD_FIELDS if public_info.get(k) in (None, "")]
        for k in missing:
            errors.append(f"required field 'public_info.{k}' is empty (required for non-API models)")

    reported_scores = card_data.get("reported_scores") or {}
    missing = [k for k in REQUIRED_REPORTED_SCORE_FIELDS if reported_scores.get(k) in (None, "")]
    for k in missing:
        errors.append(f"required field 'reported_scores.{k}' is empty")

    terms = card_data.get("terms") or {}
    if terms.get("terms_version") != CURRENT_TERMS_VERSION:
        errors.append(
            f"terms.terms_version must be \"{CURRENT_TERMS_VERSION}\" "
            f"(got {terms.get('terms_version')!r})"
        )
    if terms.get("terms_accepted") is not True:
        errors.append("terms.terms_accepted must be true")
    if not terms.get("accepted_on_behalf_of"):
        errors.append("terms.accepted_on_behalf_of must be a non-empty string")
    if not terms.get("accepted_at"):
        errors.append("terms.accepted_at must be a non-empty string")

    warranties = card_data.get("warranties") or {}
    for field in WARRANTY_FIELDS:
        if warranties.get(field) is not True:
            errors.append(f"warranties.{field} must be true")

    return errors


EXPECTED_VIDEO_COUNT = 198
EXPECTED_DURATION_S = 5.0
DURATION_TOLERANCE = 0.001
FPS_TOLERANCE = 0.01


def _ffprobe_video_info(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "fps": int(num) / int(den),
        "width": stream["width"],
        "height": stream["height"],
        "duration": float(data["format"]["duration"]),
    }


def _validate_run_dir(run_dir: Path, card_fps: float, card_resolution: str) -> list[str]:
    errors = []

    mp4s = sorted(run_dir.glob("*.mp4"))
    if len(mp4s) != EXPECTED_VIDEO_COUNT:
        errors.append(
            f"found {len(mp4s)} video(s) but expected {EXPECTED_VIDEO_COUNT}"
        )
        return errors  # naming/probe checks meaningless with wrong count

    for i, path in enumerate(mp4s, start=1):
        expected_prefix = f"{i:04d}_"
        if not path.name.startswith(expected_prefix):
            errors.append(
                f"file #{i} must start with '{expected_prefix}', got '{path.name}'"
            )

    card_w, card_h = (int(x) for x in card_resolution.split("x"))
    fps_values: list[float] = []
    for path in mp4s:
        try:
            info = _ffprobe_video_info(path)
        except Exception as exc:
            errors.append(f"could not probe {path.name}: {exc}")
            continue

        if not math.isclose(info["duration"], EXPECTED_DURATION_S, abs_tol=DURATION_TOLERANCE):
            errors.append(
                f"{path.name} is {info['duration']:.4f}s — must be exactly {EXPECTED_DURATION_S}s "
                f"(tolerance ±{DURATION_TOLERANCE}s)"
            )
        if info["width"] != card_w or info["height"] != card_h:
            errors.append(
                f"{path.name} resolution {info['width']}x{info['height']} "
                f"does not match card ({card_resolution})"
            )
        fps_values.append(info["fps"])

    if fps_values:
        if len(set(fps_values)) > 1:
            errors.append("videos have inconsistent FPS within this run directory")
        elif not math.isclose(fps_values[0], card_fps, abs_tol=FPS_TOLERANCE):
            errors.append(
                f"video FPS {fps_values[0]:.4f} does not match card fps={card_fps}"
            )

    return errors


def _validate_descriptions(descriptions: Path, run_dirs: list[Path]) -> list[str]:
    """Validate descriptions.csv structure and coverage of submitted videos."""
    errors = []

    try:
        with descriptions.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception as exc:
        return [f"descriptions.csv could not be parsed: {exc}"]

    required_cols = {"scenario", "description", "generated_video_name"}
    missing_cols = required_cols - set(fieldnames)
    if missing_cols:
        errors.append(
            f"descriptions.csv missing required column(s): {sorted(missing_cols)}"
        )
        return errors

    covered = {row["generated_video_name"].strip() for row in rows}
    for run_dir in run_dirs:
        for mp4 in sorted(run_dir.glob("*.mp4")):
            if mp4.name not in covered:
                errors.append(
                    f"[{run_dir.name}] {mp4.name} not found in "
                    f"descriptions.csv generated_video_name column"
                )

    return errors


def _require_boto3():
    """Lazily import boto3, with an actionable error if the extra isn't installed."""
    try:
        import boto3
    except ImportError:
        sys.exit(
            "ERROR: boto3 is required to upload to S3.\n"
            "  Install it with:\n"
            "    uv sync --extra submission\n"
            "  or:\n"
            "    pip install '.[submission]'\n"
            "  (Not needed for --validate-only or --dest local packaging.)"
        )
    return boto3


def _upload(s3_client, local: Path, key: str) -> int:
    s3_client.upload_file(str(local), BUCKET, key)
    size = local.stat().st_size
    print(f"  {key.split('/', 2)[-1]:<60}  {size:>12,} B")
    return size


def _upload_dir(s3_client, local_dir: Path, s3_prefix: str) -> tuple[int, int]:
    files, total = 0, 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir)
        size = _upload(s3_client, path, f"{s3_prefix}/{rel}")
        files += 1
        total += size
    return files, total


def _copy(src: Path, dest: Path, label: str) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    size = src.stat().st_size
    print(f"  {label:<60}  {size:>12,} B")
    return size


def _copy_dir(src_dir: Path, dest_dir: Path, dest_root: Path) -> tuple[int, int]:
    files, total = 0, 0
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir)
        dest_file = dest_dir / rel
        size = _copy(path, dest_file, str(dest_file.relative_to(dest_root)))
        files += 1
        total += size
    return files, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package or upload a Physics-IQ submission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Without --dest: uploads directly to S3 (requires AWS credentials as env vars\n"
            "                and the 'submission' extra: uv sync --extra submission).\n"
            "With --dest:    copies files to a local folder instead — no credentials or\n"
            "                boto3 needed."
        ),
    )
    parser.add_argument("--run", required=True, help="run_id  (org__model__YYYY-MM-DD)")
    parser.add_argument("--card", required=True, help="Path to filled-in submission.yaml")
    parser.add_argument("--descriptions", required=True, help="Path to descriptions.csv")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more run directories",
    )
    parser.add_argument(
        "--dest",
        metavar="DIR",
        default=None,
        help="Local output folder (skips S3 upload; folder must not already exist)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip video file validation (count, naming, FPS, resolution, duration)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all validation checks and exit without uploading or packaging",
    )
    args = parser.parse_args()

    # ── validate run_id ──────────────────────────────────────────────────────
    if not RUN_ID_RE.match(args.run):
        sys.exit(
            f"ERROR: invalid run_id '{args.run}'\n"
            "  Expected: <org>__<descriptive-id>__<YYYY-MM-DD>  (lowercase, no spaces)"
        )

    # ── validate local paths ─────────────────────────────────────────────────
    card = Path(args.card)
    descriptions = Path(args.descriptions)
    run_dirs = [Path(r) for r in args.runs]

    errors = []
    if not card.is_file():
        errors.append(f"Card not found: {card}")
    if not descriptions.is_file():
        errors.append(f"Descriptions file not found: {descriptions}")
    for rd in run_dirs:
        if not rd.is_dir():
            errors.append(f"Run directory not found: {rd}")
    if errors:
        sys.exit("ERROR:\n" + "\n".join(f"  {e}" for e in errors))

    # ── validate card YAML ───────────────────────────────────────────────────
    with card.open() as f:
        try:
            card_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            sys.exit(f"ERROR: could not parse card YAML: {exc}")

    card_errors = _validate_card(card_data)
    if card_errors:
        sys.exit(
            "ERROR: submission card has errors:\n"
            + "\n".join(f"  {e}" for e in card_errors)
            + "\n\n  Please read submission/submission_terms.md before filling in the terms: section."
        )

    # ── validate descriptions CSV ────────────────────────────────────────────
    desc_errors = _validate_descriptions(descriptions, run_dirs)
    if desc_errors:
        sys.exit(
            "ERROR: descriptions.csv validation failed:\n"
            + "\n".join(f"  {e}" for e in desc_errors)
        )

    # ── validate run directories ─────────────────────────────────────────────
    if not args.skip_validation:
        print("Validating run directories …")
        card_fps = float(card_data["public_info"]["fps"])
        card_resolution = card_data["public_info"]["resolution"]
        val_errors = []
        for run_dir in run_dirs:
            errs = _validate_run_dir(run_dir, card_fps, card_resolution)
            for e in errs:
                val_errors.append(f"[{run_dir.name}] {e}")
        if val_errors:
            sys.exit(
                "ERROR: video validation failed:\n"
                + "\n".join(f"  {e}" for e in val_errors)
            )
        print("Validation passed.\n")

    # ── validate-only mode ───────────────────────────────────────────────────
    if args.validate_only:
        n = len(run_dirs)
        print(
            f"All checks passed — {n} run(s), {EXPECTED_VIDEO_COUNT} videos each. "
            f"Ready to submit."
        )
        return

    # ── local package mode ───────────────────────────────────────────────────
    if args.dest is not None:
        dest = Path(args.dest)
        if dest.exists():
            sys.exit(
                f"ERROR: destination already exists: {dest}\n"
                "  Remove it or choose a different path."
            )

        print(f"\nPackaging to  {dest}/\n")
        total_files, total_bytes = 0, 0

        size = _copy(card, dest / "submission.yaml", "submission.yaml")
        total_files += 1
        total_bytes += size

        size = _copy(descriptions, dest / "descriptions.csv", "descriptions.csv")
        total_files += 1
        total_bytes += size

        for run_dir in run_dirs:
            f, b = _copy_dir(run_dir, dest / "runs" / run_dir.name, dest)
            total_files += f
            total_bytes += b

        print(f"\nDone — {total_files} file(s), {total_bytes:,} bytes written to {dest}/")
        return

    # ── S3 upload mode ───────────────────────────────────────────────────────
    boto3 = _require_boto3()
    s3 = boto3.client("s3", region_name=REGION)
    prefix = f"submissions/{args.run}"

    print(f"\nUploading to  s3://{BUCKET}/{prefix}/\n")
    total_files, total_bytes = 0, 0

    size = _upload(s3, card, f"{prefix}/submission.yaml")
    total_files += 1
    total_bytes += size

    size = _upload(s3, descriptions, f"{prefix}/descriptions.csv")
    total_files += 1
    total_bytes += size

    for run_dir in run_dirs:
        f, b = _upload_dir(s3, run_dir, f"{prefix}/runs/{run_dir.name}")
        total_files += f
        total_bytes += b

    print(f"\nDone — {total_files} file(s), {total_bytes:,} bytes uploaded.")


if __name__ == "__main__":
    main()
