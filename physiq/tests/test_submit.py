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
"""Tests for _validate_card, _validate_run_dir, and _validate_descriptions in submit.py.

Performance note: session-scoped fixtures create 4 real videos via ffmpeg
(correct, wrong-fps, wrong-resolution, wrong-duration). All run directories
are populated with hard links to those videos, so only OS metadata operations
are needed per test — no additional ffmpeg calls.
"""

import csv
import subprocess
from pathlib import Path

import pytest

from physiq.submit_verified import (
    CURRENT_TERMS_VERSION,
    EXPECTED_VIDEO_COUNT,
    NON_API_REQUIRED_CARD_FIELDS,
    NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS,
    UPSAMPLING_REQUIRED_CARD_FIELDS,
    _validate_card,
    _validate_descriptions,
    _validate_run_dir,
)

CARD_FPS = 24.0
CARD_RES = "64x36"
CARD_W, CARD_H = 64, 36


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_video(path: Path, *, fps=CARD_FPS, width=CARD_W, height=CARD_H, duration=5.0):
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={fps}",
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "51",
            "-y", str(path),
        ],
        check=True,
        capture_output=True,
    )


def _populate_run(dest: Path, base_video: Path, overrides: dict[int, Path] | None = None):
    """
    Create a run directory with EXPECTED_VIDEO_COUNT correctly-named MP4s using
    hard links. overrides maps 1-based slot index → source path for bad videos.
    """
    overrides = overrides or {}
    for i in range(1, EXPECTED_VIDEO_COUNT + 1):
        source = overrides.get(i, base_video)
        (dest / f"{i:04d}_video.mp4").hardlink_to(source)


# ── session-scoped source videos (ffmpeg called exactly 4 times total) ────────

@pytest.fixture(scope="session")
def correct_video(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("src") / "correct.mp4"
    _make_video(p)
    return p


@pytest.fixture(scope="session")
def wrong_fps_video(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("src_fps") / "wrong_fps.mp4"
    _make_video(p, fps=30)
    return p


@pytest.fixture(scope="session")
def wrong_res_video(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("src_res") / "wrong_res.mp4"
    _make_video(p, width=128, height=128)
    return p


@pytest.fixture(scope="session")
def wrong_dur_video(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("src_dur") / "wrong_dur.mp4"
    _make_video(p, duration=3.0)
    return p


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.ffmpeg
class TestValidRun:
    def test_passes_with_no_errors(self, tmp_path, correct_video):
        _populate_run(tmp_path, correct_video)
        assert _validate_run_dir(tmp_path, CARD_FPS, CARD_RES) == []


@pytest.mark.ffmpeg
class TestVideoCount:
    def test_too_few_videos(self, tmp_path, correct_video):
        for i in range(1, 3):  # 2 videos
            (tmp_path / f"{i:04d}_video.mp4").hardlink_to(correct_video)
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert len(errors) == 1
        assert "found 2" in errors[0]
        assert "198" in errors[0]

    def test_too_many_videos(self, tmp_path, correct_video):
        for i in range(1, 200):  # 199 videos
            (tmp_path / f"{i:04d}_video.mp4").hardlink_to(correct_video)
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert len(errors) == 1
        assert "found 199" in errors[0]

    def test_count_error_skips_further_checks(self, tmp_path, correct_video):
        # only 1 video — should return exactly one error and stop
        (tmp_path / "0001_video.mp4").hardlink_to(correct_video)
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert len(errors) == 1


@pytest.mark.ffmpeg
class TestNaming:
    def test_wrong_prefix_caught(self, tmp_path, correct_video):
        # 198 files but slot 5 has a bad prefix
        for i in range(1, EXPECTED_VIDEO_COUNT + 1):
            name = f"{i:04d}_video.mp4" if i != 5 else "0005wrong_video.mp4"
            (tmp_path / name).hardlink_to(correct_video)
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert any("0005_" in e for e in errors)

    def test_correct_naming_produces_no_naming_errors(self, tmp_path, correct_video):
        _populate_run(tmp_path, correct_video)
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert not any("must start with" in e for e in errors)


@pytest.mark.ffmpeg
class TestFPS:
    def test_all_wrong_fps_caught(self, tmp_path, wrong_fps_video):
        # all 198 videos at 30 fps, card says 24
        _populate_run(tmp_path, wrong_fps_video)
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert any("fps" in e.lower() for e in errors)
        assert any("30." in e for e in errors)

    def test_inconsistent_fps_caught(self, tmp_path, correct_video, wrong_fps_video):
        # 197 videos at 24 fps + 1 at 30 fps
        _populate_run(tmp_path, correct_video, overrides={1: wrong_fps_video})
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert any("inconsistent" in e for e in errors)


@pytest.mark.ffmpeg
class TestResolution:
    def test_wrong_resolution_caught(self, tmp_path, correct_video, wrong_res_video):
        _populate_run(tmp_path, correct_video, overrides={1: wrong_res_video})
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert any("128x128" in e for e in errors)
        assert any(CARD_RES in e for e in errors)

    def test_error_names_the_offending_file(self, tmp_path, correct_video, wrong_res_video):
        _populate_run(tmp_path, correct_video, overrides={7: wrong_res_video})
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert any("0007_video.mp4" in e for e in errors)


@pytest.mark.ffmpeg
class TestDuration:
    def test_wrong_duration_caught(self, tmp_path, correct_video, wrong_dur_video):
        _populate_run(tmp_path, correct_video, overrides={1: wrong_dur_video})
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert any("is 3." in e for e in errors)
        assert any("5.0s" in e for e in errors)

    def test_error_names_the_offending_file(self, tmp_path, correct_video, wrong_dur_video):
        _populate_run(tmp_path, correct_video, overrides={42: wrong_dur_video})
        errors = _validate_run_dir(tmp_path, CARD_FPS, CARD_RES)
        assert any("0042_video.mp4" in e for e in errors)


# ── card validation tests ─────────────────────────────────────────────────────

def _valid_card() -> dict:
    """Minimal fully-valid card dict."""
    return {
        "public_info": {
            "organization": "TestCo",
            "model": "TestModel",
            "model_version": "test-v1",
            "input_type": "t2v",
            "fps": 24,
            "resolution": "1280x720",
            "bon_sampling": 1,
            "prompt_upsampling": False,
            "descriptions": "op",
            "availability": "api",
            "model_source": "https://example.com",
            "date": "2026-07-02",
            "upsample_cost": 0.0,
            "upsample_gpu": "H100",
            "upsample_ngpu": 8,
            "upsample_time": 5.0,
            "generation_cost": 0.45,
            "generation_gpu": "100",
            "generation_ngpu": 8,
            "generation_time": 42.5,
        },
        "reported_scores": {
            "physiq_mean": 0.81,
            "physiq_std": 0.03,
            "spatial_iou_mean": 0.76,
            "spatial_iou_std": 0.04,
            "weighted_spatial_iou_mean": 0.77,
            "weighted_spatial_iou_std": 0.04,
            "spatiotemporal_iou_mean": 0.70,
            "spatiotemporal_iou_std": 0.05,
            "mse_mean": 0.018,
            "mse_std": 0.002,
        },
        "terms": {
            "terms_version": CURRENT_TERMS_VERSION,
            "terms_accepted": True,
            "accepted_on_behalf_of": "TestCo",
            "accepted_at": "2026-07-02T10:00:00Z",
        },
        "warranties": {
            "has_rights_to_submit": True,
            "no_ip_infringement": True,
            "permitted_by_provider_tos": True,
            "no_personal_data_in_content": True,
            "metadata_accurate": True,
        },
    }


class TestCardValidation:
    def test_valid_card_passes(self):
        assert _validate_card(_valid_card()) == []

    def test_missing_required_metadata_field(self):
        card = _valid_card()
        del card["public_info"]["organization"]
        errors = _validate_card(card)
        assert any("organization" in e for e in errors)

    def test_empty_required_metadata_field(self):
        card = _valid_card()
        card["public_info"]["model"] = ""
        errors = _validate_card(card)
        assert any("model" in e for e in errors)

    def test_missing_required_reported_score_field(self):
        card = _valid_card()
        del card["reported_scores"]["physiq_mean"]
        errors = _validate_card(card)
        assert any("reported_scores.physiq_mean" in e for e in errors)

    def test_empty_required_reported_score_field(self):
        card = _valid_card()
        card["reported_scores"]["mse_std"] = ""
        errors = _validate_card(card)
        assert any("reported_scores.mse_std" in e for e in errors)

    def test_missing_reported_scores_block_rejected(self):
        card = _valid_card()
        del card["reported_scores"]
        errors = _validate_card(card)
        assert len([e for e in errors if "reported_scores." in e]) == 10

    def test_empty_required_cost_field(self):
        card = _valid_card()
        card["public_info"]["generation_cost"] = ""
        errors = _validate_card(card)
        assert any("public_info.generation_cost" in e for e in errors)

    def test_gpu_fields_not_required_for_api_model(self):
        card = _valid_card()
        assert card["public_info"]["availability"] == "api"
        for field in NON_API_REQUIRED_CARD_FIELDS:
            card["public_info"][field] = None
        errors = _validate_card(card)
        for field in NON_API_REQUIRED_CARD_FIELDS:
            assert not any(f"public_info.{field}" in e for e in errors)

    def test_gpu_fields_required_for_non_api_model(self):
        card = _valid_card()
        card["public_info"]["availability"] = "public"
        for field in NON_API_REQUIRED_CARD_FIELDS:
            card["public_info"][field] = None
        errors = _validate_card(card)
        for field in NON_API_REQUIRED_CARD_FIELDS:
            assert any(f"public_info.{field}" in e for e in errors)

    def test_upsampling_fields_not_required_when_no_upsampling(self):
        card = _valid_card()
        card["public_info"]["prompt_upsampling"] = False
        card["public_info"]["availability"] = "public"  # would otherwise require them
        for field in UPSAMPLING_REQUIRED_CARD_FIELDS + NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS:
            card["public_info"][field] = None
        errors = _validate_card(card)
        for field in UPSAMPLING_REQUIRED_CARD_FIELDS + NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS:
            assert not any(f"public_info.{field}" in e for e in errors)

    def test_upsample_cost_required_when_upsampling_performed(self):
        card = _valid_card()
        card["public_info"]["prompt_upsampling"] = True
        card["public_info"]["upsample_cost"] = None
        errors = _validate_card(card)
        assert any("public_info.upsample_cost" in e for e in errors)

    def test_upsampling_gpu_fields_not_required_for_api_model(self):
        card = _valid_card()
        card["public_info"]["prompt_upsampling"] = True
        assert card["public_info"]["availability"] == "api"
        for field in NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS:
            card["public_info"][field] = None
        errors = _validate_card(card)
        for field in NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS:
            assert not any(f"public_info.{field}" in e for e in errors)

    def test_upsampling_gpu_fields_required_for_non_api_model(self):
        card = _valid_card()
        card["public_info"]["prompt_upsampling"] = True
        card["public_info"]["availability"] = "public"
        for field in NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS:
            card["public_info"][field] = None
        errors = _validate_card(card)
        for field in NON_API_UPSAMPLING_REQUIRED_CARD_FIELDS:
            assert any(f"public_info.{field}" in e for e in errors)

    def test_terms_accepted_null_rejected(self):
        card = _valid_card()
        card["terms"]["terms_accepted"] = None
        errors = _validate_card(card)
        assert any("terms_accepted" in e for e in errors)

    def test_terms_accepted_false_rejected(self):
        card = _valid_card()
        card["terms"]["terms_accepted"] = False
        errors = _validate_card(card)
        assert any("terms_accepted" in e for e in errors)

    def test_wrong_terms_version_rejected(self):
        card = _valid_card()
        card["terms"]["terms_version"] = "0.9"
        errors = _validate_card(card)
        assert any("terms_version" in e for e in errors)
        assert any(CURRENT_TERMS_VERSION in e for e in errors)

    def test_missing_terms_version_rejected(self):
        card = _valid_card()
        card["terms"]["terms_version"] = None
        errors = _validate_card(card)
        assert any("terms_version" in e for e in errors)

    def test_empty_accepted_on_behalf_of_rejected(self):
        card = _valid_card()
        card["terms"]["accepted_on_behalf_of"] = ""
        errors = _validate_card(card)
        assert any("accepted_on_behalf_of" in e for e in errors)

    def test_empty_accepted_at_rejected(self):
        card = _valid_card()
        card["terms"]["accepted_at"] = ""
        errors = _validate_card(card)
        assert any("accepted_at" in e for e in errors)

    def test_single_warranty_false_rejected(self):
        card = _valid_card()
        card["warranties"]["no_ip_infringement"] = False
        errors = _validate_card(card)
        assert any("no_ip_infringement" in e for e in errors)

    def test_all_warranties_false_produces_five_errors(self):
        card = _valid_card()
        for field in card["warranties"]:
            card["warranties"][field] = False
        errors = _validate_card(card)
        warranty_errors = [e for e in errors if "warranties." in e]
        assert len(warranty_errors) == 5

    def test_missing_terms_block_rejected(self):
        card = _valid_card()
        del card["terms"]
        errors = _validate_card(card)
        assert any("terms_accepted" in e for e in errors)

    def test_missing_warranties_block_rejected(self):
        card = _valid_card()
        del card["warranties"]
        errors = _validate_card(card)
        assert len([e for e in errors if "warranties." in e]) == 5


# ── descriptions validation tests ────────────────────────────────────────────

def _make_descriptions_csv(
    dest_dir: Path,
    run_dir: Path,
    *,
    omit_col: str | None = None,
    drop_entries: int = 0,
) -> Path:
    """Write a minimal descriptions.csv covering all mp4s in run_dir."""
    mp4s = sorted(run_dir.glob("*.mp4"))
    rows = [
        {
            "scenario": f"scenario_{mp4.stem}.mp4",
            "description": f"Description for {mp4.stem}",
            "generated_video_name": mp4.name,
        }
        for mp4 in mp4s[: len(mp4s) - drop_entries]
    ]
    fieldnames = ["scenario", "description", "generated_video_name"]
    if omit_col:
        fieldnames = [c for c in fieldnames if c != omit_col]
        for row in rows:
            row.pop(omit_col, None)

    path = dest_dir / "descriptions.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.ffmpeg
class TestDescriptionsValidation:
    def test_valid_csv_passes(self, tmp_path, correct_video):
        run_dir = tmp_path / "run_01"
        run_dir.mkdir()
        _populate_run(run_dir, correct_video)
        desc = _make_descriptions_csv(tmp_path, run_dir)
        assert _validate_descriptions(desc, [run_dir]) == []

    def test_missing_required_column(self, tmp_path, correct_video):
        run_dir = tmp_path / "run_01"
        run_dir.mkdir()
        _populate_run(run_dir, correct_video)
        desc = _make_descriptions_csv(tmp_path, run_dir, omit_col="generated_video_name")
        errors = _validate_descriptions(desc, [run_dir])
        assert any("generated_video_name" in e for e in errors)

    def test_missing_description_column(self, tmp_path, correct_video):
        run_dir = tmp_path / "run_01"
        run_dir.mkdir()
        _populate_run(run_dir, correct_video)
        desc = _make_descriptions_csv(tmp_path, run_dir, omit_col="description")
        errors = _validate_descriptions(desc, [run_dir])
        assert any("description" in e for e in errors)

    def test_video_not_in_descriptions_reported_per_file(self, tmp_path, correct_video):
        run_dir = tmp_path / "run_01"
        run_dir.mkdir()
        _populate_run(run_dir, correct_video)
        desc = _make_descriptions_csv(tmp_path, run_dir, drop_entries=5)
        errors = _validate_descriptions(desc, [run_dir])
        assert len(errors) == 5

    def test_unparseable_csv_returns_single_error(self, tmp_path, correct_video):
        run_dir = tmp_path / "run_01"
        desc = tmp_path / "descriptions.csv"
        desc.write_bytes(b"\xff\xfe bad binary content \x00\x01")
        errors = _validate_descriptions(desc, [run_dir])
        assert len(errors) == 1
        assert "could not be parsed" in errors[0]

    def test_multiple_run_dirs_checked(self, tmp_path, correct_video):
        run_a = tmp_path / "run_01"
        run_b = tmp_path / "run_02"
        run_a.mkdir()
        run_b.mkdir()
        _populate_run(run_a, correct_video)
        _populate_run(run_b, correct_video)
        # descriptions missing 3 entries — each missing filename errors for both runs
        desc = _make_descriptions_csv(tmp_path, run_a, drop_entries=3)
        errors = _validate_descriptions(desc, [run_a, run_b])
        assert len(errors) == 6  # 3 missing × 2 runs
        assert any("run_01" in e for e in errors)
        assert any("run_02" in e for e in errors)
