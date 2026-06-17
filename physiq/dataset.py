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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

DESCRIPTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "descriptions" / "data.yaml"
)


@dataclass
class Scene(BaseModel):
    model_config = {"populate_by_name": True}

    scene: str
    category: str
    takes: list[dict[str, int]]  # [{take: 1, left: "0001", ...}, ...]

    # hyphenated YAML keys mapped to valid Python names via alias
    description: Optional[str] = None
    #
    description_factual: Optional[str] = Field(None, alias="description-factual")
    # fields that can be used for the templater
    subject_action: Optional[str] = Field(None, alias="subject-action")
    scene_description: Optional[str] = Field(None, alias="scene-description")
    pre_action_description: Optional[str] = Field(
        None, alias="pre-action-scene-description"
    )
    subject_action_postfix: Optional[str] = Field(None, alias="subject-action-postfix")

    # extra annotation fields — carried through, not used in CSV export
    factual_error: Optional[str] = Field(None, alias="factual-error")
    temporal_error: Optional[str] = Field(None, alias="temporal-error-(i2v)")
    omitted_key_information: Optional[str] = Field(
        None, alias="omitted-key-information"
    )
    vague_language: Optional[str] = Field(None, alias="vague-language")


@dataclass
class Benchmark(BaseModel):
    scenes: list[Scene]

    @classmethod
    def from_yaml(cls, path: str) -> "Benchmark":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @staticmethod
    def _infer_test_scenario(
        id_: int, perspective: str, take: int, scene_name: str
    ) -> str:
        return (
            f"{id_:04}_perspective-{perspective}_take-{take}_trimmed-{scene_name}.mp4"
        )

    @staticmethod
    def _infer_gt_fn_video(
        id_: int, perspective: str, take: int, scene_name: str, fps: str | int
    ):
        return f"{id_:04}_testing-videos_{fps}FPS_perspective-{perspective}_take-{take}_trimmed-{scene_name}.mp4"

    @staticmethod
    def _infer_fn_mask(
        id_: int, perspective: str, take: int, scene_name: str, fps: str | int
    ):
        return f"{id_:04}_video-masks_{fps}FPS_perspective-{perspective}_take-{take}_trimmed-{scene_name}.mp4"

    @staticmethod
    def _infer_generated(id_: int, perspective: str, scene_name: str) -> str:
        return f"{id_:04}_perspective-{perspective}_trimmed-{scene_name}.mp4"

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for scene in self.scenes:
            scene_meta = scene.model_dump(exclude={"takes"})
            scene_meta["description_updated"] = (
                scene.description_factual
                if scene.description_factual
                else scene.description
            )
            for take_entry in scene.takes:
                take_id = take_entry["take"]
                for perspective in take_entry:
                    if perspective == "take":
                        continue
                    id_ = take_entry.get(perspective)
                    if id_ is None:
                        continue
                    rows.append(
                        {
                            **scene_meta,
                            "take": take_id,
                            "perspective": perspective,
                            "id": id_,
                            "scenario": self._infer_test_scenario(
                                id_, perspective, take_id, scene.scene
                            ),
                            "generated_video_name": self._infer_generated(
                                id_, perspective, scene.scene
                            ),
                        }
                    )
        df = pd.DataFrame(rows)
        return df.sort_values(by="id").reset_index(drop=True)

    def build_original_descriptions(self) -> pd.DataFrame:
        cols = ["scenario", "description", "category", "generated_video_name"]
        return self.to_dataframe()[cols]


def get_benchmark() -> Benchmark:
    return Benchmark.from_yaml(DESCRIPTIONS_PATH)


if __name__ == "__main__":
    benchmark = Benchmark.from_yaml(DESCRIPTIONS_PATH)

    csv_gen = benchmark.build_original_descriptions()
    csv = pd.read_csv(DESCRIPTIONS_PATH.parent / "descriptions_original.csv")

    import joblib
    import numpy as np

    h1 = joblib.hash(csv)
    h2 = joblib.hash(csv_gen)

    print("Hash comparison identical")
    print(h1 == h2)

    print("\n" * 2)
    mask = ~(csv_gen == csv)

    csv_gen_diff = csv_gen[mask].dropna(how="all")
    csv_diff = csv[mask].dropna(how="all")

    print("CSV Gen Diff")
    print(csv_gen_diff)
    print("CSV Orig Diff")
    print(csv_diff)
