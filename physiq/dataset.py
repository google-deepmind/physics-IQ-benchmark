from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator
from typing import Optional
import yaml, pandas as pd
from pathlib import Path 



PERSPECTIVES = ["left", "center", "right"]
DESCRIPTIONS_PATH = Path(__file__).resolve().parent.parent / "descriptions" / "descriptions.yaml"

@dataclass
class Scene(BaseModel):
    model_config = {"populate_by_name": True}

    scene: str
    category: str
    takes: list[dict[str, int]] # [{take: 1, left: "0001", ...}, ...]

    # hyphenated YAML keys mapped to valid Python names via alias
    description: Optional[str] = None
    # 
    description_factual: Optional[str] = Field(None, alias="description-factual")
    # fields that can be used for the templater
    subject_action: Optional[str] = Field(None, alias="subject-action")
    scene_description: Optional[str] = Field(None, alias="scene-description")
    pre_action_description: Optional[str] = Field(None, alias="pre-action-scene-description")
    subject_action_postfix: Optional[str] = Field(None, alias="subject-action-postfix")

    # extra annotation fields — carried through, not used in CSV export
    factual_error: Optional[str] = Field(None, alias="factual-error")
    temporal_error: Optional[str] = Field(None, alias="temporal-error-(i2v)")
    omitted_key_information: Optional[str] = Field(None, alias="omitted-key-information")
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
    def _infer_test_scenario(id_: int, perspective: str, take: int, scene_name: str) -> str:
        return f"{id_:04}_perspective-{perspective}_take-{take}_trimmed-{scene_name}.mp4"

    @staticmethod
    def _infer_generated(id_: int, perspective: str, scene_name: str) -> str:
        return f"{id_:04}_perspective-{perspective}_trimmed-{scene_name}.mp4"
    

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for scene in self.scenes:
            scene_meta = scene.model_dump(exclude={"takes"})
            scene_meta["description_updated"] = scene.description_factual if scene.description_factual else scene.description
            for take_entry in scene.takes:
                take_id = take_entry["take"]
                for perspective in take_entry:
                    if perspective == "take":
                        continue
                    id_ = take_entry.get(perspective)
                    if id_ is None:
                        continue
                    rows.append({
                        **scene_meta,
                        "take": take_id,
                        "perspective": perspective,
                        "id": id_,
                        "scenario": self._infer_test_scenario(id_, perspective, take_id, scene.scene),
                        "generated_video_name": self._infer_generated(id_, perspective, scene.scene),
                    })
        df = pd.DataFrame(rows)
        return df.sort_values(by="id").reset_index(drop=False)
    
    def build_original_descriptions(self)->pd.DataFrame:
        cols = ["scenario", "description", "category", "generated_video_name"]
        return self.to_dataframe()[cols]
    


if __name__ == "__main__":
    with open(DESCRIPTIONS_PATH, "r") as f:
        scenes = yaml.safe_load(f)

    benchmark = Benchmark.from_yaml(DESCRIPTIONS_PATH)

    csv_gen = benchmark.build_original_descriptions()
    csv = pd.read_csv(DESCRIPTIONS_PATH.parent/"descriptions.csv")


    import joblib
    import numpy as np
    h1 = joblib.hash(csv)
    h2 = joblib.hash(csv_gen)

    print("Hash comparison identical")
    print(h1 == h2)

    print("\n"*2)
    print(csv_gen == csv)

    mask = ~(csv_gen==csv)

    print(np.sum((~mask)))

    csv_gen_diff = csv_gen[mask].dropna(how="all")
    csv_diff = csv[mask].dropna(how="all")

    print("CSV Gen")
    print(csv_gen_diff)
    print("CSV")
    print(csv_diff)

    # benchmark = Benchmark(DESCRIPTIONS_PATH)
    # df = benchmark.to_dataframe()
    # print(df)
    import IPython; IPython.embed()
