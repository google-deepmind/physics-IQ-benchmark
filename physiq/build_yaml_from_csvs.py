from pathlib import Path

import pandas as pd
import yaml

CORE_SCENE_FIELDS = [
    # Original
    "description",
    "category",
    # Updated for factual errors
    "description_factual",
    # templater fields
    "subject-action",
    "pre-action scene description",
    "scene-description",
    "subject-action post-fix (for post-activation).",
    # original annotation
    "Factual error",
    "Temporal Error (I2V)",
    "omitted key information",
    "vague language",
]


def parse_scenario_filename(scenario: str) -> tuple[str, str, int, str]:
    """
    '0001_perspective-left_take-1_trimmed-ball-and-block-fall.mp4'
    -> (id=1, perspective='left', take=1, scene_slug='trimmed-ball-and-block-fall')
    """
    stem = scenario.removesuffix(".mp4")
    parts = stem.split("_", 3)
    id_ = int(parts[0])
    perspective = parts[1].removeprefix("perspective-")
    take = int(parts[2].removeprefix("take-"))
    scene_slug = parts[3]
    return id_, perspective, take, scene_slug


def build_yaml_from_csvs(
    original_csv: str | Path,
    patch_csv: str | Path,
    output_yaml: str | Path,
    extra_scene_fields: list[str] | None = None,
    original_sep: str = ",",
    patch_sep: str = ",",
) -> dict:
    extra_scene_fields = extra_scene_fields or []

    original = pd.read_csv(original_csv, sep=original_sep)
    patch = pd.read_csv(patch_csv, sep=patch_sep)
    original.columns = [c.strip() for c in original.columns]
    patch.columns = [c.strip() for c in patch.columns]

    df = original.merge(patch, on="scenario", how="left", suffixes=("_orig", ""))
    parsed = df["scenario"].apply(
        lambda s: pd.Series(
            parse_scenario_filename(s),
            index=["id", "perspective", "take", "scene_slug"],
        )
    )
    df = pd.concat([df, parsed], axis=1)

    all_scene_fields = CORE_SCENE_FIELDS + extra_scene_fields
    available_scene_fields = [f for f in all_scene_fields if f in df.columns]

    if len(all_scene_fields) != len(available_scene_fields):
        not_present_fields = all_scene_fields.copy()
        for f in available_scene_fields:
            not_present_fields.remove(f)
        print(f"The following columns are not present:\n{not_present_fields}")
        print(f"Inside the following columns of the dataframes:\n{df.columns} ")

    # Validate scene-level consistency
    for scene_slug, group in df.groupby("scene_slug"):
        for f in available_scene_fields:
            unique_vals = group[f].dropna().unique()
            if len(unique_vals) > 1:
                print(
                    f"Warning: {scene_slug} / {f} varies across rows: {unique_vals.tolist()}"
                )

    # Build list of scenes
    scene_list = []
    for scene_slug, group in df.groupby("scene_slug"):
        scene_data = {"scene": scene_slug}

        for f in available_scene_fields:
            if f in group.columns:
                val = group[f].dropna()
                if not val.empty and val.iloc[0] != "":
                    scene_data[f] = val.iloc[0]

        # Build list of takes
        takes = []
        for take_id, take_group in group.groupby("take"):
            take_entry = {"take": int(take_id)}
            for _, row in take_group.iterrows():
                take_entry[row["perspective"]] = row["id"]
            takes.append(take_entry)

        scene_data["takes"] = takes
        scene_list.append(scene_data)

    benchmark = {"scenes": scene_list}

    with open(output_yaml, "w") as f:
        yaml.dump(
            benchmark,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        )
    print(f"Written to {output_yaml}")
    return benchmark


if __name__ == "__main__":
    main_csv = "../descriptions/descriptions.csv"
    patch_csv = "../descriptions/physics-iq-prompts - descriptions.csv"
    output_yaml = "../descriptions/descriptions_test.yaml"

    build_yaml_from_csvs(main_csv, patch_csv, output_yaml)
