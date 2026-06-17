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

from pathlib import Path

import pandas as pd

REGISTRY: dict[str, type["BaseTemplater"]] = {}


def register(name: str):
    def decorator(cls):
        REGISTRY[name] = cls
        return cls

    return decorator


@register("base")
class BaseTemplater:
    def __init__(self, df: pd.DataFrame, use_action_suffix: bool = True):
        self.df = df.copy()
        self.use_action_suffix = use_action_suffix

    @classmethod
    def from_csv(cls, df_path, use_action_suffix: bool = True) -> "BaseTemplater":
        try:
            df = pd.read_csv(df_path, sep=";")
        except Exception as e:
            try:
                df = pd.read_csv(df_path)
            except Exception as e2:
                raise ValueError(
                    f"Failed to read CSV with both ';' and default separators: {e}, {e2}"
                )
        return cls(df, use_action_suffix=use_action_suffix)

    @property
    def camera_description(self) -> str:
        return "Static locked-off single-shot with fixed frame throughout filmed with constant framerate in real-time."

    @property
    def style_description(self) -> str:
        return "The scene shows a realistic scientific demonstration."

    @property
    def action_description(self) -> str:
        return "The scene only contains the described setup and actions."

    @property
    def identifier_key(self) -> str:
        return "scenario"

    @property
    def subjact_key(self) -> str:
        return "subject_action"

    @property
    def scene_key(self) -> str:
        return "scene_description"

    @property
    def scenesetup_key(self) -> str:
        return "pre_action_description"

    @property
    def subjactsuffix_key(self) -> str:
        return "subject_action_postfix"

    def get_identifier_data(self, identifier) -> pd.Series:
        row = self.df[self.df[self.identifier_key] == identifier].iloc[0]
        return row

    def data_field_handler(self, data, identifier, field_name, necessary=True):
        if pd.isna(data) and necessary:
            raise ValueError(f"{field_name} is missing for identifier {identifier}")
        elif pd.isna(data):
            return data if pd.notna(data) else ""
        else:
            return data

    def get_subjectaction_description(self, identifier) -> str:
        idenifier_data = self.get_identifier_data(identifier)
        action = idenifier_data[self.subjact_key]
        action = self.data_field_handler(
            action, identifier, self.subjact_key, necessary=False
        )
        if pd.isna(action):
            raise ValueError(
                f"Subject-action description is missing for identifier {identifier}"
            )
        if self.use_action_suffix:
            action_suffix = idenifier_data[self.subjactsuffix_key]
            action_suffix = self.data_field_handler(
                action_suffix, identifier, "subject-action post-fix", necessary=False
            )
            if action_suffix != "":
                action = f"{action} {action_suffix}"
        return action

    def get_scene_description(self, identifier) -> str:
        data = self.get_identifier_data(identifier)[self.scene_key]
        return self.data_field_handler(
            data, identifier, self.scene_key, necessary=False
        )

    def get_scenesetup_description(self, identifier) -> str:
        data = self.get_identifier_data(identifier)[self.scenesetup_key]
        return self.data_field_handler(
            data, identifier, self.scenesetup_key, necessary=False
        )

    def filter_empty(self, x) -> str:
        return x != ""

    def generate_prompt(self, identifier) -> str:
        subjectaction_description = self.get_subjectaction_description(identifier)
        scene_description = self.get_scene_description(identifier)
        scenesetup_description = self.get_scenesetup_description(identifier)

        prompt = " ".join(
            filter(
                self.filter_empty,
                [
                    self.camera_description,
                    self.style_description,
                    self.action_description,
                    scenesetup_description,
                    scene_description,
                    subjectaction_description,
                ],
            )
        )

        return prompt

    def get_non_empty(self, key: str) -> dict:
        """Returns a dictionary with counts of unique values in the specified column."""
        if key not in self.df.columns:
            raise ValueError(f"Key '{key}' not found in DataFrame columns.")
        # get identifiers where the key is not empty
        non_empty_identifiers = self.df[~self.df[key].isna()][
            self.identifier_key
        ].tolist()
        return non_empty_identifiers
