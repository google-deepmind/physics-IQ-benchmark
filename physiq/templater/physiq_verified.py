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

from physiq.templater.base import BaseTemplater, register


@register("pvideo")
class PVideoTemplater(BaseTemplater):
    """
    Following official prompting guidelines for P-Video from: https://www.pruna.ai/p-video (2026-04-21).
    Official: subject, action, scene, camera, lighting, style, audio.

    Implementation: subject-action, scenesetup_description description {merge} scene description, camera description, style description, action description.

    """

    def generate_prompt(self, identifier):
        """Following official prompting guidelines for P-Video from: https://www.pruna.ai/p-video"""
        subjectaction_description = self.get_subjectaction_description(identifier)
        scene_description = self.get_scene_description(identifier)
        scenesetup_description = self.get_scenesetup_description(identifier)

        full_scene = " ".join(
            filter(self.filter_empty, [scenesetup_description, scene_description])
        )

        prompt = ", ".join(
            filter(
                self.filter_empty,
                [
                    subjectaction_description,
                    full_scene,
                    self.camera_description,
                    self.style_description,
                    self.action_description,
                ],
            )
        )
        return prompt


@register("sora2")
class SoraTemplater(BaseTemplater):
    """
    Following official prompting guidelines for Sora from: https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide (2026-04-24).
    Offical: style, scene, cinematography, action

    Implementation:

    Style: style

    scenesetup_description description {merge} scene description
    action_description

    Cinematography:
    camera

    Actions:
    - subject-action
    """

    def generate_prompt(self, identifier):
        """Following official prompting guidelines for Sora from: https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide (2026-04-24)"""
        subjectaction_description = self.get_subjectaction_description(identifier)
        scene_description = self.get_scene_description(identifier)
        scenesetup_description = self.get_scenesetup_description(identifier)

        full_scene = " ".join(
            filter(
                self.filter_empty,
                [scenesetup_description, scene_description, self.action_description],
            )
        )

        prompt = f"Style: {self.style_description}\n\n{full_scene}\n\nCinematography:\n{self.camera_description}\n\nActions:\n- {subjectaction_description}"
        return prompt
