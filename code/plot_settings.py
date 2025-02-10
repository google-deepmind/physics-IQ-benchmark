# Copyright 2025 DeepMind Technologies Limited
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

"""Enable custom colors and names when plotting models. This file can be edited by users."""

DEFAULT_COLOR = 'grey'

class Model():

  def __init__(self, model_name, plotting_name = None, color = DEFAULT_COLOR):
    """Customize aspects like plotting name or color when plotting models."""
 
    self.model_name = model_name
    if plotting_name is None:
      self.plotting_name = model_name
    else:
      self.plotting_name = plotting_name
    self.color = color


_MODELS = []

####################################################################
# ADD CUSTOM MODELS HERE AS DESIRED
####################################################################

_MODELS.append(
  Model(model_name='VideoPoet_multiframe',
        plotting_name='VideoPoet (multiframe)',
        color='#CC5500'))
_MODELS.append(
  Model(model_name='VideoPoet_i2v',
        plotting_name='VideoPoet (i2v)',
        color='#FF7F0E'))
_MODELS.append(
  Model(model_name='Lumiere_multiframe',
        plotting_name='Lumiere (multiframe)',
        color='#1C69A7'))
_MODELS.append(
  Model(model_name='Lumiere_i2v',
        plotting_name='Lumiere (i2v)',
        color='#17BECF'))
_MODELS.append(
  Model(model_name='Runway',
        plotting_name='Runway Gen 3 (i2v)',
        color='#2ca02c'))
_MODELS.append(
  Model(model_name='Pika',
        plotting_name='Pika 1.0 (i2v)',
        color='#FFD700'))
_MODELS.append(
  Model(model_name='SVD',
        plotting_name='Stable Video Diffusion (i2v)',
        color='#9467BD'))
_MODELS.append(
  Model(model_name='Sora',
        plotting_name='Sora (i2v)',
        color='#ff0606'))
####################################################################
# END OF CUSTOMIZATION BLOCK, KEEP REST UNCHANGED
####################################################################

_MODEL_TO_COLOR = {m.model_name: m.color for m in _MODELS}
_MODEL_TO_PLOTTING_NAME = {m.model_name: m.plotting_name for m in _MODELS}

def model_to_color(model_name):
  return _MODEL_TO_COLOR.get(model_name, DEFAULT_COLOR)

def model_to_plotting_name(model_name):
  return _MODEL_TO_PLOTTING_NAME.get(model_name, model_name)

