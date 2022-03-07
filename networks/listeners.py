# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines different types of listeners."""

import haiku as hk
from emergent_communication_at_scale import types
from emergent_communication_at_scale.networks import cores
from emergent_communication_at_scale.networks import heads
from emergent_communication_at_scale.networks import torsos


class Listener(hk.Module):
  """General Recurrent Language Listener."""

  def __init__(
      self,
      torso_config: types.Config,
      core_config: types.Config,
      head_config: types.Config,
      name: str = 'listener',
  ) -> None:
    super().__init__(name=name)
    self._torso = torsos.torso_factory(**torso_config, name='torso')
    self._core = cores.core_factory(**core_config, name='core')
    self._head = heads.head_factory(**head_config, name='head')
    # Adding a dummy state to listeners to have symmetric speakers/listeners.
    hk.get_state('dummy_state', shape=(), init=hk.initializers.Constant(0.))

  def __call__(
      self,
      games: types.Games,
      training_mode: types.TrainingMode,
  ) -> types.ListenerOutputs:
    """Unroll Listener over token of messages."""
    del training_mode
    batch_size = games.speaker_outputs.action.shape[0]

    # Torso
    embedded_message = self._torso(games.speaker_outputs.action)

    # Core
    initial_state = self._core.initial_state(batch_size)
    core_out, _ = hk.static_unroll(
        self._core, embedded_message, initial_state, time_major=False)
    core_out = core_out[:, -1, :]  # Only consider the last repr. of core

    # Head
    listener_head_outputs = self._head(core_out, games)

    return types.ListenerOutputs(
        predictions=listener_head_outputs.predictions,
        targets=listener_head_outputs.targets,
    )
