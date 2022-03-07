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

"""Defines different types of speakers."""
from typing import Optional

import chex
import haiku as hk
import jax.numpy as jnp
import rlax
from emergent_communication_at_scale import types
from emergent_communication_at_scale.networks import cores
from emergent_communication_at_scale.networks import heads
from emergent_communication_at_scale.networks import torsos


class Speaker(hk.Module):
  """General Reccurrent Language Speaker."""

  def __init__(
      self,
      length: int,
      vocab_size: int,
      torso_config: types.Config,
      embedder_config: types.Config,
      core_config: types.Config,
      head_config: types.Config,
      name: str = 'speaker',
  ) -> None:
    super().__init__(name=name)
    self._length = length
    self._vocab_size = vocab_size
    self._core_config = core_config
    self._torso = torsos.torso_factory(**torso_config, name='torso')
    self._embedder = torsos.torso_factory(**embedder_config, name='embedder')
    self._core = cores.core_factory(**core_config, name='core')
    self._head = heads.head_factory(**head_config, name='head')
    hk.get_state('avg_score', shape=(), init=hk.initializers.Constant(0.))
    hk.get_state('counter', shape=(), init=hk.initializers.Constant(0.))

  def __call__(
      self,
      games: types.Games,
      training_mode: types.TrainingMode,
      actions_to_follow: Optional[chex.Array] = None,
  ) -> types.SpeakerOutputs:

    batch_size = games.speaker_inp.shape[0]
    continuous_embedding = self._torso(games.speaker_inp)
    state = cores.ToCoreState(prototype=self._core.initial_state(batch_size))(
        continuous_embedding)
    sos_symbol = self._vocab_size  # The last vocabulary entry is Start of Sequ.
    prev_token = jnp.array([sos_symbol] * batch_size)

    if training_mode == types.TrainingMode.FORCING:
      assert actions_to_follow is not None

    action_list = []
    policy_logits_list = []
    action_log_prob_list = []
    entropy_list = []
    q_values_list = []
    value_list = []

    distr = rlax.softmax(temperature=1)

    for i in range(self._length):
      step_input = self._embedder(prev_token)
      output, state = self._core(step_input, state)
      head_outputs = self._head(output)
      policy_logits = head_outputs.policy_logits
      q_values = head_outputs.q_values
      value = head_outputs.value

      if types.TrainingMode.TRAINING:
        rng = hk.next_rng_key()
        action = distr.sample(key=rng, logits=policy_logits)
      elif types.TrainingMode.EVAL:
        action = jnp.argmax(policy_logits, axis=-1)
      elif types.TrainingMode.FORCING:
        action = actions_to_follow[..., i]
      else:
        raise ValueError(f'Unknown training mode: {training_mode}.')

      action_log_prob = distr.logprob(logits=policy_logits, sample=action)
      entropy = distr.entropy(policy_logits)

      prev_token = action

      action_list.append(action)
      policy_logits_list.append(policy_logits)
      entropy_list.append(entropy)
      action_log_prob_list.append(action_log_prob)
      q_values_list.append(q_values)
      value_list.append(value)

    def maybe_stack_fn(x):
      if x[0] is None:
        return None
      else:
        return jnp.stack(x, axis=-1)

    return types.SpeakerOutputs(
        action=maybe_stack_fn(action_list),
        action_log_prob=maybe_stack_fn(action_log_prob_list),
        entropy=maybe_stack_fn(entropy_list),
        policy_logits=maybe_stack_fn(policy_logits_list),
        q_values=maybe_stack_fn(q_values_list),
        value=maybe_stack_fn(value_list),
    )
