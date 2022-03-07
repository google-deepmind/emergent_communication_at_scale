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

"""Head networks."""
from typing import Any, Iterable, Optional, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from emergent_communication_at_scale import types


def head_factory(
    head_type: Union[types.SpeakerLossType, types.ListenerLossType],
    head_kwargs: types.Config,
    kwargs: types.Config,
    name: str,
) -> Any:
  """Builds head from name and kwargs."""

  loss_specific_kwargs = kwargs.get(head_type, dict())
  all_kwargs = {**head_kwargs, **loss_specific_kwargs}

  if head_type == types.SpeakerHeadType.POLICY:
    head = PolicyHead(name=name, **all_kwargs)
  elif head_type == types.SpeakerHeadType.POLICY_QVALUE:
    head = PolicyQValueHead(name=name, **all_kwargs)
  elif head_type == types.SpeakerHeadType.POLICY_QVALUE_DUELING:
    head = PolicyQValueDuelingHead(name=name, **all_kwargs)
  elif head_type == types.ListenerHeadType.MULTIMLP:
    head = MultiMlpHead(name=name, **all_kwargs)
  elif head_type == types.ListenerHeadType.CPC:
    head = CpcHead(name=name, **all_kwargs)
  else:
    raise ValueError(f'Incorrect head type {head_type}.')
  return head


class MultiMlpHead(hk.Module):
  """MultiMLP head."""

  def __init__(
      self,
      hidden_sizes: Iterable[int],
      task: types.Task,
      name: Optional[str] = 'multi_mlp_head',
  ):
    super().__init__(name)
    self._hidden_sizes = tuple(hidden_sizes)
    self._task = task

  def __call__(
      self,
      message_rep: chex.Array,
      games: types.Games,
  ) -> types.ListenerHeadOutputs:
    assert self._task in games.labels
    mlps = jax.tree_map(
        lambda x: hk.nets.MLP(output_sizes=self._hidden_sizes + (x.shape[-1],)),
        games.labels[self._task])
    predictions = jax.tree_map(lambda x, m=message_rep: x(m), mlps)
    return types.ListenerHeadOutputs(predictions=predictions, targets=None)


class PolicyHead(hk.Module):
  """Policy head."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Iterable[int],
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name)
    self._policy_head = hk.nets.MLP(
        output_sizes=tuple(hidden_sizes) + (num_actions,))

  def __call__(self, inputs) -> types.SpeakerHeadOutputs:
    return types.SpeakerHeadOutputs(policy_logits=self._policy_head(inputs))


class DuelingHead(hk.Module):
  """Dueling value head."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Iterable[int],
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name)
    self._value_net = hk.nets.MLP(tuple(hidden_sizes) + (1,))
    self._advantage_net = hk.nets.MLP(tuple(hidden_sizes) + (num_actions,))

  def __call__(self, inputs) -> types.DuelingHeadOutputs:
    state_value = self._value_net(inputs)
    advantage = self._advantage_net(inputs)
    mean_advantage = jnp.mean(advantage, axis=-1, keepdims=True)
    q_values = state_value + advantage - mean_advantage
    return types.DuelingHeadOutputs(q_values=q_values, value=state_value)


class PolicyQValueHead(hk.Module):
  """Policy and Qvalue head."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Iterable[int],
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name)
    self._policy_head = hk.nets.MLP(
        output_sizes=tuple(hidden_sizes) + (num_actions,))
    self._value_head = hk.nets.MLP(
        output_sizes=tuple(hidden_sizes) + (num_actions,))
    self._q_value_head = DuelingHead(
        num_actions=num_actions, hidden_sizes=hidden_sizes)

  def __call__(self, inputs) -> types.SpeakerHeadOutputs:
    dueling_head_outputs = self._q_value_head(inputs)
    return types.SpeakerHeadOutputs(
        policy_logits=self._policy_head(inputs),
        q_values=dueling_head_outputs.q_values,
        value=dueling_head_outputs.value,
    )


class PolicyQValueDuelingHead(hk.Module):
  """Policy and Qvalue head."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Iterable[int],
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name)
    self._policy_head = hk.nets.MLP(
        output_sizes=tuple(hidden_sizes) + (num_actions,))
    self._value_head = hk.nets.MLP(output_sizes=tuple(hidden_sizes) + (1,))
    self._q_value_head = DuelingHead(
        num_actions=num_actions, hidden_sizes=hidden_sizes)

  def __call__(self, inputs) -> types.SpeakerHeadOutputs:
    return types.SpeakerHeadOutputs(
        policy_logits=self._policy_head(inputs),
        q_values=self._q_value_head(inputs).q_values,
        value=self._value_head(inputs),
    )


class CpcHead(hk.Module):
  """CPC head."""

  def __init__(
      self,
      hidden_sizes: Iterable[int],
      name: Optional[str] = 'cpc_head',
  ) -> None:
    super().__init__(name)
    self.proj_pred = hk.nets.MLP(output_sizes=hidden_sizes)
    self.proj_target = hk.nets.MLP(output_sizes=hidden_sizes)

  def __call__(
      self,
      message_rep: chex.Array,
      games: types.Games,
  ) -> types.ListenerHeadOutputs:

    # Takes the second view if it exist, otherwise, takes same input view.
    if types.Task.DISCRIMINATION in games.labels:
      target_inputs = games.labels[types.Task.DISCRIMINATION]
    else:
      target_inputs = games.speaker_inp

    return types.ListenerHeadOutputs(
        predictions=self.proj_pred(message_rep),
        targets=self.proj_target(target_inputs),
    )
