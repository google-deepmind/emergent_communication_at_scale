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

"""Reccurent networks."""

from typing import Optional, Sequence, Tuple

import chex
import haiku as hk
import jax.numpy as jnp
from emergent_communication_at_scale import types


def core_factory(
    core_type: types.CoreType,
    core_kwargs: types.Config,
    name: str,
) -> hk.RNNCore:
  """Builds core from name and kwargs."""
  if core_type == types.CoreType.LSTM:
    core = hk.LSTM(name=name, **core_kwargs)
  elif core_type == types.CoreType.GRU:
    core = hk.GRU(name=name, **core_kwargs)
  elif core_type == types.CoreType.IDENTITY:
    core = CustomedIdentityCore(name=name, **core_kwargs)
  else:
    raise ValueError(f'Incorrect core type {core_type}.')
  return core


class CustomedIdentityCore(hk.RNNCore):
  """A recurrent core that forwards the inputs and a mock state.

  This is commonly used when switching between recurrent and feedforward
  versions of a model while preserving the same interface.
  """

  def __init__(
      self,
      hidden_size: int,
      name: Optional[str] = None,
  ) -> None:
    """Constructs an CustomedIdentityCore.

    Args:
      hidden_size: Hidden layer size.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.hidden_size = hidden_size

  def __call__(
      self,
      inputs: Sequence[chex.Array],
      state: hk.LSTMState,
  ) -> Tuple[Sequence[chex.Array], hk.LSTMState]:
    return inputs, state

  def initial_state(self, batch_size: Optional[int]) -> hk.LSTMState:
    return hk.LSTMState(
        hidden=jnp.zeros([batch_size, self.hidden_size]),
        cell=jnp.zeros([batch_size, self.hidden_size]),
    )


class ToCoreState(hk.Module):
  """Module to get a core state from an embedding."""

  def __init__(
      self,
      prototype: types.RNNState,
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name=name)
    self._prototype = prototype

  def __call__(self, embedding: chex.Array) -> types.RNNState:
    if isinstance(self._prototype, hk.LSTMState):
      return _ToLSTMState(self._prototype.cell.shape[-1])(embedding)
    elif isinstance(self._prototype, chex.Array):
      return hk.Linear(output_size=self._prototype.shape[-1])(embedding)
    elif not self._prototype:
      return ()
    else:
      raise ValueError(f'Invalid prototype type for core state '
                       f'{type(self._prototype)}.')


class _ToLSTMState(hk.Module):
  """Module linearly mapping a tensor to an hk.LSTMState."""

  def __init__(self, output_size: int) -> None:
    super().__init__(name='to_lstm_state')
    self._linear = hk.Linear(output_size=2 * output_size)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    h, c = jnp.split(self._linear(inputs), indices_or_sections=2, axis=-1)
    return hk.LSTMState(h, c)
