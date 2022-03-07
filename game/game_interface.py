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

"""Dummy file to have Mock object for testing."""

import abc
from typing import Iterator, Union
import jax
import jax.numpy as jnp
import numpy as np
from emergent_communication_at_scale import types

NumpyValue = Union[np.ndarray, np.generic, bytes]
GameIterator = Iterator[types.GamesInputs]

# This is a very minimal game interface, so that we can sweep over game.


def batch_size_per_device(batch_size: int, num_devices: int):
  """Return the batch size per device."""
  per_device_batch_size, ragged = divmod(batch_size, num_devices)

  if ragged:
    msg = 'batch size must be divisible by num devices, got {} and {}.'
    raise ValueError(msg.format(per_device_batch_size, num_devices))

  return per_device_batch_size


def dispatch_per_device(games):
  """Helper fo split dataset per device."""

  num_games = list(games._asdict().values())[0].shape[0]
  num_devices = jax.local_device_count()

  batch_size = batch_size_per_device(num_games, num_devices=num_devices)

  def dispatch(x):
    if hasattr(x, 'shape'):
      return jnp.reshape(x, [num_devices, batch_size] + list(x.shape[1:]))
    else:
      return x

  games_per_device = jax.tree_map(dispatch, games)

  return games_per_device


class Game(abc.ABC):
  """Interface for game with terminal reward."""

  def __init__(self,
               train_batch_size,
               eval_batch_size):
    self._train_batch_size = train_batch_size
    self._eval_batch_size = eval_batch_size

    if self._eval_batch_size <= 2:
      raise ValueError(f'Eval batch size must be greater than 2 to compute '
                       f'topography similarity. Got {self._eval_batch_size}')

  @property
  def train_batch_size(self) -> int:
    return self._train_batch_size

  @property
  def eval_batch_size(self) -> int:
    return self._eval_batch_size

  @abc.abstractmethod
  def get_training_games(self, rng) -> GameIterator:
    pass

  @abc.abstractmethod
  def get_evaluation_games(self, mode: str = 'test') -> GameIterator:
    pass

  def evaluate(self, prediction, target):
    pass
