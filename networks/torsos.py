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

"""Torso networks."""

import chex
import haiku as hk
from emergent_communication_at_scale import types


def torso_factory(
    torso_type: types.TorsoType,
    torso_kwargs: types.Config,
    name: str,
):
  """Builds torso from name and kwargs."""
  if torso_type == types.TorsoType.DISCRETE:
    torso = DiscreteTorso(name=name, **torso_kwargs)
  elif torso_type == types.TorsoType.MLP:
    torso = hk.nets.MLP(name=name, **torso_kwargs)
  elif torso_type == types.TorsoType.IDENTITY:
    torso = Identity(name=name)
  else:
    raise ValueError(f'Incorrect torso type {torso_type}.')
  return torso


class DiscreteTorso(hk.Module):
  """Torso for discrete entries."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      mlp_kwargs: types.Config,
      name: str,
  ) -> None:
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._embed_dim = embed_dim
    self._mlp_kwargs = mlp_kwargs

  def __call__(self, x: chex.Array) -> chex.Array:
    #  chex.assert_rank(x, 1)
    h = hk.Embed(
        vocab_size=self._vocab_size,
        embed_dim=self._embed_dim,
    )(x,)
    return hk.nets.MLP(**self._mlp_kwargs)(h)


class Identity(hk.Module):
  """Torso for Identity."""

  def __call__(self, x: chex.Array) -> chex.Array:
    return x
