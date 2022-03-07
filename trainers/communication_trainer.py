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

"""Helper to sample a population."""

import abc
import functools
from typing import Tuple

import chex
import jax
from jaxline import utils
from emergent_communication_at_scale import types
from emergent_communication_at_scale.utils import population_storage as ps
from emergent_communication_at_scale.utils import utils as emcom_utils


@jax.pmap
def _split_keys_pmap(key):
  return tuple(jax.random.split(key))


class AbstractCommunicateTrainer(abc.ABC):
  """Abstract class implementation for training agents."""

  @abc.abstractmethod
  def communicate(
      self,
      rng: chex.PRNGKey,
      games: types.GamesInputs,
      agent_storage: ps.PopulationStorage,
  ):
    pass


class BasicTrainer(AbstractCommunicateTrainer):
  """Sample trainer that simply loop over sampled agent pairs."""

  def __init__(
      self,
      update_fn,
      n_speakers: int,
      n_listeners: int,
      num_agents_per_step: int,
  ) -> None:

    # Stores key values.
    self._n_speakers = n_speakers
    self._n_listeners = n_listeners
    self._num_agents_per_step = num_agents_per_step

    # Prepares pmap functions.
    # Special pmap wrapper to correctly handle sampling across devices.
    self._pmap_sampling = emcom_utils.run_and_broadcast_to_all_devices(
        self._sample_fn)
    self._pmap_update_fn = jax.pmap(
        functools.partial(
            update_fn,
            training_mode=types.TrainingMode.TRAINING,
            is_sharded_update=True),
        axis_name='i', donate_argnums=(0, 1, 2))

  def _sample_fn(self, rng: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
    """Basic sampling function."""

    speaker_rng, listener_rng = jax.random.split(rng)
    speaker_ids = jax.random.choice(
        key=speaker_rng,
        a=self._n_speakers,
        replace=True,
        shape=[self._num_agents_per_step],
    )

    listener_ids = jax.random.choice(
        key=listener_rng,
        a=self._n_listeners,
        replace=True,
        shape=[self._num_agents_per_step],
    )

    return speaker_ids, listener_ids

  def communicate(
      self,
      rng: chex.PRNGKey,
      games: types.GamesInputs,
      agent_storage: ps.PopulationStorage,
  ) -> Tuple[types.Config, ps.PopulationStorage]:
    """Performs one training step by looping over agent pairs."""

    # Step 1: samples the speaker/listener idx.
    sampling_rng, rng = _split_keys_pmap(rng)

    sampling_rng = utils.get_first(sampling_rng)

    speaker_ids, listener_ids = self._pmap_sampling(sampling_rng)
    chex.assert_tree_shape_prefix((speaker_ids, listener_ids),
                                  (self._num_agents_per_step,))

    # Step 2: executes a pmap update per speaker/listener pairs.
    scalars = None
    for (speaker_id, listener_id) in zip(speaker_ids, listener_ids):

      # Next rng.
      update_rng, rng = _split_keys_pmap(rng)

      # Load agent params.
      params, states, opt_states = agent_storage.load_pair(
          speaker_id=speaker_id.item(),  # `.item()` gets the scalar value.
          listener_id=listener_id.item())

      # Performs update function (forward/backward pass).
      new_params, new_states, new_opt_states, scalars = self._pmap_update_fn(
          params,
          states,
          opt_states,
          games,
          update_rng,
      )

      # Updates params in storage.
      agent_storage.store_pair(
          speaker_id=speaker_id.item(),
          listener_id=listener_id.item(),
          params=new_params,
          states=new_states,
          opt_states=new_opt_states,
      )

    # Returns the scalar of the last random pair without the pmaped dimension.
    scalars = utils.get_first(scalars)

    return scalars, agent_storage
