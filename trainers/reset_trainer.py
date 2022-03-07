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

"""Helper to sample listeners/speakers for resetting."""

import functools as fn

import chex
import jax

from jaxline import utils
from emergent_communication_at_scale import types
from emergent_communication_at_scale.utils import population_storage as ps


class ResetTrainer():
  """Class implementation for resetting speaker and listener agents."""

  def __init__(
      self,
      n_speakers: int,
      n_listeners: int,
  ):
    self._n_speakers = n_speakers
    self._n_listeners = n_listeners

  def reset(
      self,
      rng: chex.PRNGKey,
      games: types.GamesInputs,
      agent_storage: ps.PopulationStorage,
      game_init_fn,
      opt_speaker_init_fn,
      opt_listener_init_fn,
      reset_type: types.ResetMode,
  ):
    """Implements random reset."""
    # Gets first then broadcasts to ensure same rng for all devices at init.
    rng = utils.get_first(rng)
    rng_speaker, rng_listener, rng = jax.random.split(rng, num=3)
    rng = utils.bcast_local_devices(rng)

    reset_speaker_id = jax.random.randint(
        key=rng_speaker,
        shape=(1,),
        minval=0,
        maxval=self._n_speakers,
    )
    reset_listener_id = jax.random.randint(
        key=rng_listener,
        shape=(1,),
        minval=0,
        maxval=self._n_listeners,
    )

    agent_storage = self._initialize_pairs(
        rng_key=rng,
        speaker_id=reset_speaker_id.item(),
        listener_id=reset_listener_id.item(),
        games=games,
        agent_storage=agent_storage,
        game_init_fn=game_init_fn,
        opt_speaker_init_fn=opt_speaker_init_fn,
        opt_listener_init_fn=opt_listener_init_fn,
        reset_type=reset_type,
    )

    return agent_storage

  def _initialize_pairs(
      self,
      rng_key: chex.PRNGKey,
      speaker_id: int,
      listener_id: int,
      games: types.GamesInputs,
      agent_storage: ps.PopulationStorage,
      game_init_fn,
      opt_speaker_init_fn,
      opt_listener_init_fn,
      reset_type: types.ResetMode,
  ):
    """Initializes pair of agents."""
    params_init_pmap = jax.pmap(
        fn.partial(
            game_init_fn,
            training_mode=types.TrainingMode.TRAINING,
        ))
    opt_speaker_init_pmap = jax.pmap(opt_speaker_init_fn)
    opt_listener_init_pmap = jax.pmap(opt_listener_init_fn)

    # Init Params/States.
    joint_params, joint_states = params_init_pmap(init_games=games, rng=rng_key)

    # Init Opt state.
    speaker_opt_states = opt_speaker_init_pmap(joint_params.speaker)
    listener_opt_states = opt_listener_init_pmap(joint_params.listener)
    joint_opt_states = types.OptStates(
        speaker=speaker_opt_states, listener=listener_opt_states)

    if reset_type == types.ResetMode.PAIR:
      # Store reinitialized pair.
      agent_storage.store_pair(
          speaker_id=speaker_id,
          listener_id=listener_id,
          params=joint_params,
          states=joint_states,
          opt_states=joint_opt_states,
      )
    elif reset_type == types.ResetMode.SPEAKER:
      agent_storage.store_agent(
          agent_id=speaker_id,
          agent_name='speaker',
          params=joint_params.speaker,
          states=joint_states.speaker,
          opt_states=speaker_opt_states,
      )
      agent_storage.store_agent(
          agent_id=speaker_id,
          agent_name='target_speaker',
          params=joint_params.speaker,
          states=joint_states.speaker,
      )

    elif reset_type == types.ResetMode.LISTENER:
      agent_storage.store_agent(
          agent_id=listener_id,
          agent_name='listener',
          params=joint_params.listener,
          states=joint_states.listener,
          opt_states=listener_opt_states,
      )

    else:
      raise ValueError(f'Wrong type reset {reset_type}')

    return agent_storage
