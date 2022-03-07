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

"""Creates a class to store and load the param of a population of agents."""

import functools as fn
from typing import Optional, Tuple

import chex
import haiku as hk
import jax
from jaxline import utils
import optax
from emergent_communication_at_scale import types


class PopulationStorage:
  """Stores the population params and states."""

  def __init__(
      self,
      n_speakers: int,
      n_listeners: int,
  ) -> None:

    self._n_speakers = n_speakers
    self._n_listeners = n_listeners

    self._params: types.AllParams = dict(
        speaker=[None] * n_speakers,
        listener=[None] * n_listeners,
        target_speaker=[None] * n_speakers,
    )

    self._states: types.AllStates = dict(
        speaker=[None] * n_speakers,
        listener=[None] * n_listeners,
        target_speaker=[None] * n_speakers,
    )

    self._opt_states: types.AllOptStates = dict(
        speaker=[None] * n_speakers,
        listener=[None] * n_listeners,
    )

  @property
  def params(self):
    return self._params

  @property
  def states(self):
    return self._states

  @property
  def opt_states(self):
    return self._opt_states

  # Basic Accessors.
  def load_speaker(self, idx) -> types.AgentProperties:
    return types.AgentProperties(
        params=self._params['speaker'][idx],
        target_params=self._params['target_speaker'][idx],
        states=self._states['speaker'][idx],
        target_states=self._states['target_speaker'][idx],
        opt_states=self._opt_states['speaker'][idx],
    )

  def load_listener(self, idx) -> types.AgentProperties:
    return types.AgentProperties(
        params=self._params['listener'][idx],
        states=self._states['listener'][idx],
        opt_states=self._opt_states['listener'][idx],
    )

  def store_agent(
      self,
      agent_id: int,
      agent_name: str,
      params: hk.Params,
      states: hk.State,
      opt_states: Optional[optax.OptState] = None,
  ) -> None:
    """Once data for an agent is updated, store it back."""
    assert agent_name in ['speaker', 'listener', 'target_speaker']
    assert 0 <= agent_id < len(self._params[agent_name])

    self._params[agent_name][agent_id] = params
    self._states[agent_name][agent_id] = states
    if opt_states:
      self._opt_states[agent_name][agent_id] = opt_states

  # Checkpointing utilities.
  def snapshot(self) -> types.AllProperties:
    return types.AllProperties(
        params=self._params,
        states=self._states,
        opt_states=self._opt_states,
    )

  def restore(
      self,
      params: Optional[types.AllParams] = None,
      states: Optional[types.AllStates] = None,
      opt_states: Optional[types.AllOptStates] = None,
  ) -> None:
    """Restores all params/states of the agent/optimizer."""
    if params:
      assert all([
          k in ['speaker', 'listener', 'target_speaker'] for k in params.keys()
      ])
      self._params.update(params)

    if states:
      assert all([
          k in ['speaker', 'listener', 'target_speaker'] for k in states.keys()
      ])
      self._states.update(states)

    if opt_states:
      assert all([k in ['speaker', 'listener'] for k in opt_states.keys()])
      self._opt_states.update(opt_states)

  def initialize(
      self,
      rng_key: chex.PRNGKey,
      games: types.GamesInputs,
      game_init_fn,
      opt_speaker_init_fn,
      opt_listener_init_fn,
  ) -> None:
    """Initializes all the params/states of the agent/optimizer."""

    # Initializes params/states.
    self._params = dict(speaker=[], listener=[], target_speaker=[])
    self._states = dict(speaker=[], listener=[], target_speaker=[])
    self._opt_states = dict(speaker=[], listener=[])

    # Iterates over speaker/listener options.
    for agent_name, num_agents, opt_init_fn in zip(
        ('speaker', 'listener'), (self._n_speakers, self._n_listeners),
        (opt_speaker_init_fn, opt_listener_init_fn)):

      # Prepares per agents pmap function.
      params_init_pmap = jax.pmap(
          fn.partial(
              game_init_fn,
              training_mode=types.TrainingMode.TRAINING,
          ))
      opt_init_pmap = jax.pmap(opt_init_fn)

      for _ in range(num_agents):

        # Prepares rng.
        rng_key, rng = jax.random.split(rng_key)
        rng = utils.bcast_local_devices(rng)  # same network init across devices

        # Init Params/States.
        joint_params, joint_states = params_init_pmap(init_games=games, rng=rng)
        agent_params = getattr(joint_params, agent_name)
        agent_states = getattr(joint_states, agent_name)
        self._params[agent_name].append(agent_params)
        self._states[agent_name].append(agent_states)
        if agent_name == 'speaker':
          self._params['target_speaker'].append(joint_params.target_speaker)
          self._states['target_speaker'].append(joint_states.target_speaker)

        # Init Opt state.
        agent_opt_states = opt_init_pmap(agent_params)
        self._opt_states[agent_name].append(agent_opt_states)

  def load_pair(
      self, speaker_id: int,
      listener_id: int) -> Tuple[types.Params, types.States, types.OptStates]:
    """Prepares params and opt_states for a given pair of speaker/listener."""
    assert 0 <= speaker_id < len(self._params['speaker'])
    assert 0 <= listener_id < len(self._params['listener'])

    params = types.Params(
        speaker=self._params['speaker'][speaker_id],
        listener=self._params['listener'][listener_id],
        target_speaker=self._params['target_speaker'][speaker_id],
    )
    states = types.States(
        speaker=self._states['speaker'][speaker_id],
        listener=self._states['listener'][listener_id],
        target_speaker=self._states['target_speaker'][speaker_id],
    )
    opt_states = types.OptStates(
        speaker=self._opt_states['speaker'][speaker_id],
        listener=self._opt_states['listener'][listener_id])

    return params, states, opt_states

  def store_pair(
      self,
      speaker_id: int,
      listener_id: int,
      params: types.Params,
      states: types.States,
      opt_states: types.OptStates,
  ) -> None:
    """Once data for a pair speaker/listener is updated, store it back."""
    self.store_agent(
        agent_id=speaker_id,
        agent_name='speaker',
        params=params.speaker,
        states=states.speaker,
        opt_states=opt_states.speaker,
    )
    self.store_agent(
        agent_id=speaker_id,
        agent_name='target_speaker',
        params=params.target_speaker,
        states=states.target_speaker,
    )
    self.store_agent(
        agent_id=listener_id,
        agent_name='listener',
        params=params.listener,
        states=states.listener,
        opt_states=opt_states.listener,
    )
