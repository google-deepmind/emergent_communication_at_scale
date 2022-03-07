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

"""The actual logic of a agent."""

import functools
from typing import Tuple

import chex
import haiku as hk
import jax
from ml_collections import config_dict
from emergent_communication_at_scale import types
from emergent_communication_at_scale.losses import listeners as listener_losses
from emergent_communication_at_scale.losses import speakers as speaker_losses
from emergent_communication_at_scale.networks import listeners
from emergent_communication_at_scale.networks import speakers


class SpeakerListenerGame:
  """Plays a speaker/listener game with multi symbol."""

  def __init__(self, config: config_dict.ConfigDict) -> None:

    # Prepares constructor.
    speaker = functools.partial(speakers.Speaker, **config.speaker)
    listener = functools.partial(listeners.Listener, **config.listener)

    # hk.transform requires lambda to be built a posteriori in a pmap
    # pylint: disable=unnecessary-lambda
    # pylint: disable=g-long-lambda
    self._speaker = hk.transform_with_state(
        lambda games, training_mode, actions_to_follow=None: speaker()(
            games=games,
            training_mode=training_mode,
            actions_to_follow=actions_to_follow,
        ))
    self._listener = hk.transform_with_state(
        lambda games, training_mode: listener()(
            games=games,
            training_mode=training_mode,
        ))
    # pylint: enable=unnecessary-lambda
    # pylint: enable=g-long-lambda

    if config.loss.get('speaker', False):
      self._speaker_loss = speaker_losses.speaker_loss_factory(
          **config.loss.speaker)
    else:
      # We do not have speaker loss in EOL
      self._speaker_loss = None

    self._listener_loss = listener_losses.listener_loss_factory(
        **config.loss.listener)
    self._config = config

  @property
  def speaker(self):
    return self._speaker

  @property
  def listener_loss(self):
    return self._listener_loss

  def init(
      self,
      rng: chex.PRNGKey,
      init_games: types.GamesInputs,
      training_mode: types.TrainingMode,
  ) -> Tuple[types.Params, types.States]:
    """Returns speaker and listener params."""
    games = types.Games(
        speaker_inp=init_games.speaker_inp, labels=init_games.labels)

    speaker_rng, target_speaker_rng, listener_rng = jax.random.split(rng, 3)
    params_speaker, states_speaker = self._speaker.init(
        speaker_rng,
        games=games,
        training_mode=training_mode,
    )
    speaker_outputs, _ = self._speaker.apply(
        params_speaker,
        states_speaker,
        speaker_rng,
        games=games,
        training_mode=training_mode,
    )
    target_params_speaker, target_states_speaker = self._speaker.init(
        target_speaker_rng,
        games=games,
        training_mode=types.TrainingMode.FORCING,
        actions_to_follow=speaker_outputs.action,
    )
    games = games._replace(speaker_outputs=speaker_outputs)
    params_listener, state_listener = self._listener.init(
        listener_rng,
        games=games,
        training_mode=training_mode,
    )
    joint_states = types.States(
        speaker=states_speaker,
        listener=state_listener,
        target_speaker=target_states_speaker)
    joint_params = types.Params(
        speaker=params_speaker,
        listener=params_listener,
        target_speaker=target_params_speaker,
    )
    return joint_params, joint_states

  def unroll(
      self,
      params: types.Params,
      states: types.States,
      rng: chex.PRNGKey,
      games: types.GamesInputs,
      training_mode: types.TrainingMode,
  ) -> types.Games:
    """Unrolls the game for the forward pass."""

    # Prepares output.
    speaker_rng, listener_rng, rng = jax.random.split(rng, 3)
    games = types.Games(speaker_inp=games.speaker_inp, labels=games.labels)

    # Step1 : Speaker play.
    speaker_outputs, _ = self._speaker.apply(
        params.speaker,
        states.speaker,
        speaker_rng,
        games=games,
        training_mode=training_mode,
    )

    target_speaker_outputs, _ = self._speaker.apply(
        params.target_speaker,
        states.target_speaker,
        speaker_rng,
        games=games,
        training_mode=types.TrainingMode.FORCING,
        actions_to_follow=speaker_outputs.action,
    )
    games = games._replace(
        speaker_outputs=speaker_outputs,
        target_speaker_outputs=target_speaker_outputs,
    )

    # Step 2 : Listener play.
    listener_outputs, _ = self._listener.apply(
        params.listener,
        states.listener,
        listener_rng,
        games=games,
        training_mode=training_mode,
    )
    games = games._replace(listener_outputs=listener_outputs)
    return games

  def compute_loss(
      self,
      games: types.Games,
      rng: chex.PRNGKey,
  ) -> types.AgentLossOutputs:
    """Computes Listener and Speaker losses."""

    # Computes listener loss and stats.
    listener_loss_outputs = self._listener_loss.compute_listener_loss(
        games=games,
        rng=rng,
    )
    loss = listener_loss_outputs.loss
    stats = listener_loss_outputs.stats

    # Computes speaker loss and stats. (if necessary).
    if self._speaker_loss is not None:
      speaker_loss_outputs = self._speaker_loss.compute_speaker_loss(
          games=games,
          reward=listener_loss_outputs.reward,
      )
      loss += speaker_loss_outputs.loss
      stats.update(speaker_loss_outputs.stats)

    return types.AgentLossOutputs(
        loss=loss,
        probs=listener_loss_outputs.probs,
        stats=stats,
    )
