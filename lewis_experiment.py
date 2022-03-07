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

"""Emergent Communication jaxline experiment."""

from typing import List, Tuple

from absl import flags
from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp

from jaxline import utils
from ml_collections import config_dict
import numpy as np
import optax
from emergent_communication_at_scale import agent
from emergent_communication_at_scale import types
from emergent_communication_at_scale.game import game_factory
from emergent_communication_at_scale.trainers import communication_trainer
from emergent_communication_at_scale.trainers import imitation_trainer
from emergent_communication_at_scale.trainers import reset_trainer
from emergent_communication_at_scale.utils import experiment_with_checkpointing as jaxline_ckpt
from emergent_communication_at_scale.utils import language_measures
from emergent_communication_at_scale.utils import population_storage as ps
from emergent_communication_at_scale.utils import utils as emcom_utils

# This file should only include langame and jaxline dependencies!

FLAGS = flags.FLAGS


# A split helper that operates on a pmap-ed rng_key.
@jax.pmap
def _split_three_keys_pmap(key):
  return tuple(jax.random.split(key, num=3))


class LewisExperiment(jaxline_ckpt.ExperimentWithCheckpointing):
  """Cidre experiment.

  Note: we here inherit from ExperimentWithCheckpointing to abstract the ckpt
  mechanism that is entangled in jaxline.
  Beware that  ExperimentWithCheckpointing inherits from
  experiment.AbstractExperiment in jaxline.
  """

  def __init__(
      self,
      mode: str,
      init_rng: chex.PRNGKey,
      config: config_dict.ConfigDict,
  ) -> None:
    """Initializes experiment."""
    super().__init__(mode=mode, init_rng=init_rng, config=config)

    self._mode = mode
    self._init_rng = init_rng
    #  ConfigDict are not completely resolved when a reference is given inside
    #  a structure such as a list or a tuple.
    self._config = emcom_utils.resolve_dictionary(config)

    # By default, we do not use a population
    pop_config = self._config.population
    self._n_speakers = pop_config.get('n_speakers', 1)
    self._n_listeners = pop_config.get('n_listeners', 1)
    self._num_agents_per_step = pop_config.get('num_agents_per_step', 1)

    # Prepares games.
    self._game = agent.SpeakerListenerGame(config=self._config)
    self._game_builder = game_factory.get(
        config=self._config.game,
        train_batch_size=self._config.training.batch_size,
        eval_batch_size=self._config.evaluation.batch_size,
    )

    # Prepares parameters.
    self._population_storage = ps.PopulationStorage(
        n_speakers=self._n_speakers,
        n_listeners=self._n_listeners,
    )

    # Train vs. Eval.
    if self._mode == 'train':
      # Constructs the input dataset that will be prefetched in background.
      self._train_input = utils.py_prefetch(
          lambda: self._game_builder.get_training_games(self._init_rng),
          buffer_size=10,
      )

      ### Lewis trainer
      # Constructs the trainer that sample and update agents pairs.
      self._communication_trainer = communication_trainer.BasicTrainer(
          update_fn=self._update_fn,
          n_speakers=self._n_speakers,
          n_listeners=self._n_listeners,
          num_agents_per_step=self._num_agents_per_step)

      ### Imitation trainer.
      if self._config.imitation and self._config.imitation.imitation_step:

        # Checks config values.
        if not self._config.imitation.self_imitation and self._n_speakers < 2:
          raise ValueError('Invalid imitation config: n_speaker must be larger.'
                           ' than one.')
        if self._config.imitation.self_imitation and self._n_speakers != 1:
          raise ValueError('Invalid imitation config: n_speaker must be equal'
                           ' to one for self-imitation.')

        # Cases where we perform imitation training.
        logging.info('Training option: apply imitation.')
        self._imitation_trainer = imitation_trainer.ImitateTrainer(
            n_speakers=self._n_speakers,
            imitation_update_fn=self._imitation_update_fn,
        )
      else:
        # Cases where we do not perform imitation training.
        logging.info('Training option: Do not apply imitation.')
        self._imitation_trainer = None

      ### Resets trainer.
      if config.reset and self._config.reset.reset_step:
        logging.info('Training option: apply resetting.')
        self._reset_trainer = reset_trainer.ResetTrainer(
            n_speakers=self._n_speakers,
            n_listeners=self._n_listeners,
        )
      else:
        # Cases where we do not perform resetting.
        logging.info('Training option: Do not apply resetting.')
        self._reset_trainer = None

      # Initializes network/optim param/states.
      games = next(self._game_builder.get_training_games(init_rng))
      self._population_storage.initialize(
          rng_key=init_rng,
          games=games,
          game_init_fn=self._game.init,
          opt_speaker_init_fn=self.speaker_optimizer().init,
          opt_listener_init_fn=self.listener_optimizer().init,
      )
    else:
      self._eval_batch = jax.jit(self._eval_batch)
      self._communication_trainer = None
      self._imitation_trainer = None
      self._reset_trainer = None

  def listener_optimizer(self) -> optax.GradientTransformation:
    return self.create_optimizer(self._config.listener_optimizer)

  def speaker_optimizer(self) -> optax.GradientTransformation:
    return self.create_optimizer(self._config.speaker_optimizer)

  def create_optimizer(
      self,
      config: config_dict.ConfigDict,
  ) -> optax.GradientTransformation:
    name = config.name
    kwargs = config.kwargs.get(name, dict())
    optimizer = getattr(optax, name)
    return optimizer(config.learning_rate, **kwargs)

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(
      self,
      global_step: chex.ArrayNumpy,
      rng: chex.PRNGKey,
      **unused_args,
  ) -> types.Config:
    """A single training step."""
    games = next(self._train_input)

    rng_communicate, rng_imitate, rng_reset = _split_three_keys_pmap(rng)

    # Performs one step of population training.
    # Population trainer sample agents pair before `_update_func` per pair.
    scalars, self._population_storage = self._communication_trainer.communicate(
        rng=rng_communicate,
        games=games,
        agent_storage=self._population_storage,
    )

    global_step = utils.get_first(global_step)

    # Imitation learning every imitation_step steps.
    if (self._imitation_trainer and global_step > 0 and
        global_step % self._config.imitation.imitation_step == 0):

      imit_scalar, self._population_storage = self._imitation_trainer.imitate(
          rng=rng_imitate,
          games=games,
          agent_storage=self._population_storage,
          **self._config.imitation,
      )
      scalars.update(imit_scalar)

    # Reset step.
    if (self._reset_trainer and global_step > 0 and
        global_step % self._config.reset.reset_step == 0):

      self._population_storage = self._reset_trainer.reset(
          rng=rng_reset,
          games=games,
          agent_storage=self._population_storage,
          game_init_fn=self._game.init,
          opt_speaker_init_fn=self.speaker_optimizer().init,
          opt_listener_init_fn=self.listener_optimizer().init,
          reset_type=self._config.reset.reset_type,
      )

    # Returns the scalar of the last random pair.
    return scalars

  def _update_fn(
      self,
      params: types.Params,
      states: types.States,
      opt_states: types.OptStates,
      games: types.GamesInputs,
      rng: chex.PRNGKey,
      training_mode: types.TrainingMode,
      is_sharded_update: bool = True,
  ) -> Tuple[types.Params, types.States, types.OptStates, types.Config]:
    """Applies an update to parameters and returns new state.

    Args:
      params: The current (speaker, listener) params to update.
      states: The current (speaker, listener) states.
      opt_states: The current optimizer state for speaker and listener.
      games: The input batch of games to learn on.
      rng: The random key.
      training_mode: defines the training_mode (TRAIN=sampling, EVAL=greedy).
      is_sharded_update: If set, the code assumes it's running within the
        context of a pmap, and thus would use jax.lax.pxxx functions to average
        gradients or measurementes across chips/shards.

    Returns:
      new_params: The updated params.
      new_states: The updated state.
      new_opt_states: The updated optimizer state.
      scalars: A dict of scalar measurements to log.
    """
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    grads, agent_loss_stats = grad_loss_fn(
        params,
        states=states,
        games=games,
        rng=rng,
        training_mode=training_mode,
    )
    if is_sharded_update:
      # grad_loss_fn outputs the grads divided by the number of devices
      # (jax.device_count()). We apply the psum to get the mean across devices.
      grads = jax.lax.psum(grads, axis_name='i')

    # Computes and applies updates via our optimizer.
    _, speaker_opt_update = self.speaker_optimizer()
    _, listener_opt_update = self.listener_optimizer()

    speaker_updates, new_opt_state_speaker = speaker_opt_update(
        grads.speaker, opt_states.speaker)
    new_params_speaker = optax.apply_updates(params.speaker, speaker_updates)

    listener_updates, new_opt_state_listener = listener_opt_update(
        grads.listener, opt_states.listener)
    new_params_listener = optax.apply_updates(params.listener, listener_updates)
    new_target_params = emcom_utils.update_target_params(
        rl_params=new_params_speaker,
        target_rl_params=params.target_speaker,
        target_network_update_ema=self._config.training.target_update_ema,
    )
    new_params = types.Params(
        speaker=new_params_speaker,
        listener=new_params_listener,
        target_speaker=new_target_params)

    new_opt_states = types.OptStates(
        speaker=new_opt_state_speaker, listener=new_opt_state_listener)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = jax.tree_map(lambda x: x / games.speaker_inp.shape[0],
                           agent_loss_stats)

    if is_sharded_update:
      scalars = jax.lax.pmean(scalars, axis_name='i')
    ### Stores the score of the individual speakers inside the state
    # Retrieves speaker states.
    speaker_state = states.speaker
    counter = speaker_state['speaker']['counter']
    avg_score = speaker_state['speaker']['avg_score']

    # Updates speaker by computing the average score.
    mutable_state = hk.data_structures.to_mutable_dict(speaker_state)
    mutable_state['speaker']['avg_score'] = (counter * avg_score) / (
        counter + 1) + scalars['global_accuracy'] / (
            counter + 1)
    mutable_state['speaker']['counter'] += 1
    speaker_state = hk.data_structures.to_haiku_dict(mutable_state)

    # Updates states across devices.
    speaker_state = jax.lax.pmean(speaker_state, axis_name='i')
    new_states = states._replace(speaker=speaker_state)

    return new_params, new_states, new_opt_states, scalars

  def _loss_fn(
      self,
      params: types.Params,
      states: types.States,
      games: types.GamesInputs,
      rng: chex.PRNGKey,
      training_mode: types.TrainingMode,
  ):
    rng_unroll, rng_loss = jax.random.split(rng)
    games = self._game.unroll(
        params,
        states,
        rng=rng_unroll,
        games=games,
        training_mode=training_mode,
    )
    agent_loss_outputs = self._game.compute_loss(games=games, rng=rng_loss)

    avg_fn = lambda x: x / games.speaker_inp.shape[0]

    scaled_loss = avg_fn(agent_loss_outputs.loss) / jax.device_count()

    return scaled_loss, agent_loss_outputs.stats

  def _imitation_update_fn(
      self,
      games: types.GamesInputs,
      params_student: hk.Params,
      params_oracle: hk.Params,
      state_student: hk.State,
      state_oracle: hk.State,
      opt_state: optax.OptState,
      rng: chex.PRNGKey,
  ):
    # Gets labels (output of the oracle).
    games = types.Games(speaker_inp=games.speaker_inp, labels=games.labels)
    # rng not used as training_mode=EVAL.
    oracle_outputs, _ = self._game.speaker.apply(
        params_oracle,
        state_oracle,
        rng,
        games=games,
        training_mode=types.TrainingMode.EVAL,
    )
    # Computes gradient.
    grad_supervised_loss_fn = jax.grad(self._supervised_loss_fn, has_aux=True)
    scaled_grads, loss = grad_supervised_loss_fn(
        params_student,
        state_student,
        labels=jax.lax.stop_gradient(oracle_outputs.action),
        games=games,
        rng=rng)
    grads = jax.lax.psum(scaled_grads, axis_name='i')

    # Computes and applies updates via our optimizer.
    _, speaker_opt_update = self.speaker_optimizer()
    speaker_updates, new_opt_state_speaker = speaker_opt_update(
        grads, opt_state)
    new_params_speaker = optax.apply_updates(params_student, speaker_updates)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = loss / games.speaker_inp.shape[0]
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return new_params_speaker, state_student, new_opt_state_speaker, scalars

  def _supervised_loss_fn(
      self,
      params_student: hk.Params,
      state_student: hk.Params,
      labels: chex.Array,
      games: types.GamesInputs,
      rng: chex.PRNGKey,
  ):
    prediction_outputs, _ = self._game.speaker.apply(
        params_student,
        state_student,
        rng,
        games=games,
        training_mode=types.TrainingMode.TRAINING,
    )
    logits = jnp.transpose(prediction_outputs.policy_logits, [0, 2, 1])
    labels = jax.nn.one_hot(
        labels, self._config.speaker.vocab_size, dtype=logits.dtype)
    # [B, T]
    loss = emcom_utils.softmax_cross_entropy(logits, labels)
    # Average on T and sum on B
    loss = jnp.sum(jnp.mean(loss, axis=-1), axis=0)

    avg_fn = lambda x: x / logits.shape[0]
    scaled_loss = avg_fn(loss) / jax.device_count()

    return scaled_loss, loss

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(
      self,
      global_step: chex.ArrayNumpy,
      rng: chex.PRNGKey,
      **unused_kwargs,
  ) -> types.Config:
    """See base class."""

    # Gives a mode equal to either test or valid.
    # Gives a ensemble_type in [vote, average].
    _, mode, ensemble_type = self._mode.split('_')

    # Computes metrics over the evaluation games.
    game_scalars, messages = self._eval_over_games(mode, ensemble_type, rng)

    # Computes metrics by iterating over concepts.
    # It is only computed over speaker message, independently of listener.
    message_scalars = self._eval_over_messages(messages)

    # Fuses and formats scalars.
    scalars = {**game_scalars, **message_scalars}

    scalars = jax.device_get(scalars)

    logging.info('Eval [Step %d] %s', global_step, scalars)

    return scalars

  def _eval_over_games(
      self,
      mode: str,
      ensemble_type: str,
      rng: chex.PRNGKey,
  ) -> Tuple[types.Config, List[chex.Array]]:

    # Eval at most the self._config.evaluation.max_n_agents first agents.
    n_speakers = np.min(
        [self._n_speakers, self._config.evaluation.max_n_agents])
    n_listeners = np.min(
        [self._n_listeners, self._config.evaluation.max_n_agents])

    # Initializes values.
    num_games, sum_scalars = 0, None
    topographic_similarity = []
    messages = [[] for _ in range(n_speakers)]

    # Prepares subsampling.
    subsampling_ratio = self._config.evaluation.subsampling_ratio
    assert 0.01 <= subsampling_ratio <= 1

    for samples in self._game_builder.get_evaluation_games(mode):
      for speaker_id in range(n_speakers):
        all_agents_outputs = []
        for listener_id in range(n_listeners):

          # Retrieves params.
          params, states, _ = self._population_storage.load_pair(
              speaker_id=speaker_id, listener_id=listener_id)
          params = utils.get_first(params)  # eval is on single device only
          states = utils.get_first(states)  # eval is on single device only

          # Play game.
          # rng is not used at eval time.
          agent_outputs, games = self._eval_batch(
              params=params, states=states, games=samples, rng=rng)
          all_agents_outputs.append(agent_outputs)

        # Computes scalar by averaging all listeners.
        ensemble_scalars = self._eval_all_listeners(
            ensemble_type=ensemble_type,
            predictions=all_agents_outputs,
            games=games,
        )
        # Saves ensemble stats and stats for the last listener (one pair).
        scalars = {**ensemble_scalars, **agent_outputs.stats}

        # Updates counters.
        num_games += games.speaker_inp.shape[0]

        # Accumulates the sum of scalars for each step.
        if sum_scalars is None:
          sum_scalars = scalars
        else:
          sum_scalars = jax.tree_multimap(jnp.add, scalars, sum_scalars)

        # Computes message statistics. As it is independent of the listener,
        # we here arbitrary take the last listener.
        slices = max(3, int(games.speaker_inp.shape[0] * subsampling_ratio))
        # Takes only the first slices examples.
        slice_games = jax.tree_map(lambda x, y=slices: x[:y], games)
        topographic_similarity += [
            language_measures.games_topographic_similarity(
                games=slice_games,
                meaning_sim=self._config.evaluation.topsim_meaning_similarity,
                task=self._config.evaluation.topsim_task,
                )
        ]

        # Stores message for end-game analysis.
        messages[speaker_id].append(games.speaker_outputs.action)

    # Averages per number of total games (both wrt batches and populations).
    avg_scalars = jax.tree_map(lambda x: x / num_games, sum_scalars)
    avg_scalars['topographic_similarity'] = np.mean(topographic_similarity)

    # stacks messages into a single batch.
    messages = [np.concatenate(m, axis=0) for m in messages]

    return avg_scalars, messages

  def _eval_all_listeners(
      self,
      ensemble_type: str,
      predictions: List[types.AgentLossOutputs],
      games: types.Games,
  ):

    if ensemble_type == 'vote':
      probs = [x.probs for x in predictions]
      # Stacks leaves of probs, which can be a list of dictionaries for classif.
      stacked_pred = jax.tree_multimap(lambda *vals: np.stack(vals, axis=-1),
                                       *probs)  # [B, F, listeners]
      avg_prediction = jax.tree_map(lambda x: jnp.mean(x, axis=-1),
                                    stacked_pred)  # [B, F]
      ensemble_pred = jax.tree_map(lambda x: jnp.argmax(x, axis=-1),
                                   avg_prediction)  # [B]
      scalars = self._game.listener_loss.compute_ensemble_accuracy(
          prediction=ensemble_pred, games=games)
    elif ensemble_type == 'average':
      accuracies = jnp.array([x.stats['global_accuracy'] for x in predictions])
      scalars = dict(ensemble_acc=jnp.mean(accuracies))
    else:
      raise ValueError(f'Wrong ensemble type: {ensemble_type}.')

    return scalars

  def _eval_batch(
      self,
      params: types.Params,
      states: types.States,
      games: types.GamesInputs,
      rng: chex.PRNGKey,
  ) -> Tuple[types.AgentLossOutputs, types.Games]:
    finished_game = self._game.unroll(
        params,
        states,
        rng=rng,
        games=games,
        training_mode=types.TrainingMode.EVAL,
    )
    agent_loss_outputs = self._game.compute_loss(games=finished_game,
                                                 rng=jax.random.PRNGKey(42))

    return agent_loss_outputs, finished_game

  def _eval_over_messages(self, messages: List[chex.Array]) -> types.Config:

    # Computes edit distance between messages from different speakers.
    edit_distance = []
    message_per_games = np.stack(messages, axis=1)  # [n_games, n_speaker, T]
    for message in message_per_games:
      # These messages are from the same game, and thus encode the same concept.
      edit_dist = language_measures.edit_dist(message)
      edit_dist = np.mean(edit_dist)
      edit_distance.append(edit_dist)

    return dict(edit_distance=np.mean(edit_distance))
