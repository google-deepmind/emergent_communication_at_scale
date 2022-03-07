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

"""Implement another experiment to compute ease of learning of agents."""

from typing import List, Optional

from absl import logging

import chex
import jax
from jaxline import utils
from ml_collections import config_dict
import numpy as np
import optax
from emergent_communication_at_scale import lewis_experiment
from emergent_communication_at_scale import types
from emergent_communication_at_scale.utils import checkpointer as ckpt_lib


class EaseOfLearningExperiment(lewis_experiment.LewisExperiment):
  """Ease of learning experiment.

  The ease of learning is defined as how fast a new listener acquires
  an emergent language (the speaker is fixed).
  """

  def __init__(
      self,
      mode: str,
      init_rng: chex.PRNGKey,
      config: config_dict.ConfigDict,
  ) -> None:
    """Initializes experiment."""
    if mode != 'train':
      raise ValueError(
          f'Ease of learning experiment only supports train mode: {mode}.')

    # Step 1: Loads ckpt and the related config.
    # Retrieve speaker params and config fo perform Ease of learning from a
    # given lewis configuration path
    path = config.speaker_path.path
    if not path:
      raise ValueError(f'{path} does not exist to retrieve checkpoint.')
    exp_state, lewis_cfg, _, _ = ckpt_lib.load_checkpoint(path)  # pytype: disable=attribute-error

    # Complete the eol configuration with lewis config option.
    config = config.unlock()
    config['speaker'] = lewis_cfg.speaker

    # Add dummy values that are required to start LewisExperiment
    config.training['target_update_ema'] = lewis_cfg.training.target_update_ema
    config.evaluation['batch_size'] = lewis_cfg.evaluation.batch_size
    config = config.lock()

    # Step 2: Creates the lewis experiment to perform ease of learning.
    super().__init__(mode=mode, init_rng=init_rng, config=config)

    # Overrides the speaker params with loaded ckpt.
    ckpt_params, ckpt_states = exp_state.params, exp_state.states
    speaker_params = ckpt_params['speaker'][config.speaker_path.speaker_index]
    speaker_states = ckpt_states['speaker'][config.speaker_path.speaker_index]
    self._population_storage.restore(
        params=dict(speaker=[speaker_params]),
        states=dict(speaker=[speaker_states]))

  def speaker_optimizer(self) -> optax.GradientTransformation:
    """Speaker params must be fixed => set the learning rate to zero."""
    return optax.sgd(learning_rate=0.0)

  def train_loop(
      self,
      config: config_dict.ConfigDict,
      state,
      periodic_actions: List[utils.PeriodicAction],
      writer: Optional[utils.Writer] = None,
  ) -> None:
    """Overrides `train_loop` to collect the 'accuracy' output scalar values."""

    class CollectAccuracies:
      """A helper that collects 'accuracy' output scalar values."""

      def __init__(self) -> None:
        self.collector_accuracies = []

      def update_time(self, t: float, step: int) -> None:
        del t, step  # Unused.

      def __call__(
          self,
          t: float,
          step: int,
          scalar_outputs: types.Config,
      ) -> None:
        del t, step  # Unused.
        self.collector_accuracies.append(scalar_outputs['global_accuracy'])

    collector = CollectAccuracies()
    # Weirdly type(periodic_actions) is tuple and not list!
    super().train_loop(
        config=config,
        state=state,
        periodic_actions=list(periodic_actions) + [collector],
        writer=writer,
    )

    # Fetches from device and stack the accuracy numbers.
    accuracies = np.array(jax.device_get(collector.collector_accuracies))
    logging.info('Ease of learning accuracies per listener %s', accuracies)
