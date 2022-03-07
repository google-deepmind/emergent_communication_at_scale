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

"""Class for checkpointing that fits jaxline pipeline."""

import time
from typing import List, Mapping, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import utils
from ml_collections import config_dict
from emergent_communication_at_scale.utils import checkpointer as ckpt_api
from emergent_communication_at_scale.utils import population_storage as ps


class ExperimentWithCheckpointing(experiment.AbstractExperiment):
  """Helper to save/load ckpt during traning."""

  def __init__(self,
               mode: str,
               init_rng: chex.PRNGKey,
               config: config_dict.ConfigDict) -> None:
    super().__init__(mode=mode, init_rng=init_rng)
    self._checkpointer = ckpt_api.Checkpointer(**config.checkpointing)
    self._training_steps = config.training.steps

    self._population_storage = None  # will be inited in LewisExperiment.
    self._config = None              # will be inited in LewisExperiment.

  @property
  def checkpoint_path(self) -> str:
    return self._checkpointer.checkpoint_path

  def snapshot_state(self) -> Mapping[str, chex.Array]:
    """Takes a frozen copy of the current experiment state for checkpointing."""
    jaxline_snapshot = super().snapshot_state()
    emcom_snaptshot = self._population_storage.snapshot()
    return {
        **jaxline_snapshot,
        'params': utils.get_first(emcom_snaptshot.params),
        'states': utils.get_first(emcom_snaptshot.states),
        'opt_states': utils.get_first(emcom_snaptshot.opt_states),
    }

  def restore_from_snapshot(self,
                            snapshot_state: Mapping[str, chex.Array]) -> None:
    """Restores experiment state from a snapshot."""
    super().restore_from_snapshot(snapshot_state)
    self._population_storage.restore(
        params=utils.bcast_local_devices(snapshot_state['params']),
        states=utils.bcast_local_devices(snapshot_state['states']),
        opt_states=utils.bcast_local_devices(snapshot_state['opt_states']))

  def save_checkpoint(self, step: int, rng: jnp.ndarray) -> None:
    self._checkpointer.maybe_save_checkpoint(
        self._population_storage.snapshot(),
        config=self._config,
        step=step,
        rng=rng,
        is_final=step >= self._training_steps)

  def restore_state(self, restore_path: Optional[str] = None,
                    ) -> Tuple[int, ps.PopulationStorage]:
    """Initializes experiment state from a checkpoint."""

    if restore_path is None:
      restore_path = self._checkpointer.checkpoint_path

    # Load pretrained experiment state.
    exp_state, _, step, _ = ckpt_api.load_checkpoint(restore_path)  # pytype: disable=attribute-error
    self._population_storage.restore(params=exp_state.params,
                                     states=exp_state.states,
                                     opt_states=exp_state.opt_states)
    return step, self._population_storage

  def train_loop(self,
                 config: config_dict.ConfigDict,
                 state,
                 periodic_actions: List[utils.PeriodicAction],
                 writer: Optional[utils.Writer] = None)  -> None:
    """Overrides the jaxline train_loop to add regular checkpointing."""

    is_chief = jax.process_index() == 0
    step = state.global_step
    rng = state.train_step_rng
    checkpoint_config = config.experiment_kwargs.config.checkpointing

    if config.train_checkpoint_all_hosts or is_chief:
      if checkpoint_config.save_checkpoint_interval > 0:
        periodic_actions += (
            utils.PeriodicAction(
                lambda x, *_: self.save_checkpoint(step=x, rng=rng),
                interval_type=(config.checkpoint_interval_type
                               or config.interval_type),
                interval=checkpoint_config.save_checkpoint_interval,
                run_async=False),)  # run_async True would not be thread-safe.

    for pa in periodic_actions:
      pa.update_time(time.time(), step)

    super().train_loop(config, state, periodic_actions, writer)
