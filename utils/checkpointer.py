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

"""Generic checkpointer to load and store data."""
import collections
import os
import pickle
import time
from typing import Optional, Text

from absl import logging
import jax
import jax.numpy as jnp
from jaxline import utils
from ml_collections import config_dict
from emergent_communication_at_scale import types

CkptData = collections.namedtuple("CkptData",
                                  ["experiment_state", "config", "step", "rng"])


def load_checkpoint(checkpoint_path: Text) -> Optional[CkptData]:
  """Loads a checkpoint if any is found."""

  # Step 1: Load file
  try:
    with open(checkpoint_path, "rb") as checkpoint_file:
      checkpoint_data = pickle.load(checkpoint_file)
      logging.info("Loading checkpoint from %s, saved at step %d",
                   checkpoint_path, checkpoint_data["step"])

  except FileNotFoundError:
    logging.info("No existing checkpoint found at %s", checkpoint_path)
    return None

  # Retrieve experiment states (params, states etc.)
  experiment_state = checkpoint_data["experiment_state"]
  experiment_state = jax.tree_map(utils.bcast_local_devices,
                                  experiment_state)
  return CkptData(
      experiment_state=experiment_state,
      config=checkpoint_data["config"],
      step=checkpoint_data["step"],
      rng=checkpoint_data["rng"])


class Checkpointer:
  """A checkpoint saving and loading class."""

  def __init__(
      self,
      use_checkpointing: bool,
      checkpoint_dir: Text,
      save_checkpoint_interval: int,
      filename: Text):
    if (not use_checkpointing or
        checkpoint_dir is None or
        save_checkpoint_interval <= 0):
      self._checkpoint_enabled = False
      return

    self._checkpoint_enabled = True
    self._checkpoint_dir = checkpoint_dir
    os.makedirs(self._checkpoint_dir, exist_ok=True)
    self._filename = filename
    self._checkpoint_path = os.path.join(self._checkpoint_dir, filename)
    self._last_checkpoint_time = 0
    self._checkpoint_every = save_checkpoint_interval

  @property
  def checkpoint_path(self) -> str:
    return self._checkpoint_path

  def maybe_save_checkpoint(
      self,
      xp_state: types.AllProperties,
      config: config_dict.ConfigDict,
      step: int,
      rng: jnp.ndarray,
      is_final: bool):
    """Saves a checkpoint if enough time has passed since the previous one."""
    current_time = time.time()

    # Checks whether we should perform checkpointing.
    if (not self._checkpoint_enabled or
        jax.host_id() != 0 or  # Only checkpoint the first worker.
        (not is_final and
         current_time - self._last_checkpoint_time < self._checkpoint_every)):
      return

    # Creates data to checkpoint.
    checkpoint_data = dict(
        experiment_state=jax.tree_map(lambda x: jax.device_get(x[0]), xp_state),
        config=config,
        step=step,
        rng=rng)

    # Creates a rolling ckpt.
    with open(self._checkpoint_path + "_tmp", "wb") as checkpoint_file:
      pickle.dump(checkpoint_data, checkpoint_file, protocol=2)
    try:
      os.rename(self._checkpoint_path, self._checkpoint_path + "_old")
      remove_old = True
    except FileNotFoundError:
      remove_old = False  # No previous checkpoint to remove
    os.rename(self._checkpoint_path + "_tmp", self._checkpoint_path)
    if remove_old:
      os.remove(self._checkpoint_path + "_old")
    self._last_checkpoint_time = current_time
