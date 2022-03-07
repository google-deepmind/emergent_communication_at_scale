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

from typing import Optional, Text

from absl import flags
from absl import logging

import chex
import jax

from ml_collections import config_dict
import numpy as np
from emergent_communication_at_scale import lewis_experiment


FLAGS = flags.FLAGS


def evaluate_final(config: Optional[config_dict.ConfigDict],
                   mode: Optional[Text],
                   rng: chex.PRNGKey):
  """The main evaluation loop.

  This loop loads a checkpoint and evaluates its performance on the
  test set, by calling experiment.evaluate.

  Args:
    config: Optional argument. Defines the config.
    mode: optional argument. Defines the mode of evalution. Coud be any value in
    eval_{test/valid}_{average/vote}. Default (eval_test_average).
    rng: select evaluation seed (recommended to always use the same)
  """
  if config is None:
    config = FLAGS.config.experiment_kwargs.config
  else:
    config = config.experiment_kwargs.config

  if config.checkpointing.use_checkpointing:
    logging.info('\nEvaluating the final checkpoint on the test set.\n')
    init_rng, eval_rng = jax.random.split(rng)
    exp = lewis_experiment.LewisExperiment(mode=mode,
                                           init_rng=init_rng,
                                           config=config)

    step, _ = exp.restore_state(exp.checkpoint_path)
    exp.evaluate(global_step=np.array(step), rng=eval_rng)
  else:
    logging.info('\nCheckpointing not available for evaluation.\n')
    return
