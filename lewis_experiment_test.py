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

"""Tests for emergent_communication_at_scale.Lewis."""
from absl.testing import absltest
import jax
from jaxline import platform
from jaxline import train
from emergent_communication_at_scale import lewis_experiment
from emergent_communication_at_scale.configs import lewis_config
from emergent_communication_at_scale.utils import eval_utils


class LewisTest(absltest.TestCase):

  def test_lewis(self):

    config = lewis_config.get_config('debug')
    xp_config = config.experiment_kwargs.config
    xp_config.game.name = 'dummy'  # Use dummy dataset.
    # Defines smaller architectures for quick testing.
    config.length = 5
    xp_config.speaker.core_config.core_kwargs.hidden_size = 8
    xp_config.listener.core_config.core_kwargs.hidden_size = 8
    xp_config.listener.head_config.head_kwargs.hidden_sizes = [8]

    checkpointer = platform.create_checkpointer(config, 'train')
    writer = platform.create_writer(config, 'train')
    ## Creates a `dummy` checkpoint for evaluation.
    temp_dir = self.create_tempdir().full_path
    xp_config.checkpointing.checkpoint_dir = temp_dir

    # Training step
    train.train(
        lewis_experiment.LewisExperiment,
        config,
        checkpointer,
        writer)

    # Evaluation step
    eval_utils.evaluate_final(config=config,
                              mode='eval_test_average',
                              rng=jax.random.PRNGKey(42))

if __name__ == '__main__':
  absltest.main()
