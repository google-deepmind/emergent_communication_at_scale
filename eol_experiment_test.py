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

"""Tests for emergent_communication_at_scale.ease_of_learning."""

from absl.testing import absltest
from jaxline import platform
from jaxline import train
from jaxline import utils
from emergent_communication_at_scale import eol_experiment
from emergent_communication_at_scale import lewis_experiment
from emergent_communication_at_scale.configs import ease_of_learning_config
from emergent_communication_at_scale.configs import lewis_config


class EaseOfLearningTest(absltest.TestCase):

  def test_ease_learning(self):

    # Performs Experiment training to build a checkpoint for ease of learning.
    config = lewis_config.get_config('debug')
    config.training_steps = 1
    xp_config = config.experiment_kwargs.config
    xp_config.game.name = 'dummy'  # Use dummy dataset.
    # Defines smaller architectures for quick testing.
    config.length = 5
    xp_config.speaker.core_config.core_kwargs.hidden_size = 8
    xp_config.listener.core_config.core_kwargs.hidden_size = 8
    xp_config.listener.head_config.head_kwargs.hidden_sizes = [8]

    checkpointer = platform.create_checkpointer(config, 'train')
    writer = platform.create_writer(config, 'train')

    ## Creates a `dummy` checkpoint for the ease of learning experiment.
    temp_dir = self.create_tempdir().full_path
    xp_config.checkpointing.checkpoint_dir = temp_dir
    train.train(
        lewis_experiment.LewisExperiment,
        config,
        checkpointer,
        writer)

    # Ease of learning test.
    utils.GLOBAL_CHECKPOINT_DICT = {}  # Overrides jaxline global chkpt dict.
    config = ease_of_learning_config.get_config('debug')
    config.training_steps = 1
    xp_config = config.experiment_kwargs.config
    xp_config.game.name = 'dummy'
    # Defines smaller architectures for quick testing.
    config.length = 5
    xp_config.listener.core_config.core_kwargs.hidden_size = 8
    xp_config.listener.head_config.head_kwargs.hidden_sizes = [8]

    xp_config.speaker_path.path = f'{temp_dir}/agents.pkl'

    train.train(eol_experiment.EaseOfLearningExperiment, config,
                checkpointer, writer)  # Uses same checkpointer and writer.


if __name__ == '__main__':
  absltest.main()
