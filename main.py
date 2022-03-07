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

"""Specifies which experiment to launch."""


from absl import app
from absl import flags
import jax
from jaxline import platform
from emergent_communication_at_scale import eol_experiment
from emergent_communication_at_scale import lewis_experiment
from emergent_communication_at_scale.utils import eval_utils


FLAGS = flags.FLAGS


def main(argv):

  flags.mark_flag_as_required('config')
  config = FLAGS.config

  if config.experiment == 'lewis':
    # Training
    platform.main(lewis_experiment.LewisExperiment, argv)

    # Evaluation
    eval_utils.evaluate_final(config,
                              mode='eval_test_average',
                              rng=jax.random.PRNGKey(42))  # Deterministic eval

  elif config.experiment == 'ease_of_learning':
    platform.main(eol_experiment.EaseOfLearningExperiment, argv)

  else:
    raise ValueError(f'{config.experiment} not recognized. '
                     'Only lewis and ease_of_learning are supported')

if __name__ == '__main__':
  app.run(main)
