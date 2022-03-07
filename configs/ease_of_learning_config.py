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

"""Base config."""

from jaxline import base_config
from ml_collections import config_dict
from emergent_communication_at_scale import types

TASK_OVERRIDE = {}


def get_config(sweep='debug'):
  """Return config object for training."""

  config = base_config.get_base_config()
  get_value = lambda x, c=config: c.get_oneway_ref(x)

  config.experiment = 'ease_of_learning'

  # Define global storage folder (ckpt, logs etc.)
  config.checkpoint_dir = '/tmp/cidre_ckpts'

  # Overwrite plotting options
  config.interval_type = 'steps'
  config.log_train_data_interval = 100
  config.log_tensors_interval = 300

  # Basic jaxline logging options
  config.interval_type = 'secs'
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60

  config.training_steps = int(1e4)

  # Put here values that are referenced multiple times
  config.vocab_size = 20
  config.length = 10
  config.task = types.Task.CLASSIFICATION

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              debug=False,
              training=dict(
                  batch_size=1024,
                  length=get_value('length'),
                  steps=get_value('training_steps'),
              ),
              listener_optimizer=dict(
                  name='adam',
                  learning_rate=0.001,
                  kwargs=dict(adam=dict()),
              ),
              speaker_path=dict(
                  path=get_value('checkpoint_dir') + '/agents.pkl',
                  speaker_index=0),
              listener=dict(
                  torso_config=dict(
                      torso_type=types.TorsoType.DISCRETE,
                      torso_kwargs=dict(
                          vocab_size=get_value('vocab_size'),
                          embed_dim=10,
                          mlp_kwargs=dict(output_sizes=(),))),
                  core_config=dict(
                      core_type=types.CoreType.LSTM,
                      core_kwargs=dict(hidden_size=512),
                  ),
                  head_config=dict(
                      head_type=types.ListenerHeadType.CPC,
                      head_kwargs=dict(hidden_sizes=[256]),
                      kwargs=dict(
                          cpc=dict(),
                          mlp=dict(),
                          multi_mlp=dict(task=get_value('task')),
                      ),
                  ),
              ),
              population=dict(),   # Unused for EOL
              imitation=dict(),    # Unused for EOL
              reset=dict(),        # Unused for EOL
              evaluation=dict(),   # Unused for EOL
              loss=dict(
                  speaker=dict(),  # Unused for EOL
                  listener=dict(
                      loss_type=types.ListenerLossType.CPC,
                      reward_type=types.RewardType.SUCCESS_RATE,
                      kwargs=dict(
                          classif=dict(task=get_value('task')),
                          cpc=dict(num_distractors=-1, cross_device=True),
                      )),
              ),
              game=dict(
                  name='visual_game',
                  kwargs=dict(
                      dummy=dict(
                          max_steps=get_value('training_steps')),
                      visual_game=dict(
                          dataset_name='byol_imagenet2012',
                          # Important: Make sure to download the data
                          # and update here.
                          dataset_path='emergent_communication_at_scale/emcom_datasets/',
                          coeff_noise=0.0,
                          shuffle_training=True,
                          is_one_hot_label=True,
                      ))),
              checkpointing=dict(
                  use_checkpointing=True,
                  checkpoint_dir=get_value('checkpoint_dir'),
                  save_checkpoint_interval=0,
                  filename='agents_eol.pkl'
              ),
          ),))

  if sweep == 'debug':
    config.experiment_kwargs.config.debug = True
    config.interval_type = 'steps'
    config.training_steps = int(1)
    config.log_train_data_interval = 1
    config.log_tensors_interval = 1
    exp_config = config.experiment_kwargs.config
    exp_config.training.batch_size = 8

  elif sweep == 'celeba':
    # Game
    exp_config = config.experiment_kwargs.config
    exp_config.game.kwargs.visual_game.dataset_name = 'byol_celeb_a2'

  elif sweep == 'imagenet':
    pass
  else:
    raise ValueError(f'Sweep {sweep} is not recognized.')

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config
