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

  config.experiment = 'lewis'
  config.training_steps = int(2e5)

  # Define global storage folder (ckpt, logs etc.)
  config.checkpoint_dir = '/tmp/cidre_ckpts'

  # Basic jaxline logging options
  config.interval_type = 'secs'
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60

  # Put here values that are referenced multiple times
  config.vocab_size = 20
  config.length = 10
  config.task = types.Task.CLASSIFICATION

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              debug=False,
              speaker_optimizer=dict(
                  name='adam',
                  learning_rate=0.0001,
                  kwargs=dict(adam=dict()),
              ),
              listener_optimizer=dict(
                  name='adam',
                  learning_rate=0.0001,
                  kwargs=dict(adam=dict()),
              ),
              training=dict(
                  batch_size=1024,
                  length=get_value('length'),
                  target_update_ema=0.99,
                  steps=get_value('training_steps'),
              ),
              population=dict(
                  n_speakers=1,
                  n_listeners=1,
                  num_agents_per_step=1,
              ),
              speaker=dict(
                  length=get_value('length'),
                  vocab_size=get_value('vocab_size'),
                  torso_config=dict(
                      torso_type=types.TorsoType.IDENTITY,
                      torso_kwargs=dict(),
                  ),
                  embedder_config=dict(
                      torso_type=types.TorsoType.DISCRETE,
                      torso_kwargs=dict(
                          vocab_size=get_value('vocab_size') + 1,
                          embed_dim=10,
                          mlp_kwargs=dict(output_sizes=(),)),
                  ),
                  core_config=dict(
                      core_type=types.CoreType.LSTM,
                      core_kwargs=dict(hidden_size=256),
                  ),
                  head_config=dict(
                      head_type=types.SpeakerHeadType.POLICY_QVALUE_DUELING,
                      head_kwargs=dict(
                          hidden_sizes=(), num_actions=get_value('vocab_size')),
                      kwargs=dict(),
                  ),
              ),
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
              imitation=dict(
                  nbr_students=1,
                  imitation_step=None,
                  imitation_type=types.ImitationMode.BEST,
                  self_imitation=False,
              ),
              reset=dict(reset_step=None, reset_type=types.ResetMode.PAIR),
              evaluation=dict(
                  batch_size=1024,
                  subsampling_ratio=0.01,
                  max_n_agents=10,
                  topsim_meaning_similarity=types.MeaningSimilarity.INPUTS,
                  topsim_task=types.Task.CLASSIFICATION,
              ),
              loss=dict(
                  speaker=dict(
                      loss_type=types.SpeakerLossType.REINFORCE,
                      use_baseline=True,
                      speaker_entropy=1e-4,
                      kwargs=dict(
                          policy_gradient=dict(),
                          reinforce=dict(speaker_kl_target=0.5)),
                  ),
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
                      dummy=dict(  # Dataset used for testing.
                          max_steps=get_value('training_steps')),
                      visual_game=dict(
                          dataset_name='byol_imagenet2012',
                          # Important: Make sure to download the data
                          # and update here.
                          dataset_path='emergent_communication_at_scale/emcom_datasets/',
                          coeff_noise=0.0,
                          num_evaluation_epochs=5,
                          shuffle_evaluation=True,
                          shuffle_training=True,
                          is_one_hot_label=False,
                      ))),
              checkpointing=dict(
                  use_checkpointing=True,
                  checkpoint_dir=get_value('checkpoint_dir'),
                  save_checkpoint_interval=300,
                  filename='agents.pkl'
              ),
          ),))

  exp_config = config.experiment_kwargs.config
  if sweep == 'debug':
    config.experiment_kwargs.config.debug = True
    config.training_steps = int(1)
    config.interval_type = 'steps'
    config.log_train_data_interval = 1
    config.log_tensors_interval = 1
    exp_config.checkpointing.save_checkpoint_interval = 1
    exp_config.training.batch_size = 8
    exp_config.evaluation.batch_size = 8
    exp_config.evaluation.subsampling_ratio = 0.5

  elif sweep == 'celeba':
    # Game
    exp_config = config.experiment_kwargs.config
    exp_config.game.kwargs.visual_game.dataset_name = 'byol_celeb_a2'
    # Evaluation
    exp_config.evaluation.topsim_meaning_similarity = types.MeaningSimilarity.ATTRIBUTES
    exp_config.evaluation.subsampling_ratio = 0.01
    exp_config.evaluation.topsim_task = types.Task.ATTRIBUTE  # used for topsim

  elif sweep == 'imagenet':
    pass

  elif sweep == 'imagenet_imitation':
    # Set population size
    exp_config.population.n_speakers = 10
    exp_config.population.n_listeners = 10
    exp_config.population.num_agents_per_step = 10
    # Set imitation parameters
    exp_config.imitation.nbr_students = 4
    exp_config.imitation.imitation_step = 10

  else:
    raise ValueError(f'Sweep {sweep} not recognized.')

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config
