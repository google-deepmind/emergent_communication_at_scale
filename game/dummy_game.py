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

"""Lewis Game."""

import numpy as np
from emergent_communication_at_scale import types
from emergent_communication_at_scale.game.game_interface import dispatch_per_device
from emergent_communication_at_scale.game.game_interface import Game


def iterator(num_games, max_steps, mode):
  """Iterator for dummy game."""

  obs = types.GamesInputs(
      speaker_inp=np.eye(num_games),
      labels=np.ones((num_games,)),
      misc=dict(),
  )
  if mode == 'train':
    obs = dispatch_per_device(obs)  # Dispatch only at training.
  for _ in range(max_steps):
    yield obs


class DummyGame(Game):
  """Dummy game for testing."""

  def __init__(self, train_batch_size, eval_batch_size, max_steps):
    super().__init__(train_batch_size, eval_batch_size)
    self._max_steps = max_steps

  def get_training_games(self, rng):
    del rng
    return iterator(self._train_batch_size, self._max_steps, mode='train')

  def get_evaluation_games(self, mode: str = 'test'):
    return iterator(self._eval_batch_size, self._max_steps, mode=mode)

  def evaluate(self, prediction, target):
    pass
