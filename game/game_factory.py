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

"""Init file to locate correct dataset and its configuration."""


from ml_collections import config_dict
from emergent_communication_at_scale.game import visual_game
from emergent_communication_at_scale.game.dummy_game import DummyGame
from emergent_communication_at_scale.game.game_interface import Game


def get(
    config: config_dict.ConfigDict,
    train_batch_size: int,
    eval_batch_size: int,
) -> Game:
  """Simple helper to return the correct dataset and its tokenizer."""

  name = config.name
  game_kwargs = config.kwargs[name]

  if name == 'visual_game':
    game = visual_game.LogitLoader(
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        **game_kwargs,
    )

  elif name == 'dummy':
    game = DummyGame(
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        **game_kwargs,
    )

  else:
    raise ValueError(f'Invalid game name, {name}.')

  return game
