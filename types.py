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

"""Constructs typing of different Objects needed in CIDRE."""

import enum
from typing import Any, Dict, List, NamedTuple, Optional, Union

import chex
import haiku as hk
from ml_collections import config_dict
import optax

Config = Union[Dict[str, Any], config_dict.ConfigDict]
RNNState = chex.ArrayTree
AllParams = Dict[str, List[Optional[hk.Params]]]
AllStates = Dict[str, List[Optional[hk.State]]]
AllOptStates = Dict[str, List[Optional[optax.OptState]]]


class TrainingMode(enum.Enum):
  TRAINING = 'training'
  EVAL = 'eval'
  FORCING = 'forcing'


class ImitationMode(enum.Enum):
  BEST = 'best'
  RANDOM = 'random'
  WORST = 'worst'


class ResetMode:
  PAIR = 'pair'
  SPEAKER = 'speaker'
  LISTENER = 'listener'


class Task:
  CLASSIFICATION = 'classification'
  REGRESSION = 'regression'
  MULTICLASSIFICATION = 'multiclassification'
  LANDMARK = 'landmark'
  ATTRIBUTE = 'attributes'
  DISCRIMINATION = 'discrimination'
  IMAGES = 'images'


class MeaningSimilarity:
  INPUTS = 'inputs'
  ATTRIBUTES = 'attributes'


class RewardType:
  SUCCESS_RATE = 'success_rate'
  LOG_PROB = 'log_prob'


class CoreType:
  LSTM = 'lstm'
  GRU = 'gru'
  IDENTITY = 'identity'


class TorsoType:
  DISCRETE = 'discrete'
  MLP = 'mlp'
  IDENTITY = 'identity'


class ListenerHeadType:
  MLP = 'mlp'
  CPC = 'cpc'
  MULTIMLP = 'multi_mlp'


class SpeakerHeadType:
  POLICY = 'policy'
  POLICY_QVALUE = 'policy_q_value'
  POLICY_QVALUE_DUELING = 'policy_q_value_dueling'


class ListenerHeadOutputs(NamedTuple):
  predictions: chex.ArrayTree
  targets: Optional[chex.ArrayTree] = None


class SpeakerHeadOutputs(NamedTuple):
  policy_logits: chex.Array
  q_values: Optional[chex.Array] = None
  value: Optional[chex.Array] = None


class DuelingHeadOutputs(NamedTuple):
  q_values: chex.Array
  value: chex.Array


class Params(NamedTuple):
  speaker: hk.Params
  listener: hk.Params
  target_speaker: Optional[hk.Params]


class States(NamedTuple):
  speaker: hk.State
  listener: hk.State
  target_speaker: Optional[hk.State]


class OptStates(NamedTuple):
  speaker: optax.OptState
  listener: optax.OptState


class AgentProperties(NamedTuple):
  params: hk.Params
  opt_states: optax.OptState
  states: hk.State
  target_params: Optional[hk.Params] = None
  target_states: Optional[hk.State] = None


class AllProperties(NamedTuple):
  params: AllParams
  states: AllStates
  opt_states: AllOptStates


class SpeakerOutputs(NamedTuple):
  action: chex.Array
  action_log_prob: chex.Array
  entropy: chex.Array
  policy_logits: chex.Array
  q_values: Optional[chex.Array] = None
  value: Optional[chex.Array] = None


class ListenerOutputs(NamedTuple):
  predictions: chex.ArrayTree
  targets: Optional[chex.ArrayTree] = None


class AgentLossOutputs(NamedTuple):
  loss: chex.Array
  probs: chex.Array
  stats: Config


class ListenerLossOutputs(NamedTuple):
  loss: chex.Array
  accuracy: chex.Array
  probs: chex.Array
  stats: Config
  reward: Optional[chex.Array] = None


class SpeakerLossOutputs(NamedTuple):
  loss: chex.Array
  stats: Config


class ListenerLossType:
  CLASSIF = 'classif'
  CPC = 'cpc'


class SpeakerLossType:
  REINFORCE = 'reinforce'
  POLICYGRADIENT = 'policy_gradient'


class GamesInputs(NamedTuple):
  speaker_inp: chex.Array
  labels: Optional[chex.ArrayTree] = None
  misc: Dict[str, Any] = dict()  # to store debug information


class Games(NamedTuple):
  speaker_inp: chex.Array
  labels: Optional[chex.ArrayTree] = None
  speaker_outputs: Optional[SpeakerOutputs] = None
  target_speaker_outputs: Optional[SpeakerOutputs] = None
  listener_outputs: Optional[ListenerOutputs] = None
