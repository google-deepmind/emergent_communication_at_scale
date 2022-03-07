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

"""Defines Postprocessing per dataset."""

import abc
from typing import Dict, NamedTuple
import numpy as np
import tensorflow.compat.v2 as tf
import tree
from emergent_communication_at_scale import types


####################
#  Abstract class  #
####################


class DatasetProcessing(abc.ABC):
  """Interface for postprocessing dataset with default options, e.g., noise."""

  @abc.abstractmethod
  def _process(self, data: Dict[str, np.ndarray]) -> types.GamesInputs:
    pass

  def __call__(self, data: Dict[str, tf.Tensor]) -> types.GamesInputs:
    """Main postprocessing call."""
    return self._process(data)


class DatasetInfo(NamedTuple):
  dataset_processing_fn: DatasetProcessing


#############
#  Factory  #
#############


def get(dataset_name: str, to_onehot_label: bool) -> DatasetInfo:
  """Simple helper to return the correct dataset and its tokenizer."""

  all_dataset = dict(
      byol_imagenet2012=DatasetInfo(
          dataset_processing_fn=ImagenetProcessing(to_onehot_label)),

      byol_celeb_a2=DatasetInfo(
          dataset_processing_fn=CelebAProcessing(to_onehot_label)),

  )

  if dataset_name in all_dataset:
    return all_dataset[dataset_name]

  else:
    raise ValueError(f'Invalid dataset name: {dataset_name}.'
                     f'Supported: {[all_dataset.keys()]}')


###########
#  Utils  #
###########


def _tree_to_one_hot(x, max_val):
  x = tree.map_structure(
      lambda a: tf.one_hot(tf.cast(a, tf.int32), depth=max_val, axis=-1),
      x)
  return x


###########################
#  Dataset preprocessing  #
###########################


class ImagenetProcessing(DatasetProcessing):
  """Turns Dataset with logits into a Discrimation Lewis Game."""

  def __init__(self, to_onehot_label):
    self._to_onehot_label = to_onehot_label

  def _process(self, data: Dict[str, np.ndarray]) -> types.GamesInputs:

    if self._to_onehot_label:
      labels = _tree_to_one_hot(data['label'], max_val=1000)
    else:
      labels = data['label']

    return types.GamesInputs(
        speaker_inp=data['logit'],
        labels={types.Task.CLASSIFICATION: {'class': labels}},
        misc=dict()
    )


class CelebAProcessing(DatasetProcessing):
  """Turns Dataset with logits into a Discrimation Lewis Game."""

  def __init__(self, to_onehot_label):
    self._to_onehot_label = to_onehot_label

  def _process(self, data: Dict[str, np.ndarray]) -> types.GamesInputs:

    attributes = _tree_to_one_hot(data['attributes'], max_val=2)

    # Config option to avoid crashing the memory while performing CPC
    if self._to_onehot_label:
      labels = _tree_to_one_hot(data['label'], max_val=10178)  # Big
    else:
      labels = data['label']

    return types.GamesInputs(
        speaker_inp=data['logit'],
        labels={
            types.Task.CLASSIFICATION: {'class': labels},
            types.Task.ATTRIBUTE: attributes,
            types.Task.LANDMARK: data['landmarks'],
        },
        misc=dict(image_id=data['image_id']),
    )


################
#  Miscellaneous
################


class NoiseBatchProcessingTf:
  """Adds Gaussian noise to the speaker input (logit or image)."""

  def __init__(self, coeff_noise):
    self._coeff_noise = coeff_noise

  def __call__(self, data: types.GamesInputs):
    """Apply noise on data through tensorflow (fast)."""

    x = data.speaker_inp

    # Collects input batch statistic.
    stddev = tf.math.reduce_std(x, axis=0)
    shape = tf.shape(x)

    ### Process first view.
    noise_view1 = tf.random.normal(shape, mean=0.0, stddev=stddev)
    data = data._replace(speaker_inp=x + self._coeff_noise * noise_view1)

    ### Process second view.
    # Creates second view.
    if types.Task.DISCRIMINATION not in data.labels:
      data.labels[types.Task.DISCRIMINATION] = tf.identity(x)  # Simulate copy.
    # We here assume that the view1/view2 stats are the same.
    noise_view2 = tf.random.normal(shape, mean=0.0, stddev=stddev)
    data.labels[types.Task.DISCRIMINATION] += self._coeff_noise * noise_view2

    return data
