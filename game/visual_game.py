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

"""Visual Lewis Game.

Pretrained image logits/representations are loaded thought tfds before being
split into distractors and targets.
"""
from typing import Any, Callable, Optional

from absl import logging
import chex
import jax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from emergent_communication_at_scale.game import dataset as visual_dataset
from emergent_communication_at_scale.game.game_interface import batch_size_per_device
from emergent_communication_at_scale.game.game_interface import Game

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Randomness option
EVAL_FIX_SEED = 42
tf.random.set_seed(42)  # Make tensorflow deterministic.


def _load_dataset(dataset_name: str,
                  dataset_path: str,
                  prefix: str,
                  shuffle_files: bool,
                  shuffle_seed: int,
                  reshuffle_iteration: bool) -> tf.data.Dataset:
  """Allow to load dataset from either tfds recording or dm.auto_dataset."""

  return tfds.load(dataset_name,
                   split=prefix,
                   shuffle_files=shuffle_files,
                   data_dir=dataset_path,
                   read_config=tfds.ReadConfig(
                       shuffle_seed=shuffle_seed,
                       shuffle_reshuffle_each_iteration=reshuffle_iteration))


def _process_dataset(
    ds,
    batch_size: Optional[int],  # None or zero -> no batch size
    num_epoch: Optional[int],  # None -> infinite loop
    cache: bool,
    use_shards: bool,  # to avoid multi-device sample duplication
    drop_remainder: bool,
    shuffle: bool,
    shuffling_buffer: Optional[int] = None,
    seed: Optional[int] = None,
    preprocess_fn: Optional[Callable[[Any], Any]] = None,
    batchprocess_fn: Optional[Callable[[Any], Any]] = None,
    prefetch: int = AUTOTUNE,
    num_parallel_call: int = AUTOTUNE,
) -> tf.data.Dataset:
  """Creates dataset by enforcing valid call orders."""

  if use_shards:  # Must be done BEFORE repeat
    ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())

  if preprocess_fn:
    ds = ds.map(preprocess_fn, num_parallel_calls=num_parallel_call)

  if cache:  # Must be done BEFORE shuffling
    ds = ds.cache()

  ds = ds.repeat(num_epoch)

  if shuffle:
    assert seed is not None, '"seed" must be defined when shuffling'
    assert shuffling_buffer, '"shuffle_buffer" must be defined when shuffling'
    ds = ds.shuffle(shuffling_buffer, seed=seed, reshuffle_each_iteration=True)

  if batch_size is not None and batch_size > 0:
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

  ds = ds.prefetch(prefetch)

  if batchprocess_fn:
    ds = ds.map(batchprocess_fn, num_parallel_calls=num_parallel_call)

  return ds


class LogitLoader(Game):
  """Simple container class."""

  def __init__(
      self,
      dataset_name: str,
      dataset_path: str,
      train_batch_size: int,
      eval_batch_size: int,
      shuffle_training: bool = True,
      shuffle_evaluation: bool = False,
      num_evaluation_epochs: int = 1,
      use_shards=True,
      drop_remainder: bool = False,
      coeff_noise: float = 0.0,
      is_one_hot_label: bool = True,
  ):
    """Dataset using modified loader from @defauw.

    Args:
      dataset_name: name of the dataset to load
      dataset_path: path to load tfds dataset from cns.
      train_batch_size: training minibatch size.
      eval_batch_size: evaluation (val & test) batch size.
      shuffle_training: whether or not to shuffle the training dataset.
      shuffle_evaluation: whether shuffling evaluation (negative samples CPC)
      num_evaluation_epochs: how many time to iterate over evaluation
      use_shards: use multiple shards across devices.
      drop_remainder: drop last elements of dataset (or pad them).
      coeff_noise: ratio of additional gaussian noise, at evaluation (for now).
      is_one_hot_label: specifies if labels need to be hotified.
    Returns:
      An dataset container object.

    """
    super().__init__(train_batch_size, eval_batch_size)

    self._dataset_name = dataset_name
    self._path = dataset_path.format(dataset_name)
    self._use_shards = use_shards

    self._shuffle_training = shuffle_training
    self._shuffle_evaluation = shuffle_evaluation
    self._num_evaluation_epochs = num_evaluation_epochs

    self._drop_remainder = drop_remainder

    dataset = visual_dataset.get(dataset_name, is_one_hot_label)
    self._dataset_processing_fn = dataset.dataset_processing_fn

    self._batchprocess_fn = None
    if coeff_noise > 0.:
      self._batchprocess_fn = visual_dataset.NoiseBatchProcessingTf(coeff_noise)

  @property
  def dataset_name(self):
    return self._dataset_name

  def get_training_games(self, rng: chex.PRNGKey):
    """See base class."""

    # Computes the batch size we need to dispatch per device.
    batch_size = batch_size_per_device(
        self.train_batch_size, num_devices=jax.device_count())

    ds = _load_dataset(dataset_name=self._dataset_name,
                       dataset_path=self._path,
                       prefix='train',
                       shuffle_files=True,
                       shuffle_seed=rng[-1].item(),
                       reshuffle_iteration=True)

    ds = _process_dataset(
        ds,
        batch_size,
        cache=False,
        num_epoch=None,  # Infinite iterator
        use_shards=True,
        preprocess_fn=self._dataset_processing_fn,
        batchprocess_fn=self._batchprocess_fn,  # only tf func
        drop_remainder=self._drop_remainder,
        shuffle=self._shuffle_training,
        shuffling_buffer=32768,  # <1Go RAM when 2048 feature size
        seed=rng[-1].item(),
    )

    logging.info('Dataset looks like %s.', ds)

    # Batch per devices
    ds = ds.batch(jax.local_device_count(), drop_remainder=self._drop_remainder)

    return ds.as_numpy_iterator()

  def get_evaluation_games(self, mode: str = 'test'):
    """Builds the evaluation input pipeline."""

    assert mode in ['test', 'test_official', 'valid']

    # On a general basis, it is safer to be single host for evaluation (GPU)
    assert jax.device_count() == 1
    assert jax.local_device_count() == 1

    ds = _load_dataset(dataset_name=self._dataset_name,
                       dataset_path=self._path,
                       prefix=mode,
                       shuffle_files=self._shuffle_evaluation,
                       shuffle_seed=EVAL_FIX_SEED,
                       reshuffle_iteration=True)

    ds = _process_dataset(
        ds,
        self._eval_batch_size,
        cache=False,
        use_shards=False,  # Single device at evaluation time
        num_epoch=self._num_evaluation_epochs,  # useful for CPC eval,
        preprocess_fn=self._dataset_processing_fn,
        batchprocess_fn=self._batchprocess_fn,  # only tf func
        drop_remainder=self._drop_remainder,
        shuffle=self._shuffle_evaluation,  # useful for CPC eval
        shuffling_buffer=32768,  # <1Go RAM when 2048 feature size
        seed=EVAL_FIX_SEED,  # we want the same shuffle every time
    )

    logging.info('Dataset looks like {%s}.', ds)

    return ds.as_numpy_iterator()
