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

"""Listener losses."""

import abc
from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp
from emergent_communication_at_scale import types
from emergent_communication_at_scale.utils import utils as emcom_utils


class ListenerLoss(abc.ABC):
  """Abstract class implementing the listener loss."""

  def __init__(self, reward_type: types.RewardType) -> None:
    self._reward_type = reward_type

  @abc.abstractmethod
  def compute_ensemble_accuracy(
      self,
      prediction: chex.Array,
      games: types.Games,
  ) -> Dict[str, Any]:
    pass

  def compute_listener_loss(
      self,
      rng: chex.PRNGKey,
      games: types.Games,
  ) -> types.ListenerLossOutputs:
    """Computes the Listener loss."""

    # Computes the loss.
    output = self._compute_listener_loss(rng=rng, games=games)

    # Turns the loss into reward.
    if self._reward_type == 'logprob':
      reward = -output.loss  # [B]
    elif self._reward_type == 'success_rate':
      reward = output.accuracy  # [B]
    else:
      raise ValueError(f'Invalid reward reward type {self._reward_type}'
                       f'Should be one of [logprob, success_rate]')

    # Makes the reward non-differentiable.
    reward = jax.lax.stop_gradient(reward)  # [B]

    # Computes global loss and accuracies.
    global_loss = jnp.sum(output.loss, axis=0)
    global_accuracy = jnp.sum(output.accuracy, axis=0)

    # Adds global metrics to stats.
    stats = {
        'listener_loss': global_loss,
        'global_accuracy': global_accuracy,
        **output.stats
    }

    return types.ListenerLossOutputs(
        loss=global_loss,
        probs=output.probs,
        accuracy=global_accuracy,
        reward=reward,
        stats=stats,
    )

  @abc.abstractmethod
  def _compute_listener_loss(
      self,
      rng: chex.PRNGKey,
      games: types.Games,
  ) -> types.ListenerLossOutputs:
    pass


def listener_loss_factory(
    loss_type: types.ListenerLossType,
    kwargs: types.Config,
    **common_kwargs,
) -> ListenerLoss:
  """Factory to select the listener's loss."""

  loss_specific_kwargs = kwargs.get(loss_type, dict())
  all_kwargs = {**common_kwargs, **loss_specific_kwargs}

  if loss_type == types.ListenerLossType.CLASSIF:
    listener_loss = ClassificationListenerLoss(**all_kwargs)

  elif loss_type == types.ListenerLossType.CPC:
    listener_loss = CpcListenerLoss(**all_kwargs)

  else:
    raise ValueError(f'Incorrect listener loss type {loss_type}.')

  return listener_loss


class ClassificationListenerLoss(ListenerLoss):
  """Class implementing the classification loss."""

  def __init__(
      self,
      reward_type: types.RewardType,
      task: types.Task,
  ) -> None:
    super().__init__(reward_type=reward_type)
    self._task = task

  def compute_ensemble_accuracy(
      self,
      prediction: chex.ArrayTree,
      games: types.Games,
  ) -> types.Config:
    """Compute accuracy given a prediction."""
    assert self._task in games.labels
    labels = games.labels[self._task]  # {str: [B, F]}
    # Iterates over the attribute to compute an accuracy per attribute.
    accuracy_per_attr = jax.tree_map(lambda x, y: x == jnp.argmax(y, axis=-1),
                                     prediction, labels)  # {str: [B]}
    accuracy = jnp.stack(jax.tree_leaves(accuracy_per_attr))  # [|{}|, B]
    accuracy = jnp.mean(accuracy, axis=0)  # [B]

    return dict(ensemble_acc=jnp.sum(accuracy, axis=0))

  def _compute_listener_loss(
      self,
      rng: chex.PRNGKey,
      games: types.Games,
  ) -> types.ListenerLossOutputs:
    """Computes the Listener loss."""
    del rng  # Deterministic loss

    predictions = games.listener_outputs.predictions  # {str: [B, F]}
    assert self._task in games.labels
    labels = games.labels[self._task]  # {str: [B, F]}

    # Iterates over the attribute to compute an accuracy per attribute.
    accuracy_per_attr = jax.tree_map(
        lambda x, y: jnp.argmax(x, axis=-1) == jnp.argmax(y, axis=-1),
        predictions, labels)  # {str: [B]}
    global_accuracy = jnp.stack(
        jax.tree_leaves(accuracy_per_attr), axis=0)  # [|{}|, B]
    global_accuracy = jnp.mean(global_accuracy, axis=0)  # [B]
    listener_probs = jax.tree_map(jax.nn.softmax, predictions)
    listener_loss = jax.tree_map(emcom_utils.softmax_cross_entropy, predictions,
                                 labels)  # {str: [B]}
    listener_loss = jnp.stack(
        jax.tree_leaves(listener_loss), axis=0)  # [|{}|, B]
    listener_loss = jnp.mean(listener_loss, axis=0)  # [B]

    # Sums over the batch size.
    accuracy_per_attr = jax.tree_map(jnp.sum, accuracy_per_attr)

    stats = {f'accuracy_{k}': v for k, v in accuracy_per_attr.items()}

    return types.ListenerLossOutputs(
        loss=listener_loss,
        probs=listener_probs,
        accuracy=global_accuracy,
        stats=stats,
    )


class CpcListenerLoss(ListenerLoss):
  """Class implementing the CPC loss."""

  def __init__(
      self,
      reward_type: types.RewardType,
      num_distractors: int,
      cross_device: bool,
  ) -> None:
    super().__init__(reward_type=reward_type)
    self._num_distractors = num_distractors
    self._cross_device = cross_device

  def compute_ensemble_accuracy(self, prediction, games):
    """Computes accuracy given a prediction."""
    del games
    effective_batchsize = prediction.shape[0]
    num_distractors = self._num_distractors
    if num_distractors >= (effective_batchsize - 1):
      num_distractors = -1
    if num_distractors == -1:
      accuracy = (prediction == jnp.arange(effective_batchsize))
    else:
      accuracy = (prediction == 0)
    # Transforms accuracy from bool to integer.
    accuracy = accuracy * 1
    return dict(ensemble_acc=jnp.sum(accuracy, axis=0))

  def _compute_listener_loss(
      self,
      rng: chex.PRNGKey,
      games: types.Games,
  ) -> types.ListenerLossOutputs:
    """Computes CPC loss."""
    effective_batchsize, feature_dim = games.listener_outputs.targets.shape

    # Warning: at evaluation time, batch size is small.
    # Use all the batch as distractors at eval time.
    if self._num_distractors >= (effective_batchsize - 1):
      self._num_distractors = -1

    if self._num_distractors == -1:
      # Computes CPC on the full batch.
      predictions = games.listener_outputs.predictions
      targets = games.listener_outputs.targets
      batch_indices = jnp.arange(effective_batchsize)

      # If we are on multiple devices we have to gather targets from other
      # devices and offset batch indices by the device id.
      # We do not pmap the init to gain compilation time so we do not gather
      # across devices at init.
      if jax.device_count() > 1 and self._cross_device:
        targets = jax.lax.all_gather(
            targets, axis_name='i')  # Num_devices, B, F
        targets = targets.reshape(-1, feature_dim)  # Large_Batch, F
        global_batch_indices = batch_indices + jax.lax.axis_index(
            'i') * effective_batchsize
      else:
        global_batch_indices = batch_indices

      cosine_sim = -emcom_utils.cosine_loss(predictions[:, None, :],
                                            targets[None, :, :])
      listener_probs = jax.nn.softmax(cosine_sim, axis=-1)
      listener_loss = -jax.nn.log_softmax(
          cosine_sim, axis=-1)[batch_indices, global_batch_indices]
      accuracy = (jnp.argmax(cosine_sim, axis=-1) == global_batch_indices)

    else:
      # Computes CPC on a predefined numbner of distractors.
      batch_distractors = []
      for i in range(effective_batchsize):
        key_rng, rng = jax.random.split(rng)
        potential_distractors_idx = list(range(effective_batchsize))
        potential_distractors_idx.remove(i)
        distractor_idx = jax.random.choice(
            key_rng,
            jnp.array(potential_distractors_idx),
            shape=[self._num_distractors],
            replace=False)
        distractors = jnp.take(
            games.listener_outputs.targets, distractor_idx, axis=0)
        target = games.listener_outputs.targets[i:(i + 1)]
        batch_distractors.append(jnp.concatenate([target, distractors], axis=0))

      targets = jnp.stack(batch_distractors, axis=0)
      cosine_sim = -emcom_utils.cosine_loss(
          games.listener_outputs.predictions[:, None, :], targets)
      listener_probs = jax.nn.softmax(cosine_sim, axis=-1)
      # By construction the target is in position 0.
      listener_loss = -jax.nn.log_softmax(cosine_sim, axis=-1)[:, 0]
      accuracy = (jnp.argmax(cosine_sim, axis=-1) == 0)
    # Transforms accuracy from bool to integer.
    accuracy = accuracy * 1

    return types.ListenerLossOutputs(
        loss=listener_loss,
        probs=listener_probs,
        accuracy=accuracy,
        stats=dict(),
    )
