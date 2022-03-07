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

"""Speaker losses."""

import abc
import chex
import jax
import jax.numpy as jnp
import rlax
from emergent_communication_at_scale import types


class SpeakerLoss(abc.ABC):
  """Abstract class implementing speaker losses."""

  @abc.abstractmethod
  def compute_speaker_loss(
      self,
      games: types.Games,
      reward: chex.Array,
  ) -> types.SpeakerLossOutputs:
    pass


def speaker_loss_factory(
    loss_type: types.SpeakerLossType,
    kwargs: types.Config,
    **common_kwargs,
) -> SpeakerLoss:
  """Speaker loss factory."""
  loss_specific_kwargs = kwargs.get(loss_type, dict())
  all_kwargs = {**common_kwargs, **loss_specific_kwargs}

  if loss_type == types.SpeakerLossType.REINFORCE:
    speaker_loss = ReinforceSpeakerLoss(**all_kwargs)

  elif loss_type == types.SpeakerLossType.POLICYGRADIENT:
    speaker_loss = PolicyGradientSpeakerLoss(**all_kwargs)

  else:
    raise ValueError(f'Incorrect speaker loss type {loss_type}.')
  return speaker_loss


class ReinforceSpeakerLoss(SpeakerLoss):
  """Class implementing the reinforce loss for the speaker."""

  def __init__(
      self,
      speaker_entropy: float,
      speaker_kl_target: float,
      use_baseline: bool = False,
  ) -> None:
    self._speaker_entropy = speaker_entropy
    self._speaker_kl_target = speaker_kl_target
    self._use_baseline = use_baseline

  def compute_speaker_loss(
      self,
      games: types.Games,
      reward: chex.Array,
  ) -> types.SpeakerLossOutputs:
    """Computes the reinforce loss."""
    if self._use_baseline:
      assert games.speaker_outputs.value is not None
      # Transforms to [T, B, F] receives [B, F, T].
      value = jnp.transpose(games.speaker_outputs.value, [2, 0, 1])
      value = jnp.squeeze(value, axis=-1)
    else:
      value = 0.0
    # Transforms to [T, B] receives [B, T].
    action_log_prob = jnp.transpose(games.speaker_outputs.action_log_prob,
                                    [1, 0])
    entropy = jnp.transpose(games.speaker_outputs.entropy, [1, 0])
    # Policy loss via Reinforce.
    sg_value = jax.lax.stop_gradient(value)
    policy_loss = -jnp.mean((reward - sg_value) * action_log_prob, axis=0)
    policy_loss = jnp.sum(policy_loss, axis=0)
    entropy = jnp.sum(jnp.mean(entropy, axis=0), axis=0)
    entropy_loss = -entropy * self._speaker_entropy

    if self._use_baseline:
      value_loss = jnp.mean(jnp.square(reward - value), axis=0)
      value_loss = jnp.sum(value_loss, axis=0)
      value_stats = jnp.sum(jnp.mean(value, axis=0), axis=0)
    else:
      value_loss = 0.0
      value_stats = 0.0

    # Transforms to [T, B, F] receives [B, F, T].
    speaker_policy_logits = jnp.transpose(games.speaker_outputs.policy_logits,
                                          [2, 0, 1])
    target_speaker_policy_logits = jnp.transpose(
        games.target_speaker_outputs.policy_logits, [2, 0, 1])

    kl_target_loss = rlax.softmax().kl(
        speaker_policy_logits,
        target_speaker_policy_logits) * self._speaker_kl_target

    kl_target_loss = jnp.sum(jnp.mean(kl_target_loss, axis=0), axis=0)

    speaker_loss = policy_loss + entropy_loss + value_loss + kl_target_loss
    stats = dict(
        value=value_stats,
        value_loss=value_loss,
        speaker_loss=speaker_loss,
        policy_loss=policy_loss,
        entropy_loss=entropy_loss,
        kl_target_loss=kl_target_loss,
        speaker_entropy=entropy,
    )

    return types.SpeakerLossOutputs(loss=speaker_loss, stats=stats)


class PolicyGradientSpeakerLoss(SpeakerLoss):
  """Class implementing the policy loss for the speaker."""

  def __init__(
      self,
      speaker_entropy: float,
      use_baseline: bool = False,
  ) -> None:
    self._speaker_entropy = speaker_entropy
    self._use_baseline = use_baseline

  def compute_speaker_loss(
      self,
      games: types.Games,
      reward: chex.Array,
  ) -> types.SpeakerLossOutputs:
    """Computes the policy gradient loss."""
    # Policy loss via policy gradient.
    # Transforms to [T, B, F] receives [B, F, T].
    assert games.speaker_outputs.q_values is not None
    q_values = jnp.transpose(games.speaker_outputs.q_values, [2, 0, 1])
    if self._use_baseline:
      assert games.speaker_outputs.value is not None
      value = jnp.transpose(games.speaker_outputs.value, [2, 0, 1])
      value = jnp.squeeze(value, axis=-1)
    else:
      value = 0.0

    # Transforms to [T, B] receives [B, T].
    action = jnp.transpose(games.speaker_outputs.action, [1, 0])
    entropy = jnp.transpose(games.speaker_outputs.entropy, [1, 0])
    action_log_prob = jnp.transpose(games.speaker_outputs.action_log_prob,
                                    [1, 0])
    q_value_chosen = rlax.batched_index(q_values, action)

    sg_q_value_chosen = jax.lax.stop_gradient(q_value_chosen)
    sg_value = jax.lax.stop_gradient(value)
    policy_loss = -jnp.mean(
        (sg_q_value_chosen - sg_value) * action_log_prob, axis=0)
    policy_loss = jnp.sum(policy_loss, axis=0)
    entropy_loss = -jnp.mean(entropy, axis=0) * self._speaker_entropy
    entropy_loss = jnp.sum(entropy_loss, axis=0)

    action_value_loss = jnp.mean(jnp.square(reward - q_value_chosen), axis=0)
    action_value_loss = jnp.sum(action_value_loss, axis=0)

    if self._use_baseline:
      value_loss = jnp.mean(jnp.square(reward - value), axis=0)
      value_loss = jnp.sum(value_loss, axis=0)
      value_stats = jnp.sum(jnp.mean(value, axis=0), axis=0)
    else:
      value_loss = 0.0
      value_stats = 0.0
    speaker_loss = policy_loss + entropy_loss + action_value_loss + value_loss
    stats = dict(
        q_value_chosen=jnp.sum(jnp.mean(q_value_chosen, axis=0), axis=0),
        value=value_stats,
        speaker_loss=speaker_loss,
        action_value_loss=action_value_loss,
        value_loss=value_loss,
        policy_loss=policy_loss,
        entropy_loss=entropy_loss,
    )

    return types.SpeakerLossOutputs(loss=speaker_loss, stats=stats)


class DummySpeakerLoss(SpeakerLoss):
  """Class implementing the policy loss for the speaker."""

  def compute_speaker_loss(
      self,
      games: types.Games,
      reward: chex.Array,
  ) -> types.SpeakerLossOutputs:
    """Computes the policy gradient loss."""
    del games, reward
    return types.SpeakerLossOutputs(loss=0., stats={})
