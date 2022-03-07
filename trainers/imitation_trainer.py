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

"""Helper to sample a oracles/students."""

import chex
import jax
from jaxline import utils
import numpy as np
from emergent_communication_at_scale import types
from emergent_communication_at_scale.utils import population_storage as ps


@jax.pmap
def _split_keys_pmap(key):
  return tuple(jax.random.split(key))


class ImitateTrainer():
  """Class implementation for imitation over speaker agents."""

  def __init__(
      self,
      n_speakers: int,
      imitation_update_fn,
  ):
    # Pmap imitation update function.
    self._n_speakers = n_speakers
    self._pmap_imitation_learning = jax.pmap(imitation_update_fn, axis_name='i')

  def imitate(
      self,
      rng: chex.PRNGKey,
      games: types.GamesInputs,
      agent_storage: ps.PopulationStorage,
      nbr_students: int,
      imitation_step: int,
      imitation_type: types.ImitationMode,
      self_imitation: bool,
  ):
    """Implements imitation learning with different modes."""
    del imitation_step  # Unused.
    def get_oracle_student_id(rng: chex.PRNGKey):
      if self._n_speakers > 1:
        speaker_ids = list(
            jax.random.choice(
                key=utils.get_first(rng),
                a=self._n_speakers,
                shape=[nbr_students + 1],
                replace=False))
        # Gets a speaker as oracle (depending on imitation mode).
        # Sets the rest as students.
        func = lambda id: utils.get_first(  # pylint: disable=g-long-lambda
            agent_storage.load_speaker(id).states['speaker']['avg_score'])
        scores = list(map(func, speaker_ids))

        if imitation_type == types.ImitationMode.BEST:
          oracle_id = speaker_ids[np.argmax(scores)]
        elif imitation_type == types.ImitationMode.WORST:
          oracle_id = speaker_ids[np.argmin(scores)]
        elif imitation_type == types.ImitationMode.RANDOM:
          oracle_id = speaker_ids[0]
        else:
          raise ValueError(f'Wrong imitation type: {imitation_type}.')

        speaker_ids.remove(oracle_id)

      elif (self._n_speakers == 1) and self_imitation:
        # Self-imitation case.
        speaker_ids = [0]
        oracle_id = 0
      else:
        raise ValueError('There is no imitation.')

      return speaker_ids, oracle_id

    rng_sampling, rng_training = _split_keys_pmap(rng)
    student_ids, oracle_id = get_oracle_student_id(rng=rng_sampling)

    # Imitation or self-imitation scenarios.
    oracle_properties = agent_storage.load_speaker(oracle_id)
    for student_id in student_ids:
      student_properties = agent_storage.load_speaker(student_id)

      new_params, new_states, new_opt_state, imit_scalar = self._pmap_imitation_learning(
          params_oracle=oracle_properties.params,
          params_student=student_properties.params,
          state_oracle=oracle_properties.states,
          state_student=student_properties.states,
          opt_state=student_properties.opt_states,
          games=games,
          rng=rng_training)
      # Updates params/states.
      agent_storage.store_agent(
          agent_id=student_id,
          agent_name='speaker',
          params=new_params,
          states=new_states,
          opt_states=new_opt_state)

    # Returns the scalar of the last imitation training with no pmaped dim.
    return dict(imitation_loss=utils.get_first(imit_scalar)), agent_storage
