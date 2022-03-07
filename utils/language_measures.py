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

"""Implementes different measures for the language (e.g., topographic similarity).
"""
import itertools

import editdistance
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from emergent_communication_at_scale import types
from emergent_communication_at_scale.utils import utils as emcom_utils


def edit_dist(alist):
  """Computes edit distance of strings.

  Args:
    alist: a sequence of symbols (integers).

  Returns:
    a list of normalized levenshtein distance values for each pair of elements.
  """
  distances = []
  for i, el1 in enumerate(alist[:-1]):
    for _, el2 in enumerate(alist[i+1:]):
      # Normalized edit distance (same in our case as length is fixed)
      distances.append(editdistance.eval(el1.tolist(), el2.tolist()) / len(el1))
  return distances


def vmap_cosine_dist(alist):
  """Applies jnp_distancecosine on all possible distinct combinations of alist."""
  all_combinations = jax.tree_map(
      lambda x: jnp.array(list(itertools.combinations(x, 2))), alist)
  distances = jax.tree_map(jax.vmap(jnp_distancecosine), all_combinations)
  if isinstance(distances, dict):
    # Average the distances across attributes.
    distances = np.mean([*distances.values()], 0)
  return distances


@jax.jit
def jnp_distancecosine(vectors):
  """Computes cosine similarity between (u,v) = vectors."""
  u, v = vectors
  dist = emcom_utils.cosine_loss(u, v)
  return dist


def topographic_similarity(messages, meanings):
  """Computes topographic similarity."""

  if isinstance(meanings, dict):
    ashape = [*meanings.values()][0].shape[0]
  else:
    ashape = meanings.shape[0]
  assert messages.shape[0] == ashape

  distance_messages = edit_dist(messages)
  distance_inputs = list(vmap_cosine_dist(meanings))

  corr = stats.spearmanr(distance_messages, distance_inputs).correlation
  return corr


def games_topographic_similarity(games,
                                 meaning_sim,
                                 task=types.Task.CLASSIFICATION):
  """Factory to get the right meaning space."""
  messages = games.speaker_outputs.action
  if meaning_sim == types.MeaningSimilarity.INPUTS:
    meanings = games.speaker_inp
  elif meaning_sim == types.MeaningSimilarity.ATTRIBUTES:
    assert task in games.labels
    meanings = games.labels[task]
    if task == types.Task.MULTICLASSIFICATION:
      meanings = jax.tree_map(lambda x: x.reshape(x.shape[0], -1), meanings)
  else:
    raise ValueError(f'Wrong topsim_meaning_similarity value: {meaning_sim}')

  return topographic_similarity(messages=messages, meanings=meanings)


if __name__ == '__main__':
  all_meanings = jnp.array([[1., 0., 0., 0., 1., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 1., 0., 0.],
                            [1., 0., 0., 0., 0., 0., 1., 0.],
                            [1., 0., 0., 0., 0., 0., 0., 1.],
                            [0., 1., 0., 0., 1., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 1., 0., 0.],
                            [0., 1., 0., 0., 0., 0., 1., 0.],
                            [0., 1., 0., 0., 0., 0., 0., 1.],
                            [0., 0., 1., 0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 1., 0., 0.],
                            [0., 0., 1., 0., 0., 0., 1., 0.],
                            [0., 0., 1., 0., 0., 0., 0., 1.],
                            [0., 0., 0., 1., 1., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 1., 0.],
                            [0., 0., 0., 1., 0., 0., 0., 1.]])

  high_messages = jnp.array([[0, 0, 0],
                             [0, 0, 1],
                             [0, 0, 2],
                             [0, 0, 3],
                             [0, 1, 0],
                             [0, 1, 1],
                             [0, 1, 2],
                             [0, 1, 3],
                             [2, 0, 0],
                             [2, 0, 1],
                             [2, 0, 2],
                             [2, 0, 3],
                             [2, 1, 0],
                             [2, 1, 1],
                             [2, 1, 2],
                             [2, 1, 3]])
  low_messages = jnp.array([[1, 2, 3],
                            [2, 2, 0],
                            [3, 0, 1],
                            [1, 3, 1],
                            [1, 0, 3],
                            [2, 0, 0],
                            [3, 1, 1],
                            [1, 0, 1],
                            [0, 2, 3],
                            [0, 2, 0],
                            [0, 0, 1],
                            [0, 3, 1],
                            [1, 2, 0],
                            [2, 2, 1],
                            [3, 0, 0],
                            [1, 3, 0]])

  low_topsim = topographic_similarity(messages=low_messages,
                                      meanings=all_meanings)
  high_topsim = topographic_similarity(messages=high_messages,
                                       meanings=all_meanings)

  print(low_topsim, high_topsim)
