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

"""Helper to create Celeb_a split wi."""

import collections
import os
import re
from absl import app


DATASET_PATH = '.'

IMG = os.path.join(DATASET_PATH, 'img_align_celeba')

CLASS_IDX = os.path.join(DATASET_PATH, 'identity_CelebA.txt')
ATTRIBUTE_VALUES = os.path.join(DATASET_PATH, 'list_attr_celeba.txt')
LANDMARKS = os.path.join(DATASET_PATH, 'list_landmarks_align_celeba.txt')
TRAIN_SPLIT = os.path.join(DATASET_PATH, 'list_eval_partition.txt')


use_split_perso = True
ratio = 5


def main(argv):
  del argv

  dataset = collections.OrderedDict()

  ### Load images
  with open(CLASS_IDX, 'r') as f:
    for line in f:
      # Parse line
      line = line.strip()
      img_id, label = line.split(' ')
      dataset[img_id] = dict()  # create sample entries
      dataset[img_id]['label'] = int(label)
      dataset[img_id]['image_id'] = int(img_id.split('.')[0])
      dataset[img_id]['filename'] = img_id.encode('utf-8')

  attribute_names = []
  with open(ATTRIBUTE_VALUES, 'r') as f:
    for i, line in enumerate(f):
      # Parse line
      line = line.strip()
      if i == 0:
        assert len(dataset) == int(line)
      elif i == 1:
        attribute_names = line.split(' ')
      else:
        line = re.sub(' +', ' ', line)
        info = line.split(' ')
        img_id, attr_values = info[0], info[1:]
        attr_values = [val == '1' for val in  attr_values]
        attributes = {k: v for k, v in zip(attribute_names, attr_values)}
        # Store data
        dataset[img_id]['attributes'] = attributes

  landmark_names = []
  with open(LANDMARKS, 'r') as f:
    for i, line in enumerate(f):
      # Parse line
      line = line.strip()
      if i == 0:
        assert len(dataset) == int(line)
      elif i == 1:
        landmark_names = line.split(' ')
      else:
        line = re.sub(' +', ' ', line)
        info = line.split(' ')
        img_id, landmarks = info[0], info[1:]
        landmarks = [int(l) for l in landmarks]
        landmarks = {k: v for k, v in zip(landmark_names, landmarks)}
        # Store data
        dataset[img_id]['landmarks'] = landmarks

  # Split train/test set from official split
  image_train, image_valid, image_test = [], [], []
  if use_split_perso:
    counter_label = collections.Counter()
    for data in dataset.values():
      label = data['label']
      count = counter_label[label]
      if count > 0 and count % ratio == 0:
        if label % 2 == 0:
          image_valid.append(data)
        else:
          image_test.append(data)
      else:
        image_train.append(data)
      counter_label[label] += 1

  else:
    with open(TRAIN_SPLIT, 'r') as f:
      for line in f:
        # Parse line
        line = line.strip()
        img_id, split_id = line.split(' ')
        split_id = int(split_id)
        if split_id == 0:
          image_train.append(dataset[img_id])
        elif split_id == 1:
          image_valid.append(dataset[img_id])
        elif split_id == 2:
          image_test.append(dataset[img_id])
        else:
          assert False

  print('Done!')

  return image_train, image_valid, image_test


if __name__ == '__main__':
  app.run(main)
