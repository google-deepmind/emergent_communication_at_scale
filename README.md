# Emergent communication at scale

This repo allows exploring different dynamics and RL algorithms
to train a population of agents that communicate to solve a referential game.
It is based on the [Jax](https://github.com/google/jax)/[Haiku](https://github.com/deepmind/dm-haiku)
python api, and the [Jaxline](https://github.com/deepmind/jaxline) training framework,
smoothly allowing for large-scale and multi-device experiments.
The repo contains the code which we used for the experiments reported in our
[ICLR 2022 paper](https://openreview.net/forum?id=AUGBfDIV9rL), titled "Emergent Communication At Scale".

## Downloading dataset
Download the datasets [CelebA logits](https://storage.googleapis.com/dm_emcom_at_scale_dataset/byol_celeb_a2.tar.gz)
and [ImageNet logits](https://storage.googleapis.com/dm_emcom_at_scale_dataset/byol_imagenet2012.tar.gz) in the `emcom_datasets/` directory.

## Installation
Install dependencies by running the following commands:

```shell
# Create a virtual env
python3 -m venv ~/.venv/emcom
# Switch to the virtual env
source ~/.venv/emcom/bin/activate
# Install other dependencies
pip install -r requirements.txt
# Manually install the latest Jaxline version, since required 0.0.6 is not yet on pypi.
pip install git+https://github.com/deepmind/jaxline
```

### Lewis game
To run the Lewis game experiment
The Lewis game experiment trains a population of agents and
computes different metrics on the test set. To execute the Lewis game
experiment, run (from the `root` directory):

```shell
$ python -m emergent_communication_at_scale.main --config=emergent_communication_at_scale/configs/lewis_config.py
```

The ease of learning experiment assesses how easy and fast an
emergent language is transmitted to new listeners.
See our [paper](https://openreview.net/forum?id=AUGBfDIV9rL) for more details
about this metric.
To run the ease of learning experiment,
*given a previous checkpoint saved by the Lewis experiment*, you can use this command:

```shell
$ python -m emergent_communication_at_scale.main --config=emergent_communication_at_scale/configs/ease_of_learning_config.py
```

## Citing this work
If you find this code or the ideas in the paper useful in your research,
please consider citing the paper:

```bibtex
@inproceedings{emcom_scale,
    title={Emergent Communication At Scale},
    author={Chaabouni, Rahma and Strub, Florian and Altché, Florent and Tallec, Corentin and Trassov, Eugene and Davoodi, Elnaz and Mathewson, Kory and Tieleman, Olivier and Lazaridou, Angeliki and Piot, Bilal},
    booktitle={International Conference on Learning Representations},
    year={2022}
}
```

## License

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY).  You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

### Dependencies

editdistance used by permission under MIT license Copyright (c) 2013 Hiroyuki Tanaka.

NumPy copyright 2022 NumPy.

SciPy copyright 2022 SciPy.

## Disclaimer
This is not an official Google product.
