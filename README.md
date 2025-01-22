# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash

wget https://openslr.org/resources/11/4-gram.arpa.gz
wget https://us.openslr.org/resources/11/librispeech-vocab.txt

mv 4gram.arpa lm/
mv librispeech-vocab.txt lm/

python3 train.py
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

## Обучение

Модель: DeepSpeech2Model
LM: 4gram(https://www.openslr.org/11/)
Словарь: librispeech-vocab.txt(https://www.openslr.org/11/)
Токенизатор: BPE токенизатор

Train set: Train-100-clean (больший датасет не разместить в kaggle)
Val set: dev-clean
Test set: Test-clean

Кол-во эпох: 7 эпох при длине 750 (далее выходит на плато)

BatchSize: 32 (возможно больший batchsize был бы лучше, но он затратнее)

Аугментация:
   Усиление звука
   Добавление Цветного шума
   Маскирование по времени
   Маскирование по частоте

GPU: 4 часа на P100

## Качество

test_CER_(Argmax): 0.1003
test_WER_(Argmax): 0.3092
test_CER_(BeamSearch): 0.0761
test_WER_(BeamSearch): 0.1859

Качество соответствует 70 баллам

В BeamSearch и используется LM 4gram что и, скорее всего, дало прирост метрики (+10 баллов)
Также используется bpe токенизатор, но эксперименты с ним/без не проводились

Итого модель может претендовать на 80 баллов

Отчет Wandb:
https://wandb.ai/pospelov_artem/hw_asr/runs/mo2d5300?nw=nwuserpospelovartem

To run inference (evaluate the model or save predictions):

```bash
wget https://drive.google.com/file/d/1J2r4nRMAlEAXrPQtQKSFuHdJgo8i6XmX/view?usp=sharing
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
# ASR_HW2
