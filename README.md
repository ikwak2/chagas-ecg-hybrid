# Chagas ECG Hybrid Model

This repository provides the implementation of a hybrid deep learning framework for automated
Chagas disease detection from 12-lead electrocardiograms (ECGs). The model integrates deep waveform
representations with physiologically interpretable handcrafted ECG features through an
auxiliary-conditioned cross-attention mechanism.

This code accompanies the extended journal manuscript submitted to *Physiological Measurement*
(IOP Publishing) as part of the George B. Moody PhysioNet Challenge 2025 extended paper collection.

---

## Overview

The proposed framework consists of:
- A 1D ResNet backbone for local ECG morphology encoding
- A bidirectional GRU for long-range temporal modeling
- An auxiliary-conditioned single-query cross-attention module that injects
  handcrafted ECG features and demographic information
- A source-aware weighted training objective for handling heterogeneous and weakly labeled datasets

In addition, the repository includes a lightweight handcrafted feature extraction pipeline focused
on QRS and conduction-related descriptors.

---

## Datasets

The experiments are based on data provided by the **George B. Moody PhysioNet Challenge 2025**,
including:
- SaMi-Trop (serologically confirmed Chagas-positive cohort)
- PTB-XL (predominantly Chagas-negative cohort)
- CODE-15% (mixed cohort with self-reported diagnostic labels)

The datasets can be accessed directly through PhysioNet:
```
https://moody-challenge.physionet.org/2025/
```

---

## Handcrafted ECG Features

The handcrafted feature extraction pipeline computes the following physiologically motivated
descriptors from each 12-lead ECG recording:
- Mean QRS duration
- Mean QRS electrical axis (derived from leads I and aVF)
- RSR prime ratio in lead V1
- Wide S ratio in lead I or V6

These features are used as auxiliary inputs to condition the temporal attention mechanism in the deep model.

---

## Installation

The implementation was developed with reference to the official PhysioNet Challenge 2025 Python example code available [here](https://github.com/physionetchallenges/python-example-2025/tree/main?tab=readme-ov-file).

We recommend using a Python virtual environment.

```bash
git clone https://github.com/USERNAME/chagas-ecg-hybrid.git
cd chagas-ecg-hybrid
pip install -r requirements.txt
```

## Training

Training scripts assume that ECG recordings have been preprocessed according to the PhysioNet Challenge guidelines. You can train your model by running

```
    python train_model.py -d training_data -m model
```

where

- `training_data` (input; required) is a folder with the training data files, which must include the labels; and
- `model` (output; required) is a folder for saving your model.

You can run your trained model by running

```
    python run_model.py -d holdout_data -m model -o holdout_outputs
```

where

- `holdout_data` (input; required) is a folder with the holdout data files, which will not necessarily include the labels;
- `model` (input; required) is a folder for loading your model; and
- `holdout_outputs` (output; required) is a folder for saving your model outputs.

The [Challenge website](https://physionetchallenges.org/2025/#data) provides a training database with a description of the contents and structure of the data files.

You can evaluate your model by pulling or downloading the [evaluation code](https://github.com/physionetchallenges/evaluation-2025) and running

```
    python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv
```
where

- `holdout_data`(input; required) is a folder with labels for the holdout data files, which must include the labels;
- `holdout_outputs` (input; required) is a folder containing files with your model's outputs for the data; and
- `scores.csv` (output; optional) is file with a collection of scores for your model.

You can use the provided training set for the `training_data` and `holdout_data` files, but we will use different datasets for the validation and test sets, and we will not provide the labels to your code.

## Citation

If you use this code or find it helpful in your research, please cite the corresponding paper:

```
@article{Im_2026,
doi = {10.1088/1361-6579/ae8b72},
url = {https://doi.org/10.1088/1361-6579/ae8b72},
year = {2026},
month = {jul},
publisher = {IOP Publishing},
volume = {47},
number = {7},
pages = {075014},
author = {Im, Hyuno and Lee, Nahyun and Kang, Taeyoung and Kim, Taehwan and Kim, Donggun and Lee, Dongjae and Oh, Seungsang and Gong, Wuming and Kwak, Il-Youp},
title = {Auxiliary-conditioned cross-attention with physiologically interpretable features for chagas disease detection from 12-lead ECGs},
journal = {Physiological Measurement},
abstract = {Objective. Chagas disease remains a major public-health concern in endemic regions, and chronic cardiac involvement often manifests as conduction abnormalities detectable on standard 12-lead electrocardiograms (ECGs). Reliable automated screening remains challenging because of dataset heterogeneity and label uncertainty, particularly when combining strongly labeled cohorts with large weakly labeled repositories. Approach. We propose a hybrid architecture that integrates a 1D ResNet encoder for local ECG morphology, a bidirectional GRU for long-range temporal context, and handcrafted physiological features and demographics through an auxiliary-conditioned cross-attention module. The auxiliary vector, comprising age, sex, and QRS/conduction descriptors, is projected into a query token that selectively attends to deep sequential embeddings for feature-aware temporal aggregation. To exploit heterogeneous sources while reflecting source reliability, we further adopt a source-aware weighted binary cross-entropy objective. Main results. As team CAUETUMN in the PhysioNet/Computing in Cardiology Challenge 2025, the framework achieved a score of 0.347 on the organizer-held REDS-II leaderboard-validation set during the official phase and 0.218 on the final hidden test set, ranking 17th among 41 eligible teams. Significance. These results suggest that conditioning detection on interpretable QRS and conduction descriptors supports a transparent and physiologically informed screening framework, while highlighting the difficulty of generalizing across heterogeneous cohorts.}
}
```
