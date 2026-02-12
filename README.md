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

TBD
