# 🔋 Battery Degradation Trajectory Prediction based on Generative Diffusion Model

[![License](https://img.shields.io/github/license/BatICM/SkeletonDiff-BatteryLife)](https://github.com/BatICM/SkeletonDiff-BatteryLife/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository implements the battery degradation trajectory prediction algorithm described in the bachelor thesis: "Battery Degradation Trajectory Prediction Method Based on Generative Diffusion Model".

## Overview

Accurate prediction of lithium-ion battery State of Health (SOH) is crucial for energy systems, but traditional deterministic models struggle to capture the highly non-linear and uncertain nature of battery degradation paths. This project presents a novel approach using a generative diffusion model. Conditioned on early observation data, our method successfully shifts from traditional deterministic point prediction to probabilistic distribution modeling, providing more reliable long-term degradation trajectories.

## 🚀 Key Features

- **Two-Stage Network Architecture**: Combines a deterministic backbone to precisely extract the main degradation trend and a Transformer-based diffusion branch to fit the residual distribution of real trajectories.
- **Physical Constraint Guarantee**: Employs non-negative increment decoding to strictly enforce the monotonic decrease constraint of battery capacity.
- **Data-Driven Clustering & Augmentation**: Extracts 15-dimensional key health features from 124 LFP batteries, utilizing K-Means grouping and data augmentation to adapt to model inputs and alleviate sample imbalance.
- **Enhanced EOL Reliability**: Integrates an EOL auxiliary head, comprehensive loss functions (curve EOL consistency, knee point constraint), DDIM accelerated sampling, and a multi-signal fusion mechanism to significantly improve End of Life (EOL) prediction accuracy.

## 🛠 Method

Our framework consists of four main components:

1. **Data Preprocessing & Feature Extraction**: Extracts statistical and shape-aware features from early cycles, followed by K-Means lifetime clustering (Short, Medium, Long) to build group-specific datasets.
2. **Stage 1: Deterministic Pretraining**: Trains a baseline network to learn the primary deterministic skeleton of the degradation trajectory.
3. **Stage 2: Residual Diffusion Training**: Trains the conditional diffusion model to learn the stochastic variations (residuals) on top of the deterministic skeleton.
4. **Inference & Fusion**: Generates multiple plausible future trajectories and selects the most robust representative curve using a multi-signal EOL fusion strategy.

## 📂 Dataset & Preparation

This project utilizes cycle life data from 124 lithium iron phosphate (LFP) batteries. The original dataset is provided by the following work:

> Severson, K.A., Attia, P.M., Jin, N. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nat Energy* **4**, 383–391 (2019). [https://doi.org/10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)

Due to GitHub's file size limits, the raw data `.pkl` files are not included in this repository. To reproduce our experiments, please prepare the data by following these steps:

1. Download the original dataset associated with the paper mentioned above.
2. Format the data to generate the necessary files (`batch1.pkl`, `batch2.pkl`, `batch3.pkl`).
3. Place these files into the `data/` directory of this project.

The expected directory structure should look like this before running any scripts:
```text
Battery-Degradation-Prediction/
├── data/
│   ├── batch1.pkl
│   ├── batch2.pkl
│   └── batch3.pkl
├── code/
│   ├── 01_data_processing.py
│   └── ...