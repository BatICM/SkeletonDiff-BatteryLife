# 🔋 Battery Degradation Trajectory Prediction based on Generative Diffusion Model

[![License](https://img.shields.io/github/license/BatICM/SkeletonDiff-BatteryLife)](https://github.com/BatICM/SkeletonDiff-BatteryLife/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository implements the battery degradation trajectory prediction algorithm.

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

## 📊 Results

Tested on a dataset of 124 lithium iron phosphate (LFP) batteries, our generative approach outperforms traditional baselines (such as LSTM and Transformer) in global average indicators including RMSE, MAE, and EOL prediction errors, especially demonstrating superior stability in long-term predictions.

## Citation

[To be added]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

[To be added]

## Contact

[To be added]