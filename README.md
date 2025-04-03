# PULSE: Personalized Physiological Signal Analysis Framework

## Publication
*A Personalized Physiological Signal Analysis Framework via Unsupervised Domain Adaptation and Self-adaptive Learning*

![image](https://github.com/fdu-harry/PULSE/blob/main/PULSE.jpg)

## Citation
If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{wang2025pulse,
  title={PULSE: A personalized physiological signal analysis framework via unsupervised domain adaptation and self-adaptive learning},
  author={Wang, Yanan and Hu, Shuaicong and Liu, Jian and Wang, Aiguo and Zhou, Guohui and Yang, Cuiwei},
  journal={Expert Systems with Applications},
  pages={127317},
  year={2025},
  publisher={Elsevier}
}

## Description
This repository contains the implementation of PULSE (Personalized Unsupervised domain adaptation via seLf-adaptive lEarning), a framework for personalizing physiological signal analysis through unsupervised domain adaptation and self-adaptive learning.

## Overview

PULSE addresses the challenge of inter-subject variability in physiological signal analysis through three key components:

- **Adaptive Channel Selection and Embedding (ACSE)**: Dynamically selects informative channels through learnable attention mechanisms
- **Embedding-guided Representation Learning (ERL)**: Enhances intra-class feature consistency during general model pre-training  
- **Self-adaptive Pseudo-label Enhancement (SPE)**: Enables effective domain adaptation using minimal unlabeled data

## Repository Structure

├── MODEL_STRUCTURE.py # Model architecture implementation
├── GM_PRETRAINING.py # General model pre-training with ERL
├── PM_FINETUNING.py # Personalized model fine-tuning with SPE
├── ERL_LOSS.py # ERL loss implementation
├── SPE_LOSS.py # SPE loss implementation

## Model Architecture

The model uses a transformer-based architecture with:

Adaptive channel selection module
Position encoding for temporal information
Multi-head self-attention layers
Feed-forward networks
Training Process
General Model Pre-training
Uses ERL loss to enhance feature consistency
5-fold cross validation
Early stopping based on validation loss
Personalized Model Fine-tuning
Uses SPE loss for domain adaptation
K-means clustering for pseudo-label generation
Validation-based early stopping

## Performance

The framework achieves:

2.8%-6.5% improvements in F1 score (91.5%-95.2% → 98.0%)
2.6%-6.4% improvements in accuracy (90.8%-94.6% → 97.2%)

@article{PULSE2025,
  title={PULSE: A Personalized Physiological Signal Analysis Framework via Unsupervised Domain Adaptation and Self-adaptive Learning},
  author={Wang, Yanan and Hu, Shuaicong and Liu, Jian and Wang, Aiguo and Zhou, Guohui and Yang, Cuiwei}
