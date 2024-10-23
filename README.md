# TrackFormer: Particle Trackfitting with Transformer 

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0%2B-blue.svg)](https://www.pytorchlightning.ai/)

## Overview

TrackFormer is a unique solution to particle trajectory reconstruction that uses transformer-inspired design. Using Transformers' self-attention mechanism, this model performs track fitting, resulting in better accuracy and efficiency.

## Features

- **Transformer-based Architecture**: For fast efficient and accurate particle track fitting.
- **Built with Lightning Integration**:
- **Modular Design**: 
- **Logging and CLI integration**:

## Getting Started

1. **Clone the repository**:
2. download datasets to specific directory in Data
3. train, test split dataset  with `Data/Acts/datamaker.sh /path/to/dataset 80 10 10`
4. example usage `python main.py fit --config configs/tformer.yaml  --data ActsDataModule  --config configs/trainer.yaml`

## Important links

- [slides](https://docs.google.com/presentation/d/1YvFFSKoVI4W0tpCQEvG8J3SybidqZtn08r3DU9UGw_E/edit?usp=sharing)

- [report](https://drive.google.com/file/d/1MHhqyw6Q5gMjyYJo2VmZTj2Ry4HQGAX9/view?usp=sharing)
