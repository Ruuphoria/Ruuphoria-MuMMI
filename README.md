# Multi-Modal Mutual Information (MuMMI) Training for Robust Self-Supervised Deep Reinforcement Learning

This refined repository contains the code for the paper [Multi-Modal Mutual Information (MuMMI) Training for Robust Self-Supervised Deep Reinforcement Learning](https://arxiv.org/abs/2107.02339) (ICRA-21).

## Introduction

This work focuses on learning useful and robust deep world models using multiple, possibly unreliable, sensors. We find that current methods do not sufficiently encourage a shared representation between modalities; this can cause poor performance on downstream tasks and over-reliance on specific sensors. This version of the codebase brings improvements on these fronts and aims to further enhance the results produced by the method.

## Environment Setup

The code is tested on Ubuntu 16.04, Python 3.7 and CUDA 10.2. Please download the relevant Python packages by running the mentioned commands.

## Usage

To run this codebase or baselines on mujoco, follow the steps outlined in the Usage Instructions section.

### BibTeX

To cite this work, please use the following citation:

```
@inproceedings{Chen2021MuMMI,
title={Multi-Modal Mutual Information (MuMMI) Training for Robust Self-Supervised Deep Reinforcement Learning},
author={Kaiqi Chen and Yong Lee and Harold Soh},
year={2021},
booktitle={IEEE International Conference on Robotics and Automation (ICRA)}}
```

### Acknowledgement

This repo contains code that's based on the following repos: [Yusufma03/CVRL](https://github.com/Yusufma03/CVRL). Modified and refined by Ruuphoria.

### References
**[Ma et al., 2020]** Xiao Ma, Siwei Chen, David Hsu, Wee Sun Lee: Contrastive Variational Model-Based Reinforcement Learning for Complex Observations. In CoRL, 2020.