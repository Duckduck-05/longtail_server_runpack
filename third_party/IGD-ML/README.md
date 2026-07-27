
# Mutual Learning for Diffusion Models

This repository contains the PyTorch implementation for *Principled Long-Tailed Generative Modelling via Diffusion Models* by Pranoy Das, Kexin Fu, Abolfazl Hashemi, and Vijay Gupta.

## Overview

Deep generative models, particularly diffusion models, have achieved remarkable success but face significant challenges when trained on real-world, long-tailed datasets—where a few "head" classes dominate, and many "tail" classes are underrepresented. This paper develops a theoretical framework for long-tailed learning via diffusion models through the lens of deep mutual learning. We introduce a novel regularized training objective that combines the standard diffusion loss with a mutual learning term, enabling balanced performance across all class labels, including the underrepresented tails.

Our approach formulates the proposed regularized objective as a multi-player game, with Nash equilibrium as the solution concept. We derive a non-asymptotic first-order convergence result for the Individual Gradient Descent (IGD) algorithm to find the Nash equilibrium. The Nash gap of the score network obtained from the algorithm is upper bounded by $\mathcal{O}(\frac{1}{\sqrt{T_{\text{train}}}}+\beta)$, where $\beta$ is the regularizing parameter and $T_{\text{train}}$ is the number of training iterations. Furthermore, we theoretically establish hyperparameters for the training and sampling algorithms that ensure conditional score networks (under our model) with a worst-case sampling error of $\mathcal{O}(\epsilon+1), \forall \epsilon>0$ across all class labels. Our results offer insights and guarantees for training diffusion models on imbalanced, long-tailed data, with implications for fairness, privacy, and generalization in real-world generative modeling scenarios.

## About This Repository
Let me know if you need further adjustments or specific additions!

This repository is based on [CBDM-pytorch](https://github.com/qym7/CBDM-pytorch). It currently supports training on the CIFAR10-LT dataset under the following three mechanisms:

1. Class-Balancing Diffusion Model (CBDM) training
2. Individual DDPM models trained on each class ($\beta/\tau=0$)
3. Individual Gradient Descent (IGD) with Mutual Learning Loss function

## Running the Experiment

In the code, the regularizer $\beta$ is represented by $\tau$, as used in the CBDM literature. The repository provides scripts for training and evaluating on the CIFAR10-LT dataset. To run the code, update the `root` argument to the path where the dataset is downloaded.

### Files Used in Evaluation

The [features for CIFAR-10 and CIFAR-100](https://drive.google.com/drive/folders/1Y89vu9DGiQsHl8YvwMrr_7UT4p4Pg_wV?usp=sharing) used in precision, recall, and F-beta metrics are available. Place them in the `stats` folder to run the evaluation scripts. Note that these metrics are only evaluated if the number of samples is 50,000; otherwise, they return 0.

## Training a Model

### 1. Class-Balancing Diffusion Model (CBDM)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --train \
    --flagfile ./config/cifar100.txt \
    --parallel \
    --logdir $LOGDIR_CBDM \
    --total_steps 60001 \
    --conditional \
    --data_type cifar100lt \
    --imb_factor 0.01 \
    --img_size 32 \
    --batch_size 48 \
    --save_step 10000 \
    --sample_step 5000 \
    --cb \
    --tau 1.0
```

### 2. Individual DDPM Models for Each Class ($\beta=0$)

```bash
CUDA_VISIBLE_DEVICES=0 python main_igd.py --train \
    --flagfile ./config/cifar10.txt \
    --parallel \
    --logdir $LOGDIR_no_ml \
    --total_steps 60001 \
    --conditional \
    --data_type cifar10lt \
    --imb_factor 0.01 \
    --img_size 32 \
    --batch_size 48 \
    --save_step 10000 \
    --sample_step 5000 \
    --cb \
    --tau 0 \
    --igd \
    --eta 2e-4 \
    --lr 2e-4
```

**Note**: `lr` and `eta` are the learning rate and must have the same value. The training includes 60,000 steps of gradient updates, with model parameters saved every 10,000 steps. Images are sampled every 5,000 steps to monitor progress.

### 3. Individual Gradient Descent (IGD) with Mutual Learning Loss

```bash
CUDA_VISIBLE_DEVICES=0 python main_igd.py --train \
    --flagfile ./config/cifar10.txt \
    --parallel \
    --logdir $LOGDIR_ml \
    --total_steps 60001 \
    --conditional \
    --data_type cifar10lt \
    --imb_factor 0.01 \
    --img_size 32 \
    --batch_size 48 \
    --save_step 10000 \
    --sample_step 5000 \
    --cb \
    --tau 0.1 \
    --igd \
    --eta 2e-4 \
    --lr 2e-4
```

**Note**: `lr` and `eta` are the learning rate and must have the same value. The training includes 60,000 steps of gradient updates, with model parameters saved every 10,000 steps. Images are sampled every 5,000 steps to monitor progress.

## Evaluating a Model

Sample images and evaluate the three models above.

### 1. Evaluation Script for CBDM

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --flagfile $LOGDIR_CBDM/flagfile.txt \
    --logdir $LOGDIR_CBDM \
    --data_type cifar10lt \
    --fid_cache ./stats/cifar10.train.npz \
    --ckpt_step 60000 \
    --num_images 15000 \
    --batch_size 64 \
    --notrain \
    --eval \
    --sample_method cfg \
    --omega 1.6 \
    --prd
```

### 2. Evaluation Script for Individual DDPM (No Mutual Learning)

```bash
CUDA_VISIBLE_DEVICES=1 python main_igd.py \
    --flagfile $LOGDIR_no_ml/flagfile.txt \
    --logdir $LOGDIR_no_ml \
    --fid_cache ./stats/cifar10.train.npz \
    --ckpt_step 60000 \
    --num_images 15000 \
    --batch_size 64 \
    --notrain \
    --eval \
    --sample_method cfg \
    --omega 1.6 \
    --prd
```

### 3. Evaluation Script for IGD with Mutual Learning

```bash
CUDA_VISIBLE_DEVICES=1 python main_igd.py \
    --flagfile $LOGDIR_ml/flagfile.txt \
    --logdir $LOGDIR_ml \
    --fid_cache ./stats/cifar10.train.npz \
    --ckpt_step 60000 \
    --num_images 15000 \
    --batch_size 64 \
    --notrain \
    --eval \
    --sample_method cfg \
    --omega 1.6 \
    --prd
```

## References

If you find the code useful for your research, please consider citing:



## Acknowledgements

This implementation is based on or inspired by:

- [pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm)
- [CBDM-pytorch](https://github.com/qym7/CBDM-pytorch)

Let me know if you need further adjustments or specific additions!
