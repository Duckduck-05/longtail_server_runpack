# Improving Diffusion Models for Class-Imbalanced Training Data via Capacity Manipulation

## Install

```bash
conda create -n imbdiff-cm python=3.10
conda activate imbdiff-cm
pip install -r requirements.txt
```

Put the FID Inception checkpoint at:

```text
stats/pt_inception-2015-12-05-6726825d.pth
```

## Train

```bash
bash scripts/train_cm.sh
```

## Sample

```bash
bash scripts/sample_cm.sh
```

## Test

```bash
bash scripts/extract_real_features.sh
bash scripts/extract_cm_features.sh
bash scripts/metrics_cm.sh
```

## Classification Experiments

See [classification/README.md](classification/README.md) for the classification experiment.

## Citation

```bibtex
@inproceedings{hong2026cm,
  title={Improving Diffusion Models for Class-Imbalanced Training Data via Capacity Manipulation},
  author={Hong, Feng and Yao, Jiangchao and Shen, Yifei and Li, Dongsheng and Zhang, Ya and Wang, Yanfeng},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
