# Source papers

The active CCUA-backed table and its paper baselines. Fetch or refresh the
reference papers with:

```bash
bash papers/fetch_papers.sh
```

The PDFs themselves are **not committed** (`papers/.gitignore`) — they are
third-party copyrighted works. The script re-fetches them on demand.

| Method row | Paper | Venue | Source | File |
|---|---|---|---|---|
| `ddpm` | Denoising Diffusion Probabilistic Models | NeurIPS 2020 | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | `ddpm-neurips2020-ho.pdf` |
| `cbdm` | Class-Balancing Diffusion Models | CVPR 2023 | [arXiv:2305.00562](https://arxiv.org/abs/2305.00562) | `cbdm-cvpr2023-qin.pdf` |
| `t2h` | Long-tailed Diffusion Models with Oriented Calibration | ICLR 2024 | [OpenReview NW2s5XXwXU](https://openreview.net/forum?id=NW2s5XXwXU) | `t2h-iclr2024-zhang.pdf` |
| `cm` | Improving Diffusion Models for Class-imbalanced Training Data via Capacity Manipulation | ICLR 2026 (Oral) | [OpenReview wSGle6ag5I](https://openreview.net/forum?id=wSGle6ag5I) | `cm-iclr2026-hong.pdf` |
| `coral` | CORAL: Disentangling Latent Representations in Long-Tailed Diffusion | NeurIPS 2025 | [arXiv:2506.15933](https://arxiv.org/abs/2506.15933) | `coral-neurips2025-rodriguez.pdf` |
| `ccua` | Contrastive Conditional–Unconditional Alignment for Long-tailed Diffusion Model | arXiv preprint (v3, Jun 2026) | [arXiv:2507.09052](https://arxiv.org/abs/2507.09052) | `ccua-arxiv2507.09052-chen.pdf` |

## Notes on provenance

- **`ddpm` is not a separate objective implementation.** It is the CCUA-DDPM
  host run with the sibling long-tail losses disabled, i.e. a plain conditional
  DDPM baseline. The DDPM paper is included because it defines the backbone,
  sampler, and noise schedule.
- **T2H, CM, and CORAL are archived comparison sources.** Their vendored code
  and old runpack launchers live under `archive/legacy_source_ccua_20260901/`;
  they are not active rows in the CCUA campaign.
- **T2H and CM are not on arXiv**, and Semantic Scholar lists no open-access
  mirror. OpenReview is the source of record for both, and it serves PDFs behind
  a bot challenge, so `fetch_papers.sh` cannot download them non-interactively.
  Point it at existing local copies instead:

  ```bash
  LTX_PAPER_T2H_SRC=/path/to/t2h.pdf \
  LTX_PAPER_CM_SRC=/path/to/cm.pdf \
    bash papers/fetch_papers.sh
  ```

  Otherwise the script prints the forum URL to open in a browser.
- **CORAL** has an archived vendored copy under
  `archive/legacy_source_ccua_20260901/third_party/`; the copy here is the
  arXiv version.
- **`ccua` is the U-Net half of its repository.** Upstream ships `CCUA-DDPM`
  (U-Net) and `CCUA-SiT` (Diffusion Transformer); only the former shares this
  table's backbone, so only `CCUA-DDPM/DDPM` is vendored. Two consequences for
  reading the paper next to this table: the paper's tuned loss weights
  α = γ = 0.05 belong to the SiT/ImageNet-LT pipeline, while the U-Net pipeline's
  own script uses 1.0/1.0 (what this table runs); and the paper's batch-resample
  strategy is applied to ImageNet-LT/TinyImageNet-LT but explicitly **not** to
  CIFAR-LT, so it is off here too. CCUA's own FLD/CLIP/DINOv2 metric stack is not
  vendored — every row is scored by this repo's shared evaluator instead.
  Its Table 8 reports CIFAR-100-LT with DDPM 1000 steps, the same sampler family
  as this contract, so it is the paper number most comparable to the `ccua` row.
- **IGD-ML** is retained only in the cleanup archive and is deliberately not
  part of the active table.

## BibTeX

```bibtex
@inproceedings{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}

@inproceedings{qin2023class,
  title={Class-Balancing Diffusion Models},
  author={Qin, Yiming and Zheng, Huangjie and Yao, Jiangchao and Zhou, Mingyuan and Zhang, Ya},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

@inproceedings{zhang2024long,
  title={Long-tailed Diffusion Models with Oriented Calibration},
  author={Zhang, Tianjiao and Zheng, Huangjie and Yao, Jiangchao and Wang, Xiangfeng and Zhou, Mingyuan and Zhang, Ya and Wang, Yanfeng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}

@inproceedings{hong2026improving,
  title={Improving Diffusion Models for Class-imbalanced Training Data via Capacity Manipulation},
  author={Hong, Feng and Yao, Jiangchao and others},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}

@inproceedings{rodriguez2025coral,
  title={CORAL: Disentangling Latent Representations in Long-Tailed Diffusion},
  author={Rodriguez, Esther and Welfert, Monica and McDowell, Samuel and Stromberg, Nathan and Camarena, Julian Antolin and Sankar, Lalitha},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}

@article{chen2026contrastive,
  title={Contrastive Conditional--Unconditional Alignment for Long-tailed Diffusion Model},
  author={Chen, Fang and Villa, Alex and Liang, Gongbo and Fuxin, Li and Lu, Xiaoyi and Tang, Meng},
  journal={arXiv preprint arXiv:2507.09052},
  year={2026}
}
```
