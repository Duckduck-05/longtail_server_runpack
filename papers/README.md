# Source papers

The five methods in the unified CIFAR-LT table, one paper each. Fetch or
refresh them with:

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

## Notes on provenance

- **`ddpm` is not a separate paper's method.** That row is the CBDM repository
  run *without* `--cb`, i.e. a plain conditional DDPM baseline. The DDPM paper
  is included because it defines the backbone, sampler, and noise schedule that
  every other row builds on.
- **`t2h` is the `OC_LT` repository.** Upstream names the method T2H; `OC` is
  only the repo name. It appears once in the table, not twice — see the note in
  the top-level `README.md`.
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
- **CORAL** is also vendored inside the source tree at
  `third_party/coral-lt-diffusion/CORAL-NeurIPS2025-Rodriguezetal.pdf`; the copy
  here is the arXiv version.
- **IGD-ML** (`third_party/IGD-ML`, "Principled Long-Tailed Generative Modelling
  via Diffusion Models") is vendored but deliberately **not** part of the table,
  so its paper is not fetched here.

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
```
