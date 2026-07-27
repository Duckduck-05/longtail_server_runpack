# Unified CIFAR-LT runpack

This private, standalone package produces one fair, report-ready baseline
table. It does not call another checkout and does not concatenate incompatible
paper tables.

On a CUDA server, run one command:

```bash
bash scripts/run_server.sh
```

It creates the pinned environment, loads the packaged `.env.local` W&B
settings, downloads CIFAR-10/100 through `torchvision`, prepares the shared
metric references, resumes safely when rerun, and launches the full campaign.

## What runs

45 tasks: five methods × three datasets × seeds `0,1,2`.

| Data | Methods | Shared controls |
|---|---|---|
| CIFAR-10-LT IF100 | DDPM, CBDM, T2H, CM, CORAL | 200k updates; batch 64; LR 2e-4; U-Net base width 128; T=1000; 50k exact class-uniform samples |
| CIFAR-10-LT IF1000 | DDPM, CBDM, T2H, CM, CORAL | same |
| CIFAR-100-LT IF100 | DDPM, CBDM, T2H, CM, CORAL | same |

`OC` is not an extra sixth row: the official `OC_LT` repository calls its
method T2H. Running both names would double-count the same method.

Every completed row uses the same balanced CIFAR reference and reports FID,
KID, IS, F₈, F₁⁄₈, improved-PRD precision, and improved-PRD recall. It also
writes a separate per-class and Many/Medium/Few FID breakdown with its exact
sample counts. The resulting reports are written to:

```text
runs/unified_cifar_v1/report/table.md
runs/unified_cifar_v1/report/tail_breakdown.md
runs/unified_cifar_v1/report/per_seed.csv
runs/unified_cifar_v1/report/tail_per_seed.csv
runs/unified_cifar_v1/report/summary.json
```

The same per-seed and aggregate tables plus all task losses, samples, system
metrics, resolved commands, and artifacts are uploaded to the configured W&B
project. A missing seed or metric makes the report fail rather than silently
publishing a partial comparison.

This is a new controlled benchmark, so it must be described as such in a
paper—not as a bit-for-bit reproduction of any individual paper table. Full
method and protocol audit: [UNIFIED_CIFAR_PROTOCOL.md](UNIFIED_CIFAR_PROTOCOL.md).
The CM/CORAL-derived experimental design, seed policy, paper-boundary audit,
and ablation/scaling plan are in
[EXPERIMENT_DESIGN_CM_CORAL.md](EXPERIMENT_DESIGN_CM_CORAL.md).
