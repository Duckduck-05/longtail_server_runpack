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

GPU packing is automatic: the scheduler reads each GPU's free VRAM and packs
multiple tasks onto one GPU when there's room (starting from a 12 GB/task
estimate that self-corrects once a real task finishes), so the same command
adapts to whatever box it's rented on. Override it when needed:

```bash
bash scripts/run_server.sh --per-gpu 3 --gpus 0,1,2,3   # pin the packing/GPU set
bash scripts/run_server.sh --jobs 8                       # cap total concurrent tasks
```

or set `LTX_TASKS_PER_GPU` / `LTX_GPU_IDS` / `LTX_MAX_CONCURRENT` in
`.env.local`. Dataloader workers per task are divided automatically so packed
tasks don't oversubscribe the host's CPUs.

## What runs

45 tasks: five methods × three datasets × seeds `0,1,2`.

| Data | Methods | Shared controls |
|---|---|---|
| CIFAR-10-LT IF100 | DDPM, CBDM, T2H, CM, CORAL | 200k updates; batch 64; LR 2e-4; U-Net base width 128; T=1000; 50k exact class-uniform samples |
| CIFAR-10-LT IF1000 | DDPM, CBDM, T2H, CM, CORAL | same |
| CIFAR-100-LT IF100 | DDPM, CBDM, T2H, CM, CORAL | same |

`OC` is not an extra sixth row: the official `OC_LT` repository calls its
method T2H. Running both names would double-count the same method.

The source paper for each row is indexed in [papers/README.md](papers/README.md)
and fetched by `bash papers/fetch_papers.sh`.

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
runs/unified_cifar_v1/report/results.log   # one self-contained log: fingerprint, vendor/env, per-task status, table, W&B links
runs/unified_cifar_v1/latest.log           # full stdout of the most recent run (symlink, updates every launch)
```

The same per-seed and aggregate tables plus all task losses, samples, system
metrics, resolved commands, and artifacts are uploaded to the configured W&B
project. A missing seed or metric makes the report fail rather than silently
publishing a partial comparison.

Given the `WANDB_API_KEY` in `.env.local`, the runner (via `--wandb`, which
`scripts/run_server.sh` always passes) authenticates once at bootstrap, flips
the project to public-read, and publishes a W&B Report combining the main
table, the tail breakdown, and per-run comparison panels. The report URL is
printed at the end of the run and recorded in `results.log` and
`summary.json`. Both the visibility change and the report are best-effort:
if either call fails, the run still completes and prints the one-time manual
UI step instead.

This is a new controlled benchmark, so it must be described as such in a
paper—not as a bit-for-bit reproduction of any individual paper table. Full
method and protocol audit: [UNIFIED_CIFAR_PROTOCOL.md](UNIFIED_CIFAR_PROTOCOL.md).
The CM/CORAL-derived experimental design, seed policy, paper-boundary audit,
and ablation/scaling plan are in
[EXPERIMENT_DESIGN_CM_CORAL.md](EXPERIMENT_DESIGN_CM_CORAL.md).
