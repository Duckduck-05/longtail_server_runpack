# Unified CIFAR-LT runpack

This private, standalone package produces one fair, report-ready baseline
table. It does not call another checkout and does not concatenate incompatible
paper tables.

On a CUDA server, run one command for the CIFAR-100-LT-only campaign (the
current scope — CIFAR-10-LT comes later):

```bash
bash scripts/run_server_c100.sh
```

It creates the pinned environment, loads the packaged `.env.local` W&B
settings, downloads CIFAR-100 through `torchvision`, prepares the shared
metric references, resumes safely when rerun, and launches all 15 CIFAR-100-LT
tasks (DDPM, CBDM, T2H, CM, CORAL x seeds 0,1,2; `configs/unified_cifar_c100.yaml`).

GPU packing is automatic: the scheduler reads each GPU's free VRAM and packs
multiple tasks onto one GPU when there's room (starting from a 12 GB/task
estimate that self-corrects once a real task finishes), so the same command
adapts to whatever box it's rented on. Override it when needed:

```bash
bash scripts/run_server_c100.sh --per-gpu 3 --gpus 0,1,2,3   # pin the packing/GPU set
bash scripts/run_server_c100.sh --jobs 8                       # cap total concurrent tasks
```

or set `LTX_TASKS_PER_GPU` / `LTX_GPU_IDS` / `LTX_MAX_CONCURRENT` in
`.env.local`. Dataloader workers per task are divided automatically so packed
tasks don't oversubscribe the host's CPUs.

Once CIFAR-10-LT is back in scope, `bash scripts/run_server.sh` runs the full
45-task `configs/unified_cifar.yaml` campaign (all three cells) instead —
same machine, same environment, no extra setup.

## What runs

**Current scope — `scripts/run_server_c100.sh`:** 15 tasks, one cell.

| Data | Methods | Shared controls |
|---|---|---|
| CIFAR-100-LT IF100 | DDPM, CBDM, T2H, CM, CORAL | 300k updates; batch 64; LR 2e-4; U-Net ch=128 [1,2,2,2] attn[1] 2 blocks; EMA 0.9999; T=1000; 50k exact class-uniform samples |

**Full protocol — `scripts/run_server.sh`:** 45 tasks, all three cells (the
same controls, plus CIFAR-10-LT IF100 and IF1000). This is the complete
locked comparison `configs/unified_cifar.yaml` defines; run it once CIFAR-10-LT
is back in scope.

`OC` is not an extra sixth row: the official `OC_LT` repository calls its
method T2H. Running both names would double-count the same method.

**Why 300k updates.** Counted as images seen, 300k x 64 = 19.2M is exactly the
budget CBDM (300k x 64), CM (300k x 64) and CORAL (150k x 128) each used. A
smaller shared budget would run CBDM and CM below their own papers' design
point while running CORAL above its — undertraining two baselines is a
fairness violation in a way that a uniform surplus is not.

The source paper for each row is indexed in [papers/README.md](papers/README.md)
and fetched by `bash papers/fetch_papers.sh`.

Every completed row uses the same balanced CIFAR reference and reports FID,
KID, IS, F₈, F₁⁄₈, improved-PRD precision, and improved-PRD recall. It also
writes a separate per-class and Many/Medium/Few FID breakdown with its exact
sample counts. The resulting reports are written under
`runs/<campaign name>/report/` — `unified_cifar_c100_v1` for the current
CIFAR-100-LT-only run, `unified_cifar_v1` for the full campaign:

```text
runs/<campaign>/report/table.md
runs/<campaign>/report/tail_breakdown.md
runs/<campaign>/report/per_seed.csv
runs/<campaign>/report/tail_per_seed.csv
runs/<campaign>/report/summary.json
runs/<campaign>/report/results.log       # one self-contained log: fingerprint, vendor/env, per-task status, table, W&B links
runs/<campaign>/report/campaign_run.log  # snapshot of the whole-campaign stdout (bootstrap, GPU packing, launches, failures)
runs/<campaign>/latest.log               # live stdout of the most recent run (symlink, updates every launch)
```

The same per-seed and aggregate tables plus all task losses, samples, system
metrics, resolved commands, and artifacts are uploaded to the configured W&B
project. A missing seed or metric makes the report fail rather than silently
publishing a partial comparison.

Given the `WANDB_API_KEY` in `.env.local`, the runner (via `--wandb`, which
both `run_server_c100.sh` and `run_server.sh` always pass) authenticates once
at bootstrap, flips the project to public-read, and publishes a W&B Report
combining the main table, the tail breakdown, and per-run comparison panels.
The report URL is printed at the end of the run and recorded in
`results.log` and `summary.json`.

Every file listed above is uploaded to that run as the `evaluation-report`
artifact, so someone running this on your behalf can hand over a W&B link
alone — the tables, the per-task status, and the full campaign stdout are all
readable there without shell access. Both the visibility change and the report
are best-effort: if either call fails, the run still completes and prints the
one-time manual UI step instead.

This is a new controlled benchmark, so it must be described as such in a
paper—not as a bit-for-bit reproduction of any individual paper table. Full
method and protocol audit: [UNIFIED_CIFAR_PROTOCOL.md](UNIFIED_CIFAR_PROTOCOL.md).
The CM/CORAL-derived experimental design, seed policy, paper-boundary audit,
and ablation/scaling plan are in
[EXPERIMENT_DESIGN_CM_CORAL.md](EXPERIMENT_DESIGN_CM_CORAL.md).
