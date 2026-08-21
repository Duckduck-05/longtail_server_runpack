# Unified CIFAR Benchmark v1: audit and reporting contract

## Claim boundary

This package runs a **new controlled baseline benchmark**, not a merge of CM
and CORAL paper tables. It is suitable for a paper's common-method comparison
when described with this exact protocol. It is not evidence that a row exactly
matches a number published under a different paper's training budget, sampler,
or evaluator.

## Matrix

| Data cell | DDPM | CBDM | T2H | CM | CORAL |
|---|---:|---:|---:|---:|---:|
| CIFAR-10-LT IF100 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |
| CIFAR-10-LT IF1000 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |
| CIFAR-100-LT IF100 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |

That is 45 training-from-scratch tasks. The report refuses a cell with fewer
than all three completed seeds.

## Method identity audit

| Displayed row | Vendored implementation | Mechanism toggled |
|---|---|---|
| DDPM | `coral-lt-diffusion` | Base conditional diffusion loss only. |
| CBDM | `coral-lt-diffusion` / pinned CBDM compatibility source | `--cb --tau=1.0`. |
| T2H | official `OC_LT` | `--transfer_x0 --transfer_mode=t2h`. The repository README is titled “T2H” and its paper is *Long-Tailed Diffusion Models With Oriented Calibration*. |
| CM | official `ImbDiff-CM` | released Capacity Manipulation loss/model. |
| CORAL | official `coral-lt-diffusion` | released supervised-contrastive CORAL mechanism. |

The earlier label `oc` is an alias for the T2H source/method, not an
independent baseline. It is intentionally absent from the table so there is no
double-counted row.

## Locked factors

- Exact public CIFAR-LT exponential splits, `split_seed=0`.
- Three paired training seeds: `0,1,2`.
- 200,000 optimizer updates, batch size 64, LR 0.0002, 5,000-step warmup,
  U-Net base width 128, `ch_mult=[1,2,2,2]`, dropout 0.1, diffusion `T=1000`.
- Conditional CFG training and ancestral DDPM sampling with all 1,000 reverse
  steps; guidance value 1.0.
- 50,000 generated 32×32 RGB images per run, with exact cyclic class-uniform
  condition labels (5,000/class for CIFAR-10; 500/class for CIFAR-100).
- One array-based evaluator for every method: balanced CIFAR-train FID,
  deterministic CM-style KID (the released cubic MMD estimator, fixed subset
  RNG), Inception Score, Inception PRD F₈ and F₁⁄₈, plus VGG16-fc2
  improved-PRD precision/recall (exact k-NN radius, k=3).
- **`Recall` is not comparable to CBDM's published `Recall`.** This table uses
  the Kynkäänniemi et al. improved recall on VGG16-fc2 with k=3, matching
  CORAL. CBDM's §Metrics instead measures Recall on Inception-V3 features with
  K=5. Same name, different estimator: do not place the two side by side
  without saying so. `ImprovedPrecision` is reported here for completeness but
  appears in no baseline paper's main table.
- Each non-DDPM row also carries a bootstrap CI95 on its paired-seed advantage
  over the DDPM row of the same cell, which is how CBDM, T2H, CM and CORAL each
  frame their own gain. Three seeds is too few for mean ± std to answer
  "is this difference real?" on its own.
- A separate `tail_breakdown.md`: per-class FID and CM's class-index
  Many/Medium/Few FID groups. These use the main table's 50k class-uniform
  sample rather than CM's separately sampled 20k/split protocol, so they are
  not labelled as a CM Table-3 reproduction.

The sources use their native method-specific loss code. That is required to
evaluate the actual methods; the protocol locks all controllable data, budget,
architecture-scale, conditional label support, sampler family, sample count,
and metric factors around them.

## Port changes that affect validity

1. CORAL and T2H's public evaluators originally draw conditional labels iid.
   The runpack adds an **opt-in evaluation-only** exact cyclic label schedule;
   training and the reverse diffusion equation are unchanged.
2. CM's sampler now has an evaluation-only direct float32 NCHW/label export,
   so it reaches the shared evaluator without an 8-bit PNG round trip. Its
   sample PNGs remain available for inspection/W&B; the training and reverse
   sampler are unchanged.
3. The preflight validates all 45 tasks, no `OC` duplicate, committed vendor
   revisions, uniform-label patch, evaluator availability, and metric assets
   before launch.

## Report fields

`table.md` is intentionally one table with `Data`, `Method`, seed completion,
FID, KID, IS, F₈, F₁⁄₈, improved-PRD precision, improved-PRD recall, FID rank,
and each metric is `mean ± sample standard deviation` over the three training
seeds. The report deliberately does not average ranks across heterogeneous
metrics. `per_seed.csv`, `tail_per_seed.csv`,
`tail_breakdown.md`, and `summary.json` preserve raw rows, metric definitions,
sample counts, paths, and the complete fairness contract. W&B stores the same
per-seed and aggregate tables as versioned artifacts.

This is only the common benchmark. See
[EXPERIMENT_DESIGN_CM_CORAL.md](EXPERIMENT_DESIGN_CM_CORAL.md) for the strict
separation between it and paper-protocol sensitivity/reproduction tables.
