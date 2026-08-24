# Unified CIFAR Benchmark v1: audit and reporting contract

## Claim boundary

This package runs a **new controlled baseline benchmark**, not a merge of CM
and CORAL paper tables. It is suitable for a paper's common-method comparison
when described with this exact protocol. It is not evidence that a row exactly
matches a number published under a different paper's training budget, sampler,
or evaluator.

## Matrix

| Data cell | DDPM | CBDM | T2H | CM | CORAL | CCUA |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10-LT IF100 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |
| CIFAR-10-LT IF1000 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |
| CIFAR-100-LT IF100 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |

A CIFAR-100-LT-only variant of this same contract (18 tasks, one cell) lives in
`configs/unified_cifar_c100.yaml`; it narrows `fairness_contract.cells` and
changes nothing else.

That is 54 training-from-scratch tasks. The report refuses a cell with fewer
than all three completed seeds.

## Method identity audit

| Displayed row | Vendored implementation | Mechanism toggled |
|---|---|---|
| DDPM | `coral-lt-diffusion` | Base conditional diffusion loss only. |
| CBDM | `coral-lt-diffusion` / pinned CBDM compatibility source | `--cb --tau=1.0`. |
| T2H | official `OC_LT` | `--transfer_x0 --transfer_mode=t2h`. The repository README is titled “T2H” and its paper is *Long-Tailed Diffusion Models With Oriented Calibration*. |
| CM | official `ImbDiff-CM` | released Capacity Manipulation loss/model. |
| CORAL | official `coral-lt-diffusion` | released supervised-contrastive CORAL mechanism. |
| CCUA | official `CCUA-DDPM` (U-Net half of the CCUA repository) | `--ccua_al=1.0 --ccua_ucl=1.0`, with `--nocbdm --notransfer_x0` naming its sibling objectives off. |

The earlier label `oc` is an alias for the T2H source/method, not an
independent baseline. It is intentionally absent from the table so there is no
double-counted row.

Three notes on the CCUA row, because its repository is a superset of the CBDM
and T2H codebases and ships two pipelines:

- Only `CCUA-DDPM/DDPM` is vendored. `CCUA-SiT` is a Diffusion Transformer and
  would break the pinned U-Net backbone this table holds fixed.
- Its `ImbalanceCIFAR` construction is the same one CBDM uses, at
  `rand_number=0`, so the CCUA row trains on a byte-identical long-tailed subset
  to every other row rather than a re-derived one.
- Loss weights are `1.0/1.0`, the only setting upstream publishes for the U-Net
  pipeline. The paper's tuned `α = γ = 0.05` is a SiT/ImageNet-LT latent-space
  result and is not transferable to this pixel-space objective. Batch resampling
  is off, which is what the paper specifies for CIFAR-LT.
  Sampling is driven through `main.py --sample`, not upstream `evaluate.py`,
  because the latter imports CCUA's own FLD/CLIP/DINOv2 metric stack at module
  scope — a stack this benchmark replaces with one shared evaluator for every
  row anyway.

## Locked factors

- Exact public CIFAR-LT exponential splits, `split_seed=0`.
- Three paired training seeds: `0,1,2`.
- LR 0.0002, 5,000-step warmup, dropout 0.1, gradient clip 1.0, diffusion
  `T=1000`, conditional CFG training (10% label dropout), ADA augmentation off.
- 300,000 updates at batch 64 = 19.2M images seen, the same budget CBDM
  (300k×64), CM (300k×64) and CORAL (150k×128) each used. Checkpoints are
  written every 50k purely for crash resume; as upstream, only the newest is
  kept.
- Backbone pinned in the contract and enforced by preflight: ch=128,
  ch_mult=[1,2,2,2], attn=[1], 2 residual blocks, EMA 0.9999. All five upstream
  repos already agree on these, but only by shared flag defaults — a vendored
  source bump would otherwise change the architecture silently.
- Guidance is ω=1.0 for every row, applied through the identical formula
  `eps + ω·(eps_cond − eps_uncond)` present in all five repos. The papers
  instead tune ω per method and dataset (CBDM uses 1.6/0.8/1.0/0.8; CORAL
  tabulates its own), so this table is one point on each method's fidelity /
  diversity trade-off, not each method's best ω. `guidance_scales` sweeps
  extra ω values from the same trained checkpoint when that curve is needed.
- Sampling is ancestral DDPM with T=1000 for every row — the only mode all
  five repos support natively. CM publishes DDIM-50 numbers and T2H DDIM-100,
  so their published tables are not directly comparable to these rows.
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
3. The preflight validates all 54 tasks, no `OC` duplicate, committed vendor
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
