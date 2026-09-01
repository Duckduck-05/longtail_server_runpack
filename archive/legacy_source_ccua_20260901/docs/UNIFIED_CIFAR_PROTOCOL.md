# Unified CIFAR Benchmark v1: audit and reporting contract

> **Legacy protocol.** This document describes the older T2H-unified campaign
> kept only so active server jobs can finish and be audited. It is not the
> current main baseline. The current main table is the CCUA-DDPM objective
> matrix documented in `README.md` and
> `configs/native_cifar100_if100.yaml`.

## Claim boundary

This package runs a **new controlled common-host benchmark**, not a merge of CM
and CORAL paper tables. It is suitable for a paper's common-method comparison
when described with this exact protocol. Every active row goes through the
T2H-unified host revision `t2h-unified-common-v2`; the native repositories are
read-only provenance inputs and separate reproduction references. It is not
evidence that a row exactly matches a number published under a different
paper's training budget, sampler, or evaluator.

## Matrix

| Data cell | DDPM | CBDM | T2H | CM | CORAL | CCUA |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10-LT IF100 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |
| CIFAR-10-LT IF1000 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |
| CIFAR-100-LT IF100 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 | seeds 0,1,2 |

A CIFAR-100-LT-only variant of this same contract (27 tasks, one cell, including
the IP-SVT method and its two attribution arms) lives in
`configs/unified_cifar_c100.yaml`; it narrows `fairness_contract.cells` and
changes nothing else.

That is 54 training-from-scratch tasks. The report refuses a cell with fewer
than all three completed seeds.

## Method identity audit

| Displayed row | Common host path | Mechanism toggled |
|---|---|---|
| DDPM | `T2H-unified/unified_main.py` → `UnifiedObjective("ddpm")` | Base conditional diffusion loss only. |
| CBDM | same host → `UnifiedObjective("cbdm")` | Balanced-label two-sided prediction penalty. |
| T2H | same host → `UnifiedObjective("t2h")` | Transferred DSM target, restricted to the T2H direction. |
| CM | same host → CM hook + `UNet_CM` LoRA branch | Released Capacity Manipulation loss/model branch. |
| CORAL | same host → CORAL hook + 128-D bottleneck projection head | Released timestep-scaled supervised-contrastive mechanism. The head is discarded at inference. |
| CCUA | same host → `UnifiedObjective("ccua")` | Unconditional contrastive loss plus conditional–unconditional alignment. |

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
  The native `main.py`/`evaluate.py` path is retained only for source audit.
  The active CCUA row uses `T2H-unified/unified_main.py --sample` and the same
  shared evaluator as every other row; this avoids CCUA's private
  FLD/CLIP/DINOv2 metric stack and prevents a native sampler from entering the
  common table.

## Locked factors

- Exact public CIFAR-LT exponential splits, `split_seed=0`.
- Three paired training seeds: `0,1,2`.
- LR 0.0002, 5,000-step warmup, dropout 0.1, gradient clip 1.0, diffusion
  `T=1000`, conditional CFG training (10% label dropout), ADA augmentation off.
- 300,000 updates at batch 64 = 19.2M images seen. Checkpoints are written
  every 50k under the names `ckpt_unified_v2_<step>.pt` purely for crash
  resume; the host writes them atomically and embeds the full method/data/model
  provenance. Legacy `ckpt_<step>.pt` files are never auto-reused.
- Backbone is pinned in the contract and enforced by preflight: ch=128,
  ch_mult=[1,2,2,2], attn=[1], 2 residual blocks, EMA 0.9999. The one host uses
  the same U-Net implementation for every row; only the method-required CM LoRA
  branch and CORAL projection head add their declared method-specific modules.
- Guidance is ω=1.5 for every row, applied through the same host sampler and
  formula `eps + ω·(eps_cond − eps_uncond)`. The papers tune ω per method and
  dataset, but those paper-specific values are intentionally not mixed into this
  common table.
- Sampling is deterministic DDIM-100 (T=1000, skip=10) for every row. This is
  the locked common-host setting; native paper samplers remain available only in
  the separate reproduction campaigns.
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

The native trees are used to audit equations and hyperparameters, then the
method-specific objective is ported into the one T2H host. This is required for
a valid same-code comparison: all rows share the data construction, corruption,
optimizer/EMA, checkpoint schema, reverse sampler, sample export, and metric
evaluator. Native launchers must not be mixed into this table.

## Port changes that affect validity

1. CORAL and T2H's public evaluators originally draw conditional labels iid.
   The runpack adds an **opt-in evaluation-only** exact cyclic label schedule;
   training and the reverse diffusion equation are unchanged.
2. The common host writes namespaced float32 NCHW/label arrays plus a sampler
   provenance sidecar. The shared evaluator validates that sidecar against the
   requested checkpoint, seed, objective, DDIM settings, and class schedule
   before doing the expensive feature pass.
3. The preflight validates all 54 tasks, no `OC` duplicate, the pinned host
   revision, uniform-label schedule, evaluator availability, and metric assets
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
