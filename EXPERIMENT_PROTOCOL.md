# Legacy frozen semantic experiment protocol

> This file describes the older `deadline_full.yaml` semantic-weighting study.
> It is **not** the default CM + CORAL server command and must not be used to
> describe results from `scripts/run_server.sh`.

## 1. Claim under test

A coarse class is a semantic mixture. The proposed intervention keeps class-local content fixed and changes only the inferred relative mass among occupied semantic components. The decisive experiment tests whether that offline semantic signal changes the terminal diffusion distribution.

## 2. Common backbone and compute

For the five decisive arms and simple controls:

- repository: official CORAL codebase, patched only for deterministic seeding and per-example weighted sampling;
- architecture: identical U-Net;
- optimizer, LR, dropout, diffusion schedule, EMA, CFG dropout, number of steps, sampler, guidance search, and evaluation pipeline: identical;
- CIFAR defaults: LR `2e-4`, batch `128`, dropout `0.1`, `T=1000`, `150k` train steps;
- one GPU per run; AMP is fixed campaign-wide;
- three paired model seeds for main conclusions;
- all decisive arms use the exact same frozen dataset manifest and evaluator;
- all decisive arms use `WeightedRandomSampler(replacement=True)`, including LT with uniform weights.

No arm may receive extra training steps after its result is seen.

## 3. Arms

### Decisive gate

| Arm | Training measure | Scientific purpose |
|---|---|---|
| `lt` | uniform weights on the frozen LT manifest, replacement sampling | baseline without sampler-mechanism confound |
| `oracle` | fine-semantic oracle weights | attainable semantic correction reference |
| `predictive` | locked class-local predictive weights | candidate practical estimator |
| `pointfit` | locked class-local point-fit weights | tests whether predictive averaging is necessary |
| `permutation` | matched ESS/spectrum permutation | semantic-null control |

### Simple controls

| Arm | Purpose |
|---|---|
| `class_balanced` | class-count correction only |
| `sqrt_resampling` | softer class-frequency correction control |
| `ada` | limited-data augmentation baseline implemented by the shared codebase |
| `random_group` | grouping without semantic meaning, when a frozen array is available |

### Published baselines

- DDPM, CBDM, CORAL on the common CORAL codebase;
- Oriented Calibration/T2H from its official repository;
- Capacity Manipulation from its official repository.

Official-repository results are reported separately from strict same-backbone comparisons whenever architecture or training implementation differs.

## 4. Evaluation

### Co-primary semantic endpoints

- per-class terminal mode histogram;
- JS divergence to the frozen target semantic distribution;
- rare-mode mass;
- `R_gen` relative to LT and oracle.

### Safety and general generation endpoints

- overall FID, KID, improved precision and recall;
- per-class FID where sample size supports it;
- many/medium/few and worst-quartile summaries;
- coarse-class consistency;
- nearest-neighbor memorization/copy score;
- head-class non-inferiority.

IS is logged for comparability but is not a primary endpoint.

## 5. Statistics

- Pair arms by dataset draw and model seed.
- Bootstrap at the clean-image/class unit, not individual generated pixels.
- Report mean, standard deviation, paired difference, and 95% CI.
- Treat bootstrap resamples as uncertainty estimates, not independent replications.
- Never select the best seed.
- Guidance scale is chosen by a frozen pilot rule shared by all methods, or all pre-registered scales are reported.

## 6. Verdict

### PASS

- `R_gen(predictive) > 0` with 95% CI excluding zero;
- predictive − permutation semantic recovery is positive with CI excluding zero;
- rare-mode mass improves;
- coarse consistency and head fidelity satisfy the configured non-inferiority margins;
- no material memorization increase.

### PARTIAL

Semantic recovery is real but predictive does not beat point-fit, or generation quality has a bounded trade-off. The method claim must shrink accordingly.

### KILL

- offline recovery does not transfer to terminal generation;
- permutation produces comparable gain;
- semantic recovery is obtained through class drift;
- quality or memorization degradation invalidates the correction.

## 7. Anti-slop rules

- Fine labels are evaluation-only for practical arms. The oracle is explicitly labeled as oracle.
- Fixed `K=5` is benchmark knowledge and cannot be advertised as unknown-component discovery.
- NLL, ARI, t-SNE, or cluster separation do not replace terminal semantic-mass evaluation.
- Do not revive rejected donor, cross-class population geometry, timestep gate, or router stories from a positive weighted-sampling result.
- The final question after every table remains: **does this solve long-tail diffusion, or only this controlled semantic-allocation failure axis?**
