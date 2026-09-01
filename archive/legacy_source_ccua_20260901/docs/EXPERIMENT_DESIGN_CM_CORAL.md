# Experimental design anchored to CM and CORAL

## Non-negotiable interpretation of the two papers

The papers are useful sources of cells, mechanisms, and metrics.  They are
**not** one compatible experiment table:

| Issue | CM (ICLR 2026) | CORAL (NeurIPS 2025) | Design consequence |
|---|---|---|---|
| CIFAR cells | CIFAR-10/100, IR 50 and 100 | CIFAR-10 ρ=.01/.001, CIFAR-100 ρ=.01 | Shared cells are CIFAR-10-LT IF100 and CIFAR-100-LT IF100. CIFAR-10 IF1000 is a CORAL stress cell, not a CM-table cell. |
| Published baselines | DDPM, ADA, RS, Focal, CBDM, OC, CM | DDPM, CBDM, T2H, CORAL | The common comparison contains DDPM, CBDM, T2H/OC, CM, CORAL once each. ADA/RS/Focal remain CM-specific additions, never silently omitted from a claimed CM Table-2 reproduction. |
| Training recipe | Paper text says batch 64, 300k steps, 50-step DDIM | CIFAR core: batch 128; 150k steps except T2H 200k | A one-budget table is a new controlled benchmark, not a literal reproduction of either paper. |
| Metrics | FID, KID, improved recall, IS; CM also reports Many/Medium/Few FID | FID, IS, PRD F₈/F₁⁄₈, improved precision/recall | The main table records their union, with explicitly named recall definitions. |
| Large data | ImageNet-LT 32/64, iNaturalist 32/64; CelebA-HQ 64 | ImageNet-LT 64, CelebA-5 64 | CelebA-HQ and CelebA-5 are different datasets; do not put them in one row. |

Neither supplied paper excerpt justifies saying “every published result used
three seeds.” CM Appendix G reports Mean ± Std but does not state the number
of seeds in the supplied text. CORAL Tables 1–2 give point estimates and no
seed count or standard deviation. This runpack therefore labels its own
three-seed reruns as **new replication extensions**, not as an assertion about
the authors' seed policy.

## The reportable result set

### Tier A — confirmatory common benchmark (main paper table)

This is the default `bash scripts/run_server.sh` campaign.  It has 45
training-from-scratch tasks: five methods × three cells × seeds 0, 1, 2.

| Cell | DDPM | CBDM | T2H (OC) | CM | CORAL | Why it is included |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10-LT IF100 | 3 | 3 | 3 | 3 | 3 | Direct CM/CORAL overlap |
| CIFAR-100-LT IF100 | 3 | 3 | 3 | 3 | 3 | Direct CM/CORAL overlap; main tail-class cell |
| CIFAR-10-LT IF1000 | 3 | 3 | 3 | 3 | 3 | CORAL severe-imbalance stress test |

The locked common controls are: public exponential split with `split_seed=0`,
200k updates, batch 64, LR 2e-4, 5k warm-up, dropout .1, T=1000, class-
conditional CFG, one ancestral 1,000-step sampler, guidance 1.0, and 50k
exact class-uniform samples.  The implementations retain their released
method losses; changing a method into another source's loss would no longer
evaluate that method.

The default result must be called **“a three-seed, source-native controlled
benchmark”**. It must not be described as CM Table 2, CORAL Table 1, or an
exact comparison to their printed absolute FIDs.

### Tier B — paper-protocol sensitivity tables (supplement)

These are separate campaigns and separate tables:

1. `configs/coral2025_cifar.yaml`: the three runnable CIFAR cells from CORAL,
   with CORAL's cell-specific guidance and native 150k/200k budgets. It is a
   three-seed extension of the supplied CORAL recipe. CelebA-5 and ImageNet-LT
   are deliberately absent until their exact public split/loader is packaged.
2. `configs/cm_imagenet_lt.yaml`: a released-code CM source-port sensitivity
   campaign. It is **not** a paper-faithful CM table: the vendored CM CIFAR
   configurations use 200k updates for CIFAR-10 and 300k for CIFAR-100,
   whereas the supplied CM implementation-details prose says 300k. This
   conflict is recorded rather than resolved by guesswork.

Before claiming a CM Table-2 reproduction, port and lock the exact ADA,
resampling, and focal implementations/hyperparameters used by CM. They are
not present in the current CM vendor tree, so inventing substitutes would make
the table less credible, not more complete.

### Tier C — ablations and scale (supplement, not mixed with Tier A)

| Question | Locked cell | Rows | Replication policy |
|---|---|---|---|
| Does CM's capacity allocation matter? | CIFAR-100-LT IF100 | CM, CM θg-only, CM without LCon, CM without LDiv | three seeds per row |
| Does CORAL's bottleneck contrastive loss matter? | CIFAR-10-LT IF100 and CIFAR-100-LT IF100 | CORAL, no SupCon, frozen selected τSC/τr/w variants | screen on seed 0 only, lock once, then three seeds for every reported ablation row |
| Does the result scale? | ImageNet-LT 64×64 | DDPM, CBDM, T2H, CM, CORAL | one seed is labelled exploratory; three seeds are required before a headline claim |

No seed-0 screening number belongs in a main result table. No “best seed” is
ever selected. Hyperparameters must be fixed from published values or from a
pre-declared pilot before confirmatory seeds start.

## Metrics and reporting contract

### Main table

Every Tier-A row reports `mean ± sample std` across seeds 0/1/2:

- primary: FID ↓ and IPR Recall ↑;
- secondary quality/diversity: KID ↓, IS ↑, PRD F₈ ↑, PRD F₁⁄₈ ↑, IPR
  Precision ↑;
- W&B receives a per-seed table, aggregate table, metric definitions, and the
  exact run/config provenance.

`Recall` always means **VGG16-fc2 improved-PRD, k=3**. The two standard PRD
endpoints are named `F₈` and `F₁⁄₈`; they are never overloaded into `Recall`.
FID and KID share the pinned CBDM Inception feature extractor and a balanced
CIFAR training reference. KID uses the CM release's unbiased cubic-kernel MMD
with a fixed subset RNG seed, 100 subsets, and at most 1,000 features/subset.

### Tail diagnostics

The runner also writes `tail_breakdown.md` and `tail_per_seed.csv`:

- per-class FID for every cell;
- CM's class-index Many/Medium/Few groups: C10 = 0–2 / 3–5 / 6–9; C100 =
  0–32 / 33–65 / 66–99.

They use the same 50k class-uniform samples as the common table. That gives
5,000 generated samples/class on CIFAR-10 — exactly CORAL's per-class-FID
count for the IF1000 cell — but it is **not** CM's split protocol, which draws
20k generated images independently for each split. The output records the
actual generated/reference count in every tail row, and the paper must state
this caveat.

## Decision rule for a result claim

A method may be described as superior on a Tier-A cell only if all conditions
hold:

1. all three paired seed jobs finished and the evaluator artifacts are valid;
2. FID is lower and IPR Recall is not worse on the aggregate; 
3. the paired-seed FID advantage against each named comparator has a
   bootstrap 95% interval above zero, or the text calls the outcome
   inconclusive;
4. Many/Medium/Few diagnostics show where the gain comes from; do not claim a
   tail improvement from overall FID alone;
5. a scaling or ablation result is never substituted for the main benchmark.

This is deliberately conservative: three seeds are modest, and their standard
deviation is descriptive. The paired analysis prevents one lucky run or an
unpaired comparison from becoming a paper claim.

## Operational launch order

1. Run the packaged smoke task on the borrowed server; do not train a full
   result matrix until CUDA, downloads, metric assets, and W&B artifact upload
   have passed.
2. Launch Tier A with `bash scripts/run_server.sh` and let it resume until the
   fail-closed report contains all 15 cells.
3. Inspect `report/table.md`, `report/tail_breakdown.md`, W&B's per-seed
   tables, generated grids, and both data/reference manifests.
4. Only then spend the large-data budget on one labelled-exploratory
   ImageNet-LT 64 run. Promote it to a paper headline only after the same
   three-seed rule.
