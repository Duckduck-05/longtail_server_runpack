# CORAL 2025 reproduction audit

Scope: CORAL Table 1, CIFAR10-LT IF100, CIFAR10-LT IF1000, and CIFAR100-LT
IF100. The executable matrix has DDPM, CBDM, T2H, and CORAL for each dataset
and seeds 0/1/2: 36 tasks. This is a **new three-seed rerun**, not a claim that
the CORAL authors used three seeds: the supplied paper Table 1 reports point
estimates without a seed count or standard deviations. Hyperparameters are
locked from Appendix A/Table 3 of arXiv:2506.15933v2: CIFAR core methods train
for 150k steps with batch 128, lr 2e-4, dropout 0.1, and T=1000; T2H trains for
200k. Each evaluation requests 50k uniformly class-labelled samples.

The published comparison columns are FID, IS, F_8, improved-PRD Recall, and
F_1_8. Their exact Table-1 values are a machine-readable reference in
`contracts/coral2025_table1.json`; they are comparison targets, never injected
as run results.

## Third-party provenance and port boundary

| Component | Locked commit | Used for |
| --- | --- | --- |
| CORAL | `62c9dfad2da13f5be0f28975e5cd36727c1acc7a` | DDPM/CBDM/CORAL training and evaluation entrypoint |
| CBDM | `513f7fffe7369343499611a8f68212a8b40f11a1` | CORAL's missing unchanged dataset, score, and utility modules |
| OC_LT | `ced184378e8dc784958b6e14fd687781babd836d` | official T2H training/sampling entrypoint |

The public CORAL commit omits `dataset.py`, `score/`, `utils/`, and a no-op
`LossTracker` import. Bootstrap overlays the first three as symlinks to the
locked CBDM checkout and writes a no-op `LossTracker`; this does not alter the
model, data distribution, loss, optimizer, EMA, or sampler. CBDM's one
hard-coded feature-cache directory is replaced only by `LTX_METRICS_ROOT`.
Every resulting checkout remains identifiable by its upstream git commit plus
the patch markers in provenance.

## Fail-closed requirements

Before launch, the one-command runner creates (or preflight requires) the
balanced reference artifacts in
`LTX_METRICS_ROOT` (default `repos/cbdm/stats`):

- `cifar10.train.npz` and `cifar100.train.npz` for FID;
- `cifar10_feats.npy` and `cifar100_feats.npy` for PRD/Recall.

The source repositories do not publish these assets with the code. The runner
creates them once from the balanced CIFAR training split using the pinned CBDM
Inception feature extractor and records checksums in per-dataset manifests. It
will stop rather than use the long-tailed training distribution as the real
reference or return zero-valued PRD metrics.

## Unreproducible paper cells from the supplied sources

The paper also reports CelebA-5 and ImageNet-LT. The public CORAL and CBDM
commits here only implement CIFAR10/100 variants. Their public source lacks the
paper's CelebA-5 split/labels and ImageNet-LT construction; therefore this
runpack deliberately does not fabricate those cells. Adding a generic
ImageFolder loader would be a new implementation, not a faithful port.

The supplied CBDM metric code uses an Inception/k=5 approximation for improved
PRD, which does not match the paper. The runpack therefore bypasses that part
of the source evaluator: it extracts VGG16 fc2 features, computes exact
chunked k-NN manifold radii with `k=3`, and uses those results for the reported
Recall. The balanced VGG16 reference features/radii are generated once from
the downloaded 50k CIFAR training set and fingerprinted in the metric manifest.
FID/IS/F₈/F₁⁄₈ retain the pinned CBDM Inception/PRD implementation. T2H exports
its generated images and labels before entering this same evaluator, so all
four methods have the same five-column protocol.
