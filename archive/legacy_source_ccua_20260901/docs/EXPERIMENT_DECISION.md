# Experiment decision

## Decision

Use CORAL Table-1 CIFAR as the primary reproducible benchmark. Do not combine
CM, CORAL, and unrelated long-tail datasets in one headline table: their
published training horizons and public evaluators differ.

## Recommended order

1. **Compact confirmation:** CIFAR100-LT IF100, DDPM/CBDM/T2H/CORAL, three
   seeds. This is one complete Table-1 row (12 runs) and has the highest class
   count among the public, auto-downloadable protocols. Run it with
   `bash scripts/run_coral2025_c100_compact.sh`.
2. **Full CORAL CIFAR table:** add CIFAR10-LT IF100 and IF1000 (36 runs total).
3. **Separate CM reproduction:** run its official CIFAR100-LT IF100 protocol
   and report it in a CM-specific table. Its public vendored source has no
   iNaturalist, Places-LT, or ImageNet-LT protocol.

## Why not iNaturalist/Places-LT now?

Neither the vendored CM code nor its checked-in configs includes the required
data construction, split manifest, image resolution, or evaluation protocol.
Adding a generic ImageFolder loader would create a new experiment. It must not
be labelled CM-paper reproduction.

## Reproducibility contract

- Seeds: 0, 1, 2; report mean ± sample standard deviation.
- CIFAR: torchvision downloads raw data automatically on the target server.
- Evaluation: 50k generated samples, balanced real references, shared FID/IS/
  F₈/Recall/F₁⁄₈ evaluator for the CORAL track.
- W&B: set credentials only in the ignored root `.env.local`; all task runs and
  the final comparison Tables upload to that account.
