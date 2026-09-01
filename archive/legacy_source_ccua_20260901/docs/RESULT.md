# Delivery result

## What changed

- Added `bash scripts/run_coral2025_cifar.sh`: bootstrap → balanced-reference
  metric preparation → fail-closed preflight → 36 from-scratch CIFAR jobs.
- Pinned and audited the CORAL, CBDM, and OC_LT source commits; ported only the
  missing CORAL compatibility modules and portable metric-cache paths.
- Locked the published CIFAR budgets, guidance scales, CBDM/CORAL parameters,
  50k uniform sampling, and three paired seeds in `configs/coral2025_cifar.yaml`.
- Added machine-readable published targets and a reproducibility audit.
- W&B is loaded at execution time from this runpack's ignored `.env.local` or
  `LTX_ENV_FILE`; no key is stored in this runpack.

## Checks run

- `python -m compileall -q ltx patches tools`
- `python -m pytest -q` → 12 passed
- `python -m ltx.cli plan --config configs/coral2025_cifar.yaml` → 36 tasks
- `bash -n scripts/bootstrap.sh scripts/run_coral2025_cifar.sh`
- preflight without a GPU/repositories → correctly refused launch

## Remaining scientific limits

- The package faithfully executes the three CIFAR Table-1 datasets supported
  by the supplied public sources. CelebA-5 and ImageNet-LT cannot be ported
  faithfully because their paper splits/loaders are absent.
- The runpack uses its shared VGG16-fc2, k=3 evaluator for paper Recall and
  records its generated balanced-reference assets in the metric manifest.
