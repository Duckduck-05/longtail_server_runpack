#!/usr/bin/env bash
# One command for the complete ImageNet-LT CM baseline matrix (4 methods x 3 seeds).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.env.local" ]]; then set -a; source "$ROOT/.env.local"; set +a
elif [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a
fi
# This exports default paths and, on a new server, downloads the authenticated
# ImageNet archive/manifests declared in this runpack's .env.local.
# shellcheck source=prepare_imagenet_lt.sh
source "$ROOT/scripts/prepare_imagenet_lt.sh"
LTX_ENABLE_LEGACY_NATIVE=1 bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
bash scripts/prepare_cm_metric_assets.sh
python tools/validate_imagenet_lt.py --image-root "$LTX_IMAGENET_ROOT" \
  --train-manifest "$LTX_IMAGENET_LT_TRAIN_MANIFEST" \
  --reference-manifest "$LTX_IMAGENET_LT_REFERENCE_MANIFEST"
python -m ltx.cli preflight --config configs/cm_imagenet_lt.yaml
python -m ltx.cli run --config configs/cm_imagenet_lt.yaml
python tools/report_cm_imagenet_lt.py --config configs/cm_imagenet_lt.yaml --wandb
