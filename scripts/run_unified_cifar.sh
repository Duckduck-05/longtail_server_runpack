#!/usr/bin/env bash
# One-command source-native comparison: 45 controlled CIFAR-LT tasks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a; source "$ROOT/.env.local"; set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

# `torchvision` downloads CIFAR-10/100 automatically during this preparation
# and during training if necessary.  This derives the one balanced-reference
# metric asset set used by all five methods.
python tools/prepare_cifar_metric_assets.py \
  --repo "${LTX_REPOS_ROOT:-$ROOT/third_party}/CBDM-pytorch" \
  --data-root "${LTX_DATA_ROOT:-$ROOT/data}" \
  --output "${LTX_METRICS_ROOT:-${LTX_REPOS_ROOT:-$ROOT/third_party}/CBDM-pytorch/stats}"

set +e
python -m ltx.cli run --config configs/unified_cifar.yaml
campaign_status=$?
set -e
python tools/report_unified_cifar.py --config configs/unified_cifar.yaml --wandb
report_status=$?
[[ "$campaign_status" -eq 0 && "$report_status" -eq 0 ]]
