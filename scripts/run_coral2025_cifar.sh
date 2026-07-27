#!/usr/bin/env bash
# One-command, fail-closed CIFAR reproduction of CORAL Table 1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# A copied runpack is self-contained: read its ignored local credential file
# when present, without copying any secret into source control.
if [[ -f "$ROOT/.env.local" ]]; then
  set -a; source "$ROOT/.env.local"; set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

# bootstrap is idempotent and verifies the vendored third-party manifest.
bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
python tools/prepare_cifar_metric_assets.py \
  --repo "${LTX_REPOS_ROOT:-$ROOT/third_party}/CBDM-pytorch" \
  --data-root "${LTX_DATA_ROOT:-$ROOT/data}" \
  --output "${LTX_METRICS_ROOT:-${LTX_REPOS_ROOT:-$ROOT/third_party}/CBDM-pytorch/stats}"
set +e
python -m ltx.cli run --config configs/coral2025_cifar.yaml
campaign_status=$?
set -e
python tools/report_coral2025.py --config configs/coral2025_cifar.yaml --wandb
report_status=$?
[[ "$campaign_status" -eq 0 && "$report_status" -eq 0 ]]
