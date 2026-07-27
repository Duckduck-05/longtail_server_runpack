#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
if [[ -f "$ROOT/.env.local" ]]; then set -a; source "$ROOT/.env.local"; set +a; elif [[ -f "$ROOT/.env" ]]; then set -a; source "$ROOT/.env"; set +a; fi
bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
python tools/prepare_cifar_metric_assets.py --repo "${LTX_REPOS_ROOT:-$ROOT/third_party}/CBDM-pytorch" --data-root "${LTX_DATA_ROOT:-$ROOT/data}" --output "${LTX_METRICS_ROOT:-${LTX_REPOS_ROOT:-$ROOT/third_party}/CBDM-pytorch/stats}" --datasets cifar100
set +e; python -m ltx.cli run --config configs/coral2025_c100_compact.yaml; campaign_status=$?; set -e
python tools/report_coral2025.py --config configs/coral2025_c100_compact.yaml --wandb; report_status=$?
[[ "$campaign_status" -eq 0 && "$report_status" -eq 0 ]]
