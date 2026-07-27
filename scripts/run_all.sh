#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -gt 0 ]]; then
  source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
  exec python -m ltx.cli run --config "$1"
fi
exec bash scripts/run_unified_cifar.sh
