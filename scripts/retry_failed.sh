#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"; source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
ARGS=(--config "${1:-configs/unified_cifar.yaml}")
[[ -n "${STAGE:-}" ]] && ARGS+=(--stage "$STAGE")
python -m ltx.cli retry-failed "${ARGS[@]}"
