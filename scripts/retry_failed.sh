#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"; source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
ARGS=(--config "${1:-configs/native_cifar100_if100.yaml}")
[[ -n "${STAGE:-}" ]] && ARGS+=(--stage "$STAGE")
python -m ltx.cli retry-failed "${ARGS[@]}"
