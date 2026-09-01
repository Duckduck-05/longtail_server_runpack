#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"; source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
exec python -m ltx.cli run --config "${1:-configs/native_cifar100_if100.yaml}"
