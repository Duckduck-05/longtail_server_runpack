#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"; source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
python -m ltx.cli status --watch "${WATCH_SECONDS:-20}" --config "${1:-configs/native_cifar100_if100.yaml}"
