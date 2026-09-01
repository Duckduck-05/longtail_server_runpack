#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"
python -m ltx.cli preflight --config configs/smoke_ccua.yaml || true
exec python -m ltx.cli run --config configs/smoke_ccua.yaml --skip-preflight
