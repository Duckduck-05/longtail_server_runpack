#!/usr/bin/env bash
# Prepared local/T2H launcher only. Nothing runs until this script is invoked
# explicitly on the intended host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a; source "$ROOT/.env.local"; set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

SOURCE_CHECKPOINT="${LTX_DDPM_NATIVE_CKPT:-$ROOT/runs/unified_cifar_c100_v1/c100_if100_core/ddpm/seed_0/ckpt_200000.pt}"
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "[hybrid-smoke] native source checkpoint missing: $SOURCE_CHECKPOINT" >&2
  exit 2
fi
export LTX_DDPM_NATIVE_CKPT="$SOURCE_CHECKPOINT"
echo "[hybrid-smoke] diagnostic/non-paper native import" >&2
echo "[hybrid-smoke] source=$(realpath "$SOURCE_CHECKPOINT") sha256=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" >&2

bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

# One hybrid task, exactly 20k DDIM-50 samples, and no automatic 10k fallback.
RUN_ARGS=(--config configs/ipsvt_hybrid_smoke_c100_if100.yaml --jobs 1 --per-gpu 1)
[[ -n "${LTX_GPU_IDS:-}" ]] && RUN_ARGS+=(--gpus "$LTX_GPU_IDS")
RUN_ARGS+=("$@")
exec python -m ltx.cli run "${RUN_ARGS[@]}"
