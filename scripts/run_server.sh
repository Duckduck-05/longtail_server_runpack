#!/usr/bin/env bash
# One-command entrypoint for a fresh CUDA server.
#
# Any arguments (e.g. --per-gpu 3 --gpus 0,1,2,3) pass through to
# the CCUA-backbone CIFAR launcher -> `ltx.cli run`. With no arguments, GPU packing
# auto-detects from free VRAM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${LTX_CONDA_ENV:-longtail-ccua}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  candidate="$PYTHON_BIN"
elif command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$conda_base/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    # A reused environment must converge to the checked-in lock, otherwise a
    # colleague can silently run an old torch/CUDA metric stack.
    conda env update --file environment.yml --name "$ENV_NAME" --prune
  else
    conda env create --file environment.yml --name "$ENV_NAME"
  fi
  candidate="$(conda run --no-capture-output -n "$ENV_NAME" which python)"
elif command -v python >/dev/null 2>&1 && python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  # Fallback only for a host where conda is intentionally unavailable.
  candidate="$(command -v python)"
else
  echo "No CUDA-enabled Python or conda found. Install conda, then rerun this command." >&2
  exit 2
fi

export PYTHON_BIN="$candidate"
# The default entrypoint is the CCUA-backbone CIFAR-100-LT wave.
exec bash scripts/run_native_cifar_c100.sh "$@"
