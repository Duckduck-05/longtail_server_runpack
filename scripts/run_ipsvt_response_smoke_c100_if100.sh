#!/usr/bin/env bash
# Prepared launcher only. It does not run until invoked explicitly on the
# target machine. The two T2H smoke tasks are submitted together (--jobs 2).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a; source "$ROOT/.env.local"; set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

# `20k` is the normal smoke evaluation. Falling back to 10k is allowed only
# when either guard says the host is tight; retain the selected value in the
# environment so the task config/sample provenance records it exactly.
RUNS_ROOT="${LTX_RUNS_ROOT:-$ROOT/runs}"
PREBOOTSTRAP_PYTHON="${LTX_PREBOOTSTRAP_PYTHON:-${LTX_VENV:-$ROOT/.venv}/bin/python}"
if [[ ! -x "$PREBOOTSTRAP_PYTHON" ]]; then
  PREBOOTSTRAP_PYTHON="${PYTHON_BIN:-python3}"
fi
DISK_GUARD_GB="${LTX_IPSVT_SMOKE_MIN_FREE_DISK_GB:-${LTX_DISK_STOP_FREE_GB:-100}}"
GPU_GUARD_MB="${LTX_IPSVT_SMOKE_MIN_FREE_GPU_MB:-16000}"
SMOKE_IMAGES="${LTX_IPSVT_SMOKE_NUM_IMAGES:-20000}"

if [[ -z "${LTX_IPSVT_SMOKE_NUM_IMAGES:-}" ]]; then
  free_disk_gb="$("$PREBOOTSTRAP_PYTHON" - "$RUNS_ROOT" <<'PY'
from pathlib import Path
import shutil, sys
probe = Path(sys.argv[1]).expanduser()
while not probe.exists() and probe != probe.parent:
    probe = probe.parent
print(shutil.disk_usage(probe).free // 1024**3)
PY
)"
  free_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print int($1)}')"
  if [[ -z "$free_gpu_mb" || "$free_disk_gb" -lt "$DISK_GUARD_GB" || "$free_gpu_mb" -lt "$GPU_GUARD_MB" ]]; then
    SMOKE_IMAGES=10000
    echo "[smoke] guarded fallback: DDIM-50 evaluation uses 10k images (disk=${free_disk_gb}GB guard=${DISK_GUARD_GB}GB, gpu=${free_gpu_mb:-unavailable}MB guard=${GPU_GUARD_MB}MB)" >&2
  else
    echo "[smoke] guard passed: DDIM-50 evaluation uses 20k images (disk=${free_disk_gb}GB, gpu=${free_gpu_mb}MB)" >&2
  fi
fi
export LTX_IPSVT_SMOKE_NUM_IMAGES="$SMOKE_IMAGES"

SOURCE_CHECKPOINT="${LTX_DDPM_NATIVE_CKPT:-$ROOT/runs/unified_cifar_c100_v1/c100_if100_core/ddpm/seed_0/ckpt_200000.pt}"
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "[smoke] native source checkpoint missing: $SOURCE_CHECKPOINT" >&2
  exit 2
fi
export LTX_DDPM_NATIVE_CKPT="$SOURCE_CHECKPOINT"
echo "[smoke] source=$(realpath "$SOURCE_CHECKPOINT") sha256=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" >&2

bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

# Supply two physical GPUs (for example LTX_GPU_IDS=0,1) to actually overlap
# the two training phases. No command in this file is run during implementation.
RUN_ARGS=(--config configs/ipsvt_response_smoke_c100_if100.yaml --jobs 2 --per-gpu 1)
[[ -n "${LTX_GPU_IDS:-}" ]] && RUN_ARGS+=(--gpus "$LTX_GPU_IDS")
RUN_ARGS+=("$@")
exec python -m ltx.cli run "${RUN_ARGS[@]}"
