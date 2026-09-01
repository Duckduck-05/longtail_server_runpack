#!/usr/bin/env bash
# Launch a campaign on the shared CCUA-DDPM backbone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  source "$ROOT/.env.local"
  set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

# These are the known server-local roots.  The override prevents an old
# .env.local from sending the new CCUA campaign to an archived host/cache.
export LTX_REPOS_ROOT="$ROOT/third_party"
export LTX_DATA_ROOT="$ROOT/data"
if [[ -z "${LTX_RUNS_ROOT:-}" && -d "/home/nvidia-lab/data_mount" ]]; then
  export LTX_RUNS_ROOT="/home/nvidia-lab/data_mount/longtail_server_runpack/runs"
else
  export LTX_RUNS_ROOT="${LTX_RUNS_ROOT:-$ROOT/runs}"
fi
export LTX_METRICS_ROOT="$LTX_REPOS_ROOT/CCUA-DDPM/stats"

CONFIG="${1:-configs/native_cifar100_if100_ddpm_cbdm.yaml}"

bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online but WANDB_API_KEY is empty." >&2
  exit 2
fi

python tools/prepare_cifar_metric_assets.py \
  --repo "$LTX_REPOS_ROOT/CCUA-DDPM" \
  --data-root "$LTX_DATA_ROOT" \
  --output "$LTX_METRICS_ROOT" \
  --datasets cifar100

mkdir -p "$LTX_RUNS_ROOT"

RUN_ARGS=(--config "$CONFIG")
[[ -n "${LTX_GPU_IDS:-}" ]] && RUN_ARGS+=(--gpus "$LTX_GPU_IDS")
[[ -n "${LTX_TASKS_PER_GPU:-}" ]] && RUN_ARGS+=(--per-gpu "$LTX_TASKS_PER_GPU")
[[ -n "${LTX_MAX_CONCURRENT:-}" ]] && RUN_ARGS+=(--jobs "$LTX_MAX_CONCURRENT")

python -m ltx.cli preflight --config "$CONFIG"
python -m ltx.cli run "${RUN_ARGS[@]}"
python tools/report_native_cifar.py --config "$CONFIG" --wandb
