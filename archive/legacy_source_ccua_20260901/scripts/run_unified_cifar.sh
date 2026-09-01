#!/usr/bin/env bash
# One-command common T2H-host comparison: 54 controlled CIFAR-LT tasks.
#
# Extra arguments (e.g. --per-gpu 3 --gpus 0,1,2,3) pass straight through to
# `ltx.cli run`, so the campaign's GPU packing can be tuned per rented box
# without editing configs/server.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a; source "$ROOT/.env.local"; set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

# Every phase of this one command (bootstrap, metric-asset prep, the
# scheduler's launch decisions, the final report) lands in one file so there
# is a single log to read or hand over.
CAMPAIGN_NAME="unified_cifar_t2h_v1"
RUNS_ROOT="${LTX_RUNS_ROOT:-$ROOT/runs}"
LOG_DIR="$RUNS_ROOT/$CAMPAIGN_NAME/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$RUN_LOG" "$RUNS_ROOT/$CAMPAIGN_NAME/latest.log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "[run] full log: $RUN_LOG"

bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online but WANDB_API_KEY is empty; copy .env.example to .env.local or set WANDB_MODE=offline/disabled explicitly." >&2
  exit 2
fi
if [[ "${WANDB_MODE:-online}" == "disabled" ]]; then
  echo "[run] warning: W&B is disabled; task results will remain local under ${RUNS_ROOT}." >&2
fi

# `torchvision` downloads CIFAR-10/100 automatically during this preparation
# and during training if necessary. This derives the one balanced-reference
# metric asset set used by every common-host method row.
T2H_METRICS_ROOT="${LTX_T2H_METRICS_ROOT:-${LTX_REPOS_ROOT:-$ROOT/third_party}/T2H-unified/stats}"
python tools/prepare_cifar_metric_assets.py \
  --repo "${LTX_REPOS_ROOT:-$ROOT/third_party}/T2H-unified" \
  --data-root "${LTX_DATA_ROOT:-$ROOT/data}" \
  --output "$T2H_METRICS_ROOT"

RUN_ARGS=(--config configs/unified_cifar.yaml)
[[ -n "${LTX_GPU_IDS:-}" ]] && RUN_ARGS+=(--gpus "$LTX_GPU_IDS")
[[ -n "${LTX_TASKS_PER_GPU:-}" ]] && RUN_ARGS+=(--per-gpu "$LTX_TASKS_PER_GPU")
[[ -n "${LTX_MAX_CONCURRENT:-}" ]] && RUN_ARGS+=(--jobs "$LTX_MAX_CONCURRENT")
RUN_ARGS+=("$@")

# Both statuses are captured with `set -e` disabled. The report exits non-zero
# on purpose when a seed or metric is missing (fail-closed), so letting `set -e`
# abort here would skip the combined check below and lose the campaign status.
set +e
python -m ltx.cli run "${RUN_ARGS[@]}"
campaign_status=$?
python tools/report_unified_cifar.py --config configs/unified_cifar.yaml --wandb
report_status=$?
set -e

echo "[run] campaign_status=$campaign_status report_status=$report_status"
echo "[run] full log: $RUN_LOG"
[[ "$campaign_status" -eq 0 && "$report_status" -eq 0 ]]
