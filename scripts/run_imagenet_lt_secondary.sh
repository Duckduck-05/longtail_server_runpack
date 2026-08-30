#!/usr/bin/env bash
# Run the deferred ImageNet-LT 64x64 DDPM/CCUA setting.
#
# This is intentionally not part of run_server_c100.sh.  Set the gate explicitly
# to document why this expensive secondary cell is being launched:
#   LTX_IMAGENET_LT_GATE=access          # on ACCESS
#   LTX_IMAGENET_LT_GATE=main_complete   # after the CIFAR main table is full
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env.local" ]]; then
  set -a; source "$ROOT/.env.local"; set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

GATE="${LTX_IMAGENET_LT_GATE:-}"
case "$GATE" in
  access)
    echo "[run-imagenet-lt] ACCESS gate acknowledged"
    ;;
  main_complete)
    python tools/check_campaign_complete.py --config configs/unified_cifar_c100.yaml
    ;;
  *)
    echo "Set LTX_IMAGENET_LT_GATE=access on ACCESS or main_complete after the CIFAR main table is complete." >&2
    exit 2
    ;;
esac

CAMPAIGN_NAME="secondary_imagenet_lt_v1"
RUNS_ROOT="${LTX_RUNS_ROOT:-$ROOT/runs}"
LOG_DIR="$RUNS_ROOT/$CAMPAIGN_NAME/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$RUN_LOG" "$RUNS_ROOT/$CAMPAIGN_NAME/latest.log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "[run-imagenet-lt] full log: $RUN_LOG"

# This can download/expand the licensed ImageNet payload and should be run on
# the ACCESS filesystem or another host with enough storage.  If the root is
# already mounted, the preparation script only validates/uses the manifests.
source "$ROOT/scripts/prepare_imagenet_lt.sh"
bash scripts/bootstrap.sh
source "${LTX_VENV:-$ROOT/.venv}/bin/activate"

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online but WANDB_API_KEY is empty; set WANDB_API_KEY or use WANDB_MODE=offline/disabled." >&2
  exit 2
fi

bash scripts/prepare_cm_metric_assets.sh
python tools/validate_imagenet_lt.py \
  --image-root "$LTX_IMAGENET_ROOT" \
  --train-manifest "$LTX_IMAGENET_LT_TRAIN_MANIFEST" \
  --reference-manifest "$LTX_IMAGENET_LT_REFERENCE_MANIFEST"
python -m ltx.cli preflight --config configs/secondary_imagenet_lt.yaml

RUN_ARGS=(--config configs/secondary_imagenet_lt.yaml)
[[ -n "${LTX_GPU_IDS:-}" ]] && RUN_ARGS+=(--gpus "$LTX_GPU_IDS")
[[ -n "${LTX_TASKS_PER_GPU:-}" ]] && RUN_ARGS+=(--per-gpu "$LTX_TASKS_PER_GPU")
[[ -n "${LTX_MAX_CONCURRENT:-}" ]] && RUN_ARGS+=(--jobs "$LTX_MAX_CONCURRENT")
RUN_ARGS+=("$@")

set +e
python -m ltx.cli run "${RUN_ARGS[@]}"
campaign_status=$?
python tools/report_imagenet_lt.py --config configs/secondary_imagenet_lt.yaml --wandb
report_status=$?
set -e

echo "[run-imagenet-lt] campaign_status=$campaign_status report_status=$report_status"
echo "[run-imagenet-lt] full log: $RUN_LOG"
[[ "$campaign_status" -eq 0 && "$report_status" -eq 0 ]]
