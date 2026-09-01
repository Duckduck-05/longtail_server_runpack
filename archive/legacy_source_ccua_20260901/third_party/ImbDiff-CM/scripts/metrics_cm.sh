#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python tools/compute_metrics.py \
  --feature_dir features \
  --generated_prefix CM-cifar100-100 \
  --output outputs/cm_cifar100lt_ir100/metrics.json \
  "$@"
