#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python tools/compute_metrics.py \
  --feature_dir features \
  --generated_prefix OC-cifar100-100 \
  --output outputs/oc_cifar100lt_ir100/metrics.json \
  "$@"
