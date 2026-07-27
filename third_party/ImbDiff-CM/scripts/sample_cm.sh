#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python tools/sample_images.py \
  --config configs/cifar100lt_ir100/cm.yaml \
  --ckpt outputs/cm_cifar100lt_ir100/ckpt_300000.pt \
  "$@"
