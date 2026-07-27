#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python tools/extract_features.py \
  --mode generated \
  --name CM-cifar100-100 \
  --image_dir outputs/cm_cifar100lt_ir100/revised_gen_images-ckpt_step-300000 \
  --feature_dir features \
  "$@"
