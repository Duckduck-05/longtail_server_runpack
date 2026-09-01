#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFICATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CLASSIFICATION_DIR}"

python tools/train_classifier.py \
  --config configs/cm_cifar100lt_imb0.005_resnet20.yaml \
  "$@"
