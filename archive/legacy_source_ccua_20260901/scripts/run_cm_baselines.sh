#!/usr/bin/env bash
# One command: CM paper baseline matrix across CIFAR-10-LT, CIFAR-100-LT and ImageNet-LT.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/scripts/run_cm_imagenet_lt.sh"
