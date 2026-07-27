#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${LTX_VENV:-$ROOT/.venv}"
REPOS_ROOT="${LTX_REPOS_ROOT:-$ROOT/third_party}"
INSTALL_REPO_DEPS="${LTX_INSTALL_REPO_DEPS:-1}"

# Always converge the local venv to the interpreter selected by run_server.
# This avoids retaining a stale venv after the pinned conda environment changes.
"$PYTHON_BIN" -m venv --upgrade --system-site-packages "$VENV"
source "$VENV/bin/activate"
# Keep compatibility with the CUDA PyTorch builds commonly preinstalled on
# research servers (torch 2.12 currently requires setuptools < 82).
python -m pip install --upgrade "pip<26" "setuptools<82" wheel
python -m pip install -e .

required=(CBDM-pytorch ImbDiff-CM OC_LT coral-lt-diffusion)
for directory in "${required[@]}"; do
  if [[ ! -d "$REPOS_ROOT/$directory" ]]; then
    echo "[bootstrap] missing vendored third_party/$directory. The delivered runpack is incomplete." >&2
    exit 2
  fi
done
if [[ ! -f "$REPOS_ROOT/THIRD_PARTY_MANIFEST.json" ]]; then
  echo "[bootstrap] missing third_party/THIRD_PARTY_MANIFEST.json; refusing an unverifiable vendor tree." >&2
  exit 2
fi

# Operational / portability patches only. They are idempotent and never reset
# a vendored working tree, so the package has no runtime dependency on another
# checkout or network git clones.
cp patches/ltx_manifest_dataset.py "$REPOS_ROOT/coral-lt-diffusion/ltx_manifest_dataset.py"
python patches/apply_coral_weighted_sampler.py "$REPOS_ROOT/coral-lt-diffusion"
python patches/apply_oc_seed_patch.py "$REPOS_ROOT/OC_LT"
python patches/apply_oc_metric_weights_patch.py "$REPOS_ROOT/OC_LT"
python patches/apply_oc_sample_export.py "$REPOS_ROOT/OC_LT"
python patches/apply_uniform_eval_labels.py "$REPOS_ROOT"
python patches/apply_cbdm_metric_paths.py "$REPOS_ROOT/CBDM-pytorch"
python patches/apply_cm_imagenet_lt.py "$REPOS_ROOT/ImbDiff-CM"
python patches/apply_cm_array_export.py "$REPOS_ROOT/ImbDiff-CM"
mkdir -p "$REPOS_ROOT/ImbDiff-CM/configs/imagenet_lt"
cp patches/cm_imagenet_lt.yaml "$REPOS_ROOT/ImbDiff-CM/configs/imagenet_lt/cm.yaml"

# CORAL's public commit only contains the changed files and imports unchanged
# CBDM modules. Wire those pinned modules locally; no sibling checkout needed.
mkdir -p "$REPOS_ROOT/CBDM-pytorch/stats"
for item in dataset.py score utils stats; do
  target="$REPOS_ROOT/coral-lt-diffusion/$item"
  if [[ -e "$target" && ! -L "$target" ]]; then
    echo "[bootstrap] CORAL compatibility target already exists: $target" >&2
  elif [[ ! -e "$target" ]]; then
    ln -s "../CBDM-pytorch/$item" "$target"
  fi
done
cp patches/coral_loss_tracker.py "$REPOS_ROOT/coral-lt-diffusion/loss_tracker.py"

# OC resolves data and stats relative to its source tree.
SHARED_DATA="${LTX_DATA_ROOT:-$ROOT/data}"
mkdir -p "$SHARED_DATA"
if [[ -L "$REPOS_ROOT/OC_LT/data" && ! -e "$REPOS_ROOT/OC_LT/data" ]]; then unlink "$REPOS_ROOT/OC_LT/data"; fi
if [[ ! -e "$REPOS_ROOT/OC_LT/data" ]]; then ln -s "$SHARED_DATA" "$REPOS_ROOT/OC_LT/data"; fi
if [[ -L "$REPOS_ROOT/OC_LT/stats" && ! -e "$REPOS_ROOT/OC_LT/stats" ]]; then unlink "$REPOS_ROOT/OC_LT/stats"; fi
if [[ -d "$REPOS_ROOT/CBDM-pytorch/stats" && ! -e "$REPOS_ROOT/OC_LT/stats" ]]; then ln -s "$REPOS_ROOT/CBDM-pytorch/stats" "$REPOS_ROOT/OC_LT/stats"; fi

if [[ "$INSTALL_REPO_DEPS" == "1" ]]; then
  # Preserve the server's CUDA/PyTorch installation.  The 2021 CBDM source
  # lists obsolete TensorFlow/PyTorch pins which are not used by this runner.
  python -m pip install -r "$ROOT/requirements.runpack.txt"
fi

{
  date -Is
  sha256sum "$REPOS_ROOT/THIRD_PARTY_MANIFEST.json"
  python - <<'PY'
import torch
print('python_torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
PY
} > "$REPOS_ROOT/VENDOR_AND_ENV.txt"

mkdir -p "${LTX_RUNS_ROOT:-$ROOT/runs}" "${LTX_DATA_ROOT:-$ROOT/data}" "$ROOT/wandb"

echo
printf 'Bootstrap complete. The runner reads this runpack\047s .env.local (or $LTX_ENV_FILE).\n  source %q\n  python -m ltx.cli preflight --config configs/unified_cifar.yaml\n  bash scripts/run_server.sh\n' "$VENV/bin/activate"
