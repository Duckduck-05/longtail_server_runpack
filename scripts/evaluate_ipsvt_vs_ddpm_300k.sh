#!/usr/bin/env bash
set -euo pipefail

# Step-300k comparison for the common T2H-host CIFAR-100-LT runs.
# This consumes the canonical host arrays and applies the same evaluator to
# both objectives; it never retrains either method.

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="$ROOT/.venv/bin"
BASE="$ROOT/runs/unified_cifar_c100_t2h_v1/c100_if100_core"
DDPM="$BASE/ddpm/seed_0"
IPSVT="$BASE/ipsvt/seed_0"
REPO="$ROOT/third_party/T2H-unified"
EVALUATOR="$ROOT/tools/evaluate_coral2025.py"

DDPM_METRICS="$DDPM/metrics.unified.v2.json"
DDPM_PER_CLASS="$DDPM/metrics.per_class.v2.json"
DDPM_SAMPLE="$DDPM/samples.t2h_unified_v2.npy"
IPSVT_SAMPLE="$IPSVT/samples.t2h_unified_v2.npy"
IPSVT_LABELS="$IPSVT/samples.t2h_unified_v2.npy.labels.npy"
IPSVT_METRICS="$IPSVT/metrics.unified.v2.json"
IPSVT_PER_CLASS="$IPSVT/metrics.per_class.v2.json"
COMPARISON="$BASE/comparison_ddpm_vs_ipsvt_step300000.json"

sample_provenance_valid() {
  local sample="$1"
  local objective="$2"
  SAMPLE_PATH="$sample" EXPECTED_OBJECTIVE="$objective" "$VENV/python" - <<'PY'
import json
import os
import sys
from pathlib import Path

sample = Path(os.environ["SAMPLE_PATH"])
sidecar = sample.with_suffix(".provenance.json")
try:
    actual = json.loads(sidecar.read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    sys.exit(1)
expected = {
    "host": "T2H-unified",
    "host_revision": "t2h-unified-common-v2",
    "checkpoint_schema": 2,
    "objective": os.environ["EXPECTED_OBJECTIVE"],
    "checkpoint_step": 300000,
    "num_images": 50000,
    "sample_method": "ddim",
    "sampler_method": "ddim",
    "ddim_skip_step": 10,
    "omega": 1.5,
    "uniform_labels": True,
    "seed": 0,
    "artifact_namespace": "t2h_unified_v2",
    "T": 1000,
    "beta_1": 0.0001,
    "beta_T": 0.02,
    "var_type": "fixedlarge",
    "img_size": 32,
    "num_class": 100,
}
if not isinstance(actual, dict):
    sys.exit(1)
for key, wanted in expected.items():
    got = actual.get(key)
    if isinstance(wanted, float):
        try:
            if abs(float(got) - wanted) > 1e-12 * max(1.0, abs(wanted)):
                sys.exit(1)
        except (TypeError, ValueError):
            sys.exit(1)
    elif got != wanted:
        sys.exit(1)
PY
}

metric_artifacts_valid() {
  local objective="$1"
  local sample="$2"
  local metrics="$3"
  local per_class="$4"
  sample_provenance_valid "$sample" "$objective" || return 1
  OBJECTIVE="$objective" SAMPLE_PATH="$sample" METRICS_PATH="$metrics" \
    PER_CLASS_PATH="$per_class" "$VENV/python" - <<'PY'
import json
import os
import sys
from pathlib import Path

def read(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None

sample = read(str(Path(os.environ["SAMPLE_PATH"]).with_suffix(".provenance.json")))
metrics = read(os.environ["METRICS_PATH"])
per_class = read(os.environ["PER_CLASS_PATH"])
if sample is None or metrics is None or per_class is None:
    sys.exit(1)
expected = {
    "metric_host": "common_cifar_metrics_v2",
    "sample": sample,
}
if metrics.get("provenance") != expected or per_class.get("provenance") != expected:
    sys.exit(1)
metric_values = metrics.get("metrics")
if not isinstance(metric_values, dict) or not {
    "FID", "KID", "IS", "F_8", "F_1_8", "ImprovedPrecision", "Recall",
}.issubset(metric_values):
    sys.exit(1)
if not isinstance(per_class.get("per_class"), dict):
    sys.exit(1)
groups = per_class.get("groups")
if not isinstance(groups, dict) or not {"Many", "Medium", "Few"}.issubset(groups):
    sys.exit(1)
if sample.get("objective") != os.environ["OBJECTIVE"]:
    sys.exit(1)
sys.exit(0)
PY
}

if [[ ! -f "$IPSVT_SAMPLE" || ! -f "$IPSVT_LABELS" ]] || \
   ! sample_provenance_valid "$IPSVT_SAMPLE" ipsvt; then
  echo "[comparison] sampling IP-SVT step=300000" >&2
  cd "$REPO"
  env PATH="$VENV:$PATH" CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
    "$VENV/python" unified_main.py --sample --objective=ipsvt --ipsvt --ipsvt_mode=full \
    --data_type=cifar100lt --imb_factor=0.01 --root="$ROOT/data" \
    --logdir="$IPSVT" --seed=0 --checkpoint_prefix=ckpt_unified_v2_ \
    --ckpt_step=300000 --batch_size=64 --lr=0.0002 --warmup=5000 \
    --dropout=0.1 --grad_clip=1.0 --ema_decay=0.9999 --T=1000 \
    --beta_1=0.0001 --beta_T=0.02 --var_type=fixedlarge \
    --sample_batch_size=64 --num_images=50000 --sample_method=ddim \
    --ddim_skip_step=10 --omega=1.5 --artifact_namespace=t2h_unified_v2 \
    --sample_output="$IPSVT_SAMPLE" \
    --ch=128 --ch_mult=1 --ch_mult=2 --ch_mult=2 --ch_mult=2 \
    --attn=1 --num_res_blocks=2 --conditional --cfg --uniform_labels \
    --ipsvt_lambda_aux=1.0 --ipsvt_lambda_svt=1.0 --ipsvt_K=4 \
    --ipsvt_s=0.05 --ipsvt_delta=0.1 --ipsvt_every=4 --ipsvt_batch=16
fi

echo "[comparison] waiting for detailed DDPM metrics before metric comparison" >&2
while ! metric_artifacts_valid ddpm "$DDPM_SAMPLE" "$DDPM_METRICS" "$DDPM_PER_CLASS"; do
  sleep 60
done

if ! metric_artifacts_valid ipsvt "$IPSVT_SAMPLE" "$IPSVT_METRICS" "$IPSVT_PER_CLASS"; then
  echo "[comparison] evaluating IP-SVT step=300000" >&2
  env PATH="$VENV:$PATH" CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
    "$VENV/python" "$EVALUATOR" \
    --repo "$ROOT/third_party/T2H-unified" --data-type cifar100lt \
    --samples "$IPSVT_SAMPLE" --labels "$IPSVT_LABELS" \
    --metrics-root "$ROOT/third_party/T2H-unified/stats" \
    --output "$IPSVT_METRICS" --inception-batch-size 16 \
    --kid --kid-subsets 100 --kid-subset-size 1000 --kid-seed 2026 \
    --per-class-output "$IPSVT_PER_CLASS" --longtail-groups cm_three_way \
    --expected-host-revision t2h-unified-common-v2 \
    --expected-checkpoint-schema 2 --expected-objective ipsvt \
    --expected-checkpoint-step 300000 --expected-num-images 50000 \
    --expected-sample-method ddim --expected-sampler-method ddim \
    --expected-ddim-skip-step 10 --expected-omega 1.5 --expected-seed 0 \
    --expected-artifact-namespace t2h_unified_v2 --expected-T 1000 \
    --expected-beta-1 0.0001 --expected-beta-T 0.02 \
    --expected-var-type fixedlarge --expected-img-size 32 \
    --expected-num-class 100 --expected-uniform-labels
fi

if ! metric_artifacts_valid ipsvt "$IPSVT_SAMPLE" "$IPSVT_METRICS" "$IPSVT_PER_CLASS"; then
  echo "[comparison] IP-SVT metric artifacts are incomplete or have mismatched provenance" >&2
  exit 1
fi

ROOT="$ROOT" DDPM_METRICS="$DDPM_METRICS" DDPM_PER_CLASS="$DDPM_PER_CLASS" \
IPSVT_METRICS="$IPSVT_METRICS" IPSVT_PER_CLASS="$IPSVT_PER_CLASS" \
COMPARISON="$COMPARISON" "$VENV/python" - <<'PY'
import json
import os
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

ddpm = read(os.environ["DDPM_METRICS"])
ipsvt = read(os.environ["IPSVT_METRICS"])
ddpm_pc = read(os.environ["DDPM_PER_CLASS"])
ipsvt_pc = read(os.environ["IPSVT_PER_CLASS"])
payload = {
    "step": 300000,
    "methods": {"DDPM": ddpm, "IP-SVT": ipsvt},
    "tail_groups": {
        "DDPM": ddpm_pc.get("groups", {}),
        "IP-SVT": ipsvt_pc.get("groups", {}),
    },
    "per_class": {
        "DDPM": ddpm_pc.get("per_class", {}),
        "IP-SVT": ipsvt_pc.get("per_class", {}),
    },
    "protocol": {
        "dataset": "CIFAR-100-LT IF100",
        "sample_count": 50000,
        "labels": "exact class-uniform",
        "sampler": "CFG omega=1.5, DDIM-100",
        "metrics": "shared evaluate_coral2025.py",
    },
}
out = Path(os.environ["COMPARISON"])
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({
    "DDPM": ddpm.get("metrics", {}),
    "IP-SVT": ipsvt.get("metrics", {}),
    "DDPM_tail": ddpm_pc.get("groups", {}),
    "IP-SVT_tail": ipsvt_pc.get("groups", {}),
}, sort_keys=True))
PY

touch "$IPSVT/IPSVT_DDPM_COMPARISON_300K_COMPLETE"
echo "[comparison] complete: $COMPARISON" >&2
