from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from jsonschema import validate as jsonschema_validate

from .config import LoadedCampaign
from .gpu import query_gpus
from .utils import sha256_file


@dataclass
class Check:
    level: str
    name: str
    message: str


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ids_hash(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(ids.astype(str).tolist()).encode("utf-8")).hexdigest()


_IMAGENET_LT_CLASSES = 1000
_IMAGENET_LT_TRAIN_IMAGES = 115_846
_IMAGENET_LT_REFERENCE_PER_CLASS = 20


def _validate_imagenet_manifest(root: Path, manifest: Path, name: str, *, require_balanced: bool) -> Counter[int]:
    """Validate the licensed ImageNet-LT manifest used by a paper metric.

    ImageNet-LT training is intentionally imbalanced; the FID/KID reference
    must instead be the published 20-images-per-class split (20,000 images).
    Keep this check in the launch preflight as well as the standalone validator
    so `ltx run` cannot bypass the scientific contract.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"{name}: image root missing: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"{name}: manifest missing: {manifest}")
    counts: Counter[int] = Counter()
    missing: list[str] = []
    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 2:
            raise ValueError(f"{name}:{lineno}: expected '<relative_image> <0-based_label>'")
        relative, label_raw = fields
        try:
            label = int(label_raw)
        except ValueError as exc:
            raise ValueError(f"{name}:{lineno}: invalid label {label_raw!r}") from exc
        if not 0 <= label < _IMAGENET_LT_CLASSES:
            raise ValueError(f"{name}:{lineno}: label {label} is not in 0..999")
        counts[label] += 1
        image = root / relative
        if len(missing) < 10 and not image.is_file():
            missing.append(str(image))
    if missing:
        raise FileNotFoundError(f"{name}: manifest paths missing (first {len(missing)}): {missing}")
    if set(counts) != set(range(_IMAGENET_LT_CLASSES)):
        absent = sorted(set(range(_IMAGENET_LT_CLASSES)) - set(counts))
        raise ValueError(f"{name}: requires all 1,000 labels; absent={absent[:20]}")
    if require_balanced and len(set(counts.values())) != 1:
        raise ValueError(
            f"{name}: reference manifest must be exactly class-balanced; "
            f"min/class={min(counts.values())}, max/class={max(counts.values())}"
        )
    if require_balanced and next(iter(counts.values())) != _IMAGENET_LT_REFERENCE_PER_CLASS:
        raise ValueError(
            f"{name}: ImageNet-LT paper reference requires exactly "
            f"{_IMAGENET_LT_REFERENCE_PER_CLASS} images/class "
            f"({_IMAGENET_LT_CLASSES * _IMAGENET_LT_REFERENCE_PER_CLASS} total), "
            f"found {next(iter(counts.values()))}/class"
        )
    if not require_balanced and sum(counts.values()) != _IMAGENET_LT_TRAIN_IMAGES:
        raise ValueError(
            f"{name}: ImageNet-LT paper train manifest requires exactly "
            f"{_IMAGENET_LT_TRAIN_IMAGES} images, found {sum(counts.values())}"
        )
    return counts


def _load_manifest(path: Path) -> Tuple[Dict, np.ndarray, np.ndarray, str]:
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError(f"manifest or sidecar missing: {path}, {sidecar}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as p:
        images = np.asarray(p["images"])
        labels = np.asarray(p["train_labels"], dtype=np.int64)
        ids = np.asarray(p["sample_ids"]).astype(str)
    if images.dtype != np.uint8 or images.ndim != 4 or len(images) != len(labels):
        raise ValueError(f"invalid images/labels arrays: {images.dtype} {images.shape} labels={labels.shape}")
    if ids.ndim != 1 or len(ids) != len(labels):
        raise ValueError("sample_ids length mismatch")
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError(f"train labels are not contiguous 0..C-1: {unique.tolist()}")
    computed = _ids_hash(ids)
    if meta.get("sample_ids_sha256") != computed:
        raise ValueError("manifest sample_ids_sha256 mismatch")
    if meta.get("file_sha256") and meta["file_sha256"] != sha256_file(path):
        raise ValueError("manifest file_sha256 mismatch")
    return meta, labels, ids, computed


def _check_weight(path: Path, schema_path: Path, expected_n: int, expected_ids_hash: str,
                  labels: np.ndarray, expected_method: str) -> Tuple[List[Check], np.ndarray | None, Dict]:
    checks: List[Check] = []
    payload: Dict = {}
    if not path.exists():
        return [Check("ERROR", "weight", f"missing {path}")], None, payload
    try:
        w = np.load(path, allow_pickle=False).astype(np.float64)
        if w.ndim != 1 or len(w) != expected_n:
            raise ValueError(f"expected shape ({expected_n},), got {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0) or float(w.sum()) <= 0:
            raise ValueError("weights must be finite, non-negative and have positive mass")
        ess = float(w.sum() ** 2 / np.square(w).sum())
        checks.append(Check("PASS", "weight", f"{path.name}: n={len(w)} ESS={ess:.2f} min={w.min():.3g} max={w.max():.3g}"))
    except Exception as exc:
        return [Check("ERROR", "weight", f"{path}: {exc}")], None, payload

    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        checks.append(Check("ERROR", "weight-manifest", f"missing required sidecar {sidecar}"))
        return checks, w, payload
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        jsonschema_validate(payload, json.loads(schema_path.read_text(encoding="utf-8")))
        if payload.get("weights_sha256") != sha256_file(path):
            raise ValueError("weights_sha256 mismatch")
        if int(payload.get("num_samples", -1)) != expected_n:
            raise ValueError("num_samples mismatch")
        if payload.get("sample_ids_sha256") != expected_ids_hash:
            raise ValueError("sample order does not match frozen manifest")
        if payload.get("method") != expected_method:
            raise ValueError(f"sidecar method={payload.get('method')} expected={expected_method}")
        checks.append(Check("PASS", "weight-manifest", f"validated {sidecar.name}"))
    except Exception as exc:
        checks.append(Check("ERROR", "weight-manifest", f"{sidecar}: {exc}"))

    totals = np.bincount(labels, weights=w, minlength=int(labels.max()) + 1)
    rel = (totals.max() - totals.min()) / max(float(totals.mean()), 1e-12)
    if rel > 1e-6:
        checks.append(Check("ERROR", "within-class-only", f"{path.name}: class total weights differ (relative range {rel:.3g})"))
    else:
        checks.append(Check("PASS", "within-class-only", f"{path.name}: equal class total mass certified"))
    return checks, w, payload


def _check_semantic_bundle(campaign: LoadedCampaign, tasks) -> List[Check]:
    checks: List[Check] = []
    manifest_str = tasks[0].dataset.get("frozen_manifest", "") if tasks else ""
    if not manifest_str:
        return [Check("ERROR", "semantic-manifest", "LTX_SEMANTIC_MANIFEST is empty")]
    manifest = Path(manifest_str)
    try:
        meta, labels, ids, ids_hash = _load_manifest(manifest)
        checks.append(Check("PASS", "semantic-manifest", f"{manifest.name}: n={len(labels)} C={len(np.unique(labels))} ids={ids_hash[:12]}"))
    except Exception as exc:
        return [Check("ERROR", "semantic-manifest", str(exc))]

    if not tasks[0].semantic_eval_command:
        checks.append(Check("ERROR", "semantic-evaluator", "LTX_SEMANTIC_EVAL_COMMAND is empty"))
    else:
        required = {"{samples}", "{labels}", "{output}", "{manifest}", "{method}", "{seed}"}
        missing = sorted(x for x in required if x not in tasks[0].semantic_eval_command)
        if missing:
            checks.append(Check("ERROR", "semantic-evaluator", f"command missing placeholders: {missing}"))
        else:
            checks.append(Check("PASS", "semantic-evaluator", "frozen evaluator command has required placeholders"))

    schema = campaign.root / "contracts" / "weight_manifest.schema.json"
    arrays: Dict[str, np.ndarray] = {}
    sidecars: Dict[str, Dict] = {}
    for task in {t.method: t for t in tasks}.values():
        if task.method == "lt":
            if task.method_config.get("generated_weight") != "uniform_manifest":
                checks.append(Check("ERROR", "lt-sampler-control", "LT decisive arm must use uniform_manifest replacement sampling"))
            else:
                checks.append(Check("PASS", "lt-sampler-control", "LT uses uniform replacement sampling like weighted arms"))
            continue
        path_str = task.method_config.get("weight_file", "")
        if not path_str:
            checks.append(Check("ERROR", "critical-weight", f"{task.method}: empty weight file")); continue
        c, w, payload = _check_weight(Path(path_str), schema, len(labels), ids_hash, labels, task.method)
        checks.extend(c)
        if w is not None:
            arrays[task.method] = w; sidecars[task.method] = payload

    for method, should_use_fine in {"oracle": True, "predictive": False, "pointfit": False, "permutation": False}.items():
        p = sidecars.get(method, {})
        if p and bool(p.get("fine_labels_used_for_training", False)) != should_use_fine:
            checks.append(Check("ERROR", "fine-label-firewall", f"{method}: fine_labels_used_for_training must be {should_use_fine}"))
    if all(m in arrays for m in ("predictive", "permutation")):
        ok = True
        for c in np.unique(labels):
            a = arrays["predictive"][labels == c]; b = arrays["permutation"][labels == c]
            a = np.sort(a / a.mean()); b = np.sort(b / b.mean())
            if len(a) != len(b) or not np.allclose(a, b, rtol=1e-8, atol=1e-10):
                ok = False; break
        checks.append(Check("PASS" if ok else "ERROR", "matched-permutation",
                            "per-class normalized spectrum/ESS matched" if ok else "permutation does not match predictive spectrum within class"))
    descriptors = [(m, sidecars.get(m, {}).get("representation"), sidecars.get(m, {}).get("K"))
                   for m in ("predictive", "pointfit", "permutation") if m in sidecars]
    known_repr = {r for _, r, _ in descriptors if r is not None}
    known_k = {k for _, _, k in descriptors if k is not None}
    if len(known_repr) > 1 or len(known_k) > 1:
        checks.append(Check("ERROR", "estimator-lock", f"representation/K mismatch: {descriptors}"))
    elif descriptors:
        checks.append(Check("PASS", "estimator-lock", f"aligned descriptors: {descriptors}"))
    return checks


def run_preflight(campaign: LoadedCampaign) -> List[Check]:
    checks: List[Check] = []
    gpus = query_gpus()
    checks.append(Check("PASS" if gpus else "ERROR", "gpu",
                        ", ".join(f"{g.index}:{g.name} {g.memory_free_mb/1024:.1f}GB free" for g in gpus) if gpus else "nvidia-smi returned no GPUs"))
    target = Path(campaign.server["runtime"]["runs_root"])
    disk = shutil.disk_usage(target if target.exists() else campaign.root)
    free_gb = disk.free / 1024**3; guard = float(campaign.server.get("machine", {}).get("disk_stop_free_gb", 0))
    checks.append(Check("PASS" if free_gb >= guard else "ERROR", "disk", f"{free_gb:.1f} GB free; guard={guard:.1f} GB"))
    mode = campaign.server["runtime"].get("wandb_mode", os.getenv("WANDB_MODE", "online"))
    checks.append(Check("PASS" if mode != "online" or os.getenv("WANDB_API_KEY") else "ERROR", "wandb",
                        f"mode={mode}, project={campaign.server['runtime'].get('wandb_project','longtail')}" if mode != "online" or os.getenv("WANDB_API_KEY") else "online mode but WANDB_API_KEY is empty"))
    comparison = campaign.raw.get("comparison", {})
    candidate = str(comparison.get("candidate_method", "")).strip()
    requires_candidate = _as_bool(comparison.get("require_candidate_for_paper_claim", False))
    configured_methods = {task.method for task in campaign.tasks}
    if requires_candidate and not candidate:
        checks.append(Check("ERROR", "candidate-method", "paper claim requires LTX_CANDIDATE_METHOD, but it is empty"))
    elif candidate and candidate not in configured_methods:
        checks.append(Check("ERROR", "candidate-method", f"configured candidate {candidate!r} is not present in campaign methods {sorted(configured_methods)}"))
    elif candidate:
        checks.append(Check("PASS", "candidate-method", f"paired candidate={candidate}"))
    else:
        checks.append(Check("PASS", "candidate-method", "baseline-only report; superiority claims disabled"))

    repo_root = Path(campaign.server["runtime"]["repos_root"])
    vendor_manifest_path = repo_root / "THIRD_PARTY_MANIFEST.json"
    try:
        vendor_manifest = json.loads(vendor_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        vendor_manifest = {}
        checks.append(Check("ERROR", "vendor-manifest", f"{vendor_manifest_path}: {exc}"))
    for name, repo_cfg in campaign.raw.get("repositories", {}).items():
        path = repo_root / repo_cfg.get("directory", name)
        if not path.is_dir():
            checks.append(Check("ERROR", f"repo:{name}", f"missing vendored source {path}")); continue
        expected = str(repo_cfg.get("commit", "")).strip()
        actual = str(vendor_manifest.get("components", {}).get(name, {}).get("commit", ""))
        if expected and actual != expected:
            checks.append(Check("ERROR", f"repo:{name}", f"manifest commit={actual or 'missing'}, locked reproduction requires {expected}"))
        else:
            checks.append(Check("PASS", f"repo:{name}", f"vendored {actual or 'unversioned'}"))
    coral_dir = repo_root / campaign.raw.get("repositories", {}).get("coral", {}).get("directory", "coral")
    oc_dir = repo_root / campaign.raw.get("repositories", {}).get("oc", {}).get("directory", "oc")
    cm_dir = repo_root / campaign.raw.get("repositories", {}).get("cm", {}).get("directory", "cm")
    if coral_dir.exists() and not (coral_dir / ".ltx_weighted_sampler_patch_v2").exists():
        checks.append(Check("ERROR", "coral-patch", "v2 sampler/sample_only patch marker missing"))
    if oc_dir.exists() and not (oc_dir / ".ltx_seed_resume_patch_v2").exists():
        checks.append(Check("ERROR", "oc-patch", "v2 seed/resume patch marker missing"))
    cm_train = cm_dir / "tools" / "train.py"
    if cm_dir.exists() and (not cm_train.is_file() or "--ckpt_step" not in cm_train.read_text(encoding="utf-8")):
        checks.append(Check("ERROR", "cm-resume", "upstream --ckpt_step resume support is missing"))
    elif cm_dir.exists() and not all(marker in cm_train.read_text(encoding="utf-8") for marker in (
        "LTX_CM_RESUME_ZERO_STEP", "LTX_CM_RESUME_NEXT_STEP", "LTX_CM_RESUME_FIXED_NOISE",
        "LTX_CM_RESUME_RNG_CHECKPOINT", "LTX_CM_RESUME_RNG_RESTORE",
    )):
        checks.append(Check("ERROR", "cm-resume", "CM resume-next-step/fixed-noise/RNG patch is missing; run bootstrap"))
    elif cm_dir.exists():
        checks.append(Check("PASS", "cm-resume", "checkpoint resume starts at next update and restores fixed sampling noise/RNG state"))
    if campaign.raw.get("campaign", {}).get("paper_protocol") in {"cm_imagenet_lt_fid_kid", "cm_baselines_fid_kid"}:
        if not cm_dir.exists() or not (cm_dir / ".ltx_cm_imagenet_lt_patch_v1").is_file():
            checks.append(Check("ERROR", "cm-imagenet-port", "CM ImageNet-LT loader patch marker missing; run bootstrap"))
        imagenet_tasks = [task for task in campaign.tasks if task.dataset.get("data_type") == "imagenet_lt"]
        if not imagenet_tasks:
            checks.append(Check("ERROR", "imagenet-lt-tasks", "CM paper protocol requires ImageNet-LT tasks"))
        else:
            resolved: Dict[str, Path] = {}
            for key in ("root", "manifest", "reference_manifest"):
                values = {str(task.dataset.get(key, "")).strip() for task in imagenet_tasks}
                if not values or "" in values:
                    checks.append(Check("ERROR", f"imagenet-lt-{key}", f"set valid ImageNet-LT dataset.{key} / .env.local"))
                    continue
                if len(values) != 1:
                    checks.append(Check("ERROR", f"imagenet-lt-{key}", f"ImageNet-LT stages disagree on dataset.{key}: {sorted(values)}"))
                    continue
                path = Path(values.pop())
                valid = path.is_dir() if key == "root" else path.is_file()
                checks.append(Check("PASS" if valid else "ERROR", f"imagenet-lt-{key}", str(path) if valid else f"invalid path: {path}"))
                if valid:
                    resolved[key] = path
            if set(resolved) == {"root", "manifest", "reference_manifest"}:
                for key, require_balanced in (("manifest", False), ("reference_manifest", True)):
                    label = "train" if key == "manifest" else "reference"
                    try:
                        counts = _validate_imagenet_manifest(resolved["root"], resolved[key], label, require_balanced=require_balanced)
                        checks.append(Check("PASS", f"imagenet-lt-{label}-distribution",
                                            f"n={sum(counts.values())}, classes=1000, min/class={min(counts.values())}, max/class={max(counts.values())}"))
                    except Exception as exc:
                        checks.append(Check("ERROR", f"imagenet-lt-{label}-distribution", str(exc)))

    critical = [t for t in campaign.tasks if t.stage == campaign.raw.get("aggregation", {}).get("semantic_primary_stage", "decisive_semantic_gate")]
    if critical: checks.extend(_check_semantic_bundle(campaign, critical))
    if cm_dir.exists():
        cm_weight = cm_dir / "stats" / "pt_inception-2015-12-05-6726825d.pth"
        cm_weight_metadata = cm_weight.with_name(cm_weight.name + ".ltx.json")
        if not cm_weight.is_file():
            checks.append(Check("ERROR", "cm-stats", "CM Inception checkpoint missing; run scripts/prepare_cm_metric_assets.sh"))
        elif not cm_weight_metadata.is_file():
            checks.append(Check("ERROR", "cm-stats", f"missing pinned asset metadata: {cm_weight_metadata}"))
        else:
            try:
                metadata = json.loads(cm_weight_metadata.read_text(encoding="utf-8"))
                if int(metadata.get("bytes", -1)) != cm_weight.stat().st_size:
                    raise ValueError("asset byte count mismatch")
                if metadata.get("sha256") != sha256_file(cm_weight):
                    raise ValueError("asset sha256 mismatch")
                checks.append(Check("PASS", "cm-stats", f"pinned FID Inception sha256={metadata['sha256'][:12]}"))
            except Exception as exc:
                checks.append(Check("ERROR", "cm-stats", str(exc)))
    if oc_dir.exists() and not (oc_dir / "stats").exists():
        checks.append(Check("WARN", "oc-stats", "OC stats symlink missing"))
    ids = [t.id for t in campaign.tasks]
    checks.append(Check("PASS" if len(ids) == len(set(ids)) else "ERROR", "task-ids", f"{len(ids)} unique tasks" if len(ids) == len(set(ids)) else "duplicate task IDs"))
    checks.extend(_check_coral_metric_assets(campaign))
    checks.extend(_check_coral2025_metric_protocol(campaign, repo_root))
    return checks


def _check_coral2025_metric_protocol(campaign: LoadedCampaign, repo_root: Path) -> List[Check]:
    """Do not spend a multi-day budget on a table that cannot meet the paper metric spec."""
    if campaign.raw.get("campaign", {}).get("paper_protocol") != "coral2025_table1_cifar":
        return []
    checks: List[Check] = []
    evaluator = campaign.root / "tools" / "evaluate_coral2025.py"
    if evaluator.is_file():
        checks.append(Check("PASS", "paper-improved-prd", "shared VGG16 fc2 / k=3 evaluator present"))
    else:
        checks.append(Check("ERROR", "paper-improved-prd", f"missing shared evaluator {evaluator}"))
    if any(task.adapter == "oc" for task in campaign.tasks):
        oc = repo_root / campaign.raw["repositories"]["oc"].get("directory", "oc")
        if (oc / ".ltx_oc_sample_export_v1").exists():
            checks.append(Check("PASS", "paper-t2h-metrics", "T2H exports generated arrays/labels to the shared five-metric evaluator"))
        else:
            checks.append(Check("ERROR", "paper-t2h-metrics", "T2H sample-export patch marker missing"))
    return checks


def _check_coral_metric_assets(campaign: LoadedCampaign) -> List[Check]:
    """Fail before training when a requested paper metric cannot be computed."""
    tasks = [t for t in campaign.tasks if t.adapter == "coral" and (t.eval.get("standard_metrics", True) or t.eval.get("paper_metrics", False))]
    if not tasks:
        return []
    repo_cfg = campaign.raw.get("repositories", {}).get("coral", {})
    repo = Path(campaign.server["runtime"]["repos_root"]) / repo_cfg.get("directory", "coral")
    required_source = ("dataset.py", "score/both.py", "score/improved_prd.py", "utils/augmentation.py", "loss_tracker.py")
    missing_source = [item for item in required_source if not (repo / item).exists()]
    checks: List[Check] = []
    if missing_source:
        checks.append(Check("ERROR", "coral-source-overlay", f"missing required CORAL/CBDM compatibility files: {missing_source}"))
        return checks
    checks.append(Check("PASS", "coral-source-overlay", "CORAL imports are backed by the pinned CBDM compatibility overlay"))

    metrics_root = Path(os.environ.get("LTX_METRICS_ROOT", str(repo / "stats"))).expanduser()
    wanted = {str(t.dataset.get("data_type", "")) for t in tasks}
    names = set()
    if "cifar10" in wanted or "cifar10lt" in wanted:
        names.update(("cifar10.train.npz", "cifar10_feats.npy", "cifar10_vgg16_fc2.npy", "cifar10_vgg16_fc2_k3_radii.npy"))
    if "cifar100" in wanted or "cifar100lt" in wanted:
        names.update(("cifar100.train.npz", "cifar100_feats.npy", "cifar100_vgg16_fc2.npy", "cifar100_vgg16_fc2_k3_radii.npy"))
    missing = sorted(name for name in names if not (metrics_root / name).is_file())
    if missing:
        checks.append(Check("ERROR", "paper-metric-assets", f"{metrics_root}: missing balanced-reference assets {missing}"))
    else:
        checks.append(Check("PASS", "paper-metric-assets", f"balanced FID/PRD references present in {metrics_root}"))
    return checks
