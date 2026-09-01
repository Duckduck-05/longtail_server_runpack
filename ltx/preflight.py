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
from .completion import check_campaign_complete
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


def _check_secondary_imagenet_gate(campaign: LoadedCampaign) -> List[Check]:
    """Require an explicit ACCESS hand-off or a completed main-table proof."""
    gate = os.environ.get("LTX_IMAGENET_LT_GATE", "").strip().lower()
    if gate == "access":
        return [Check("PASS", "imagenet-lt-gate", "explicit ACCESS hand-off acknowledged")]
    if gate != "main_complete":
        return [Check(
            "ERROR", "imagenet-lt-gate",
            "set LTX_IMAGENET_LT_GATE=access on ACCESS or main_complete after the CIFAR main table is complete",
        )]

    configured = campaign.raw.get("secondary_gate", {}).get("main_config", "configs/unified_cifar_c100.yaml")
    config_path = Path(configured).expanduser()
    if not config_path.is_absolute():
        config_path = (campaign.root / config_path).resolve()
    try:
        main, incomplete = check_campaign_complete(config_path)
    except Exception as exc:
        return [Check("ERROR", "imagenet-lt-main-table", f"cannot load main-table config {config_path}: {exc}")]
    if incomplete:
        preview = "; ".join(incomplete[:8])
        if len(incomplete) > 8:
            preview += f"; ... ({len(incomplete)} incomplete)"
        return [Check("ERROR", "imagenet-lt-main-table", f"main table is not complete: {preview}")]
    return [Check("PASS", "imagenet-lt-main-table", f"verified {len(main.tasks)} main-table tasks with SUCCESS and FID")]


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
    used_adapters = {task.adapter for task in campaign.tasks}
    coral_dir = repo_root / campaign.raw.get("repositories", {}).get("coral", {}).get("directory", "coral")
    oc_dir = repo_root / campaign.raw.get("repositories", {}).get("oc", {}).get("directory", "oc")
    cm_dir = repo_root / campaign.raw.get("repositories", {}).get("cm", {}).get("directory", "cm")
    if "coral" in used_adapters and coral_dir.exists() and not (coral_dir / ".ltx_weighted_sampler_patch_v2").exists():
        checks.append(Check("ERROR", "coral-patch", "v2 sampler/sample_only patch marker missing"))
    coral_main = coral_dir / "main.py"
    if "coral" in used_adapters and coral_dir.exists() and (
        not coral_main.is_file()
        or "resume_checkpoint" not in coral_main.read_text(encoding="utf-8")
        or "allow_non_exact_resume" not in coral_main.read_text(encoding="utf-8")
    ):
        checks.append(Check("ERROR", "coral-resume", "explicit external/full-state resume support is missing; pull the current runpack and run bootstrap"))
    elif coral_dir.exists():
        checks.append(Check("PASS", "coral-resume", "local and explicit checkpoints are loaded with EMA-only warm starts opt-in"))
    if "oc" in used_adapters and oc_dir.exists() and not (oc_dir / ".ltx_seed_resume_patch_v2").exists():
        checks.append(Check("ERROR", "oc-patch", "v2 seed/resume patch marker missing"))
    uniform_patch = repo_root / ".ltx_uniform_eval_labels_patch_v1"
    if {"coral", "oc"} & used_adapters and coral_dir.exists() and oc_dir.exists() and not uniform_patch.exists():
        checks.append(Check("ERROR", "uniform-eval-labels", "exact class-uniform evaluation-label patch marker missing"))
    elif {"coral", "oc"} & used_adapters and coral_dir.exists() and oc_dir.exists():
        checks.append(Check("PASS", "uniform-eval-labels", "all conditional methods use exact class-uniform 50k labels"))
    cm_train = cm_dir / "tools" / "train.py"
    if "cm" in used_adapters and cm_dir.exists() and (not cm_train.is_file() or "--ckpt_step" not in cm_train.read_text(encoding="utf-8")):
        checks.append(Check("ERROR", "cm-resume", "upstream --ckpt_step resume support is missing"))
    elif "cm" in used_adapters and cm_dir.exists() and not all(marker in cm_train.read_text(encoding="utf-8") for marker in (
        "LTX_CM_RESUME_ZERO_STEP", "LTX_CM_RESUME_NEXT_STEP", "LTX_CM_RESUME_FIXED_NOISE",
        "LTX_CM_RESUME_RNG_CHECKPOINT", "LTX_CM_RESUME_RNG_RESTORE",
    )):
        checks.append(Check("ERROR", "cm-resume", "CM resume-next-step/fixed-noise/RNG patch is missing; run bootstrap"))
    elif "cm" in used_adapters and cm_dir.exists():
        checks.append(Check("PASS", "cm-resume", "checkpoint resume starts at next update and restores fixed sampling noise/RNG state"))
    if "cm" in used_adapters and cm_dir.exists() and any(task.adapter == "cm" and task.eval.get("metric_protocol") == "unified_cifar_v1" for task in campaign.tasks):
        if not (cm_dir / ".ltx_cm_array_export_patch_v1").exists():
            checks.append(Check("ERROR", "cm-unified-export", "CM direct sample-array export patch marker missing"))
        else:
            checks.append(Check("PASS", "cm-unified-export", "CM exports float32 arrays directly to the shared evaluator"))
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

    if campaign.raw.get("campaign", {}).get("paper_protocol") == "imagenet_lt_secondary_fid_kid":
        checks.extend(_check_secondary_imagenet_gate(campaign))
        t2h_dir = repo_root / campaign.raw.get("repositories", {}).get("t2h_unified", {}).get("directory", "T2H-unified")
        ccua_dir = repo_root / campaign.raw.get("repositories", {}).get("ccua", {}).get("directory", "CCUA-DDPM")
        imagenet_tasks = [task for task in campaign.tasks if task.dataset.get("data_type") == "imagenet_lt"]
        if not imagenet_tasks:
            checks.append(Check("ERROR", "imagenet-lt-tasks", "secondary ImageNet-LT protocol requires ImageNet-LT tasks"))
        else:
            expected = {("ddpm", 0), ("ccua", 0)}
            actual = {(task.method, int(task.seed)) for task in imagenet_tasks}
            if actual != expected or len(imagenet_tasks) != len(expected):
                checks.append(Check("ERROR", "imagenet-lt-contract", f"expected DDPM/CCUA seed 0 exactly once, found {sorted(actual)}"))
            else:
                checks.append(Check("PASS", "imagenet-lt-contract", "DDPM + CCUA, seed 0, exactly two secondary tasks"))
            ddpm_tasks = [task for task in imagenet_tasks if task.method == "ddpm"]
            ccua_tasks = [task for task in imagenet_tasks if task.method == "ccua"]
            common_host = any(task.adapter == "t2h_unified" for task in imagenet_tasks)
            expected_adapter = "t2h_unified" if common_host else None
            if common_host:
                if any(task.adapter != expected_adapter for task in imagenet_tasks):
                    checks.append(Check("ERROR", "imagenet-lt-adapter", "all ImageNet-LT rows must use the T2H-unified host"))
                required = ("unified_main.py", "ltx_manifest_dataset.py", "metrics.py", "score/inception.py")
                missing = [item for item in required if not (t2h_dir / item).is_file()]
                if missing:
                    checks.append(Check("ERROR", "t2h-imagenet-port", f"T2H ImageNet-LT host files missing: {missing}"))
                else:
                    host_source = (t2h_dir / "unified_main.py").read_text(encoding="utf-8")
                    if "t2h-unified-common-v2" not in host_source:
                        checks.append(Check("ERROR", "t2h-imagenet-port", "T2H host revision is not the pinned common revision"))
                    else:
                        checks.append(Check("PASS", "t2h-imagenet-port", "T2H host reads the pinned ImageNet-LT manifest and owns FID/KID"))
            else:
                if ddpm_tasks and any(task.adapter != "cm" for task in ddpm_tasks):
                    checks.append(Check("ERROR", "imagenet-lt-ddpm-adapter", "DDPM must use CM's native ImageNet-LT OC/transfer-off route"))
                if ddpm_tasks:
                    if not (cm_dir / ".ltx_cm_imagenet_lt_patch_v1").is_file():
                        checks.append(Check("ERROR", "cm-imagenet-port", "CM ImageNet-LT loader patch marker missing; run bootstrap"))
                    else:
                        checks.append(Check("PASS", "cm-imagenet-port", "DDPM uses CM's manifest-backed ImageNet-LT loader"))
                if ccua_tasks:
                    if not (ccua_dir / ".ltx_ccua_imagenet_lt_patch_v1").is_file():
                        checks.append(Check("ERROR", "ccua-imagenet-port", "CCUA ImageNet-LT manifest loader patch marker missing; run bootstrap"))
                    else:
                        checks.append(Check("PASS", "ccua-imagenet-port", "CCUA reads the pinned ImageNet-LT train manifest"))
                    if not (ccua_dir / ".ltx_ccua_sample_export_v1").is_file():
                        checks.append(Check("ERROR", "ccua-sample-export", "CCUA sample-output/uniform-label patch marker missing; run bootstrap"))
                    else:
                        checks.append(Check("PASS", "ccua-sample-export", "CCUA exports arrays and exact class-uniform labels"))
            setting_errors: list[str] = []
            reference_task = imagenet_tasks[0]

            def same_setting(left, right) -> bool:
                if isinstance(left, (float, int)) and isinstance(right, (float, int)):
                    return bool(np.isclose(float(left), float(right)))
                return left == right

            common_train_keys = (
                "total_steps", "inclusive_final_step", "batch_size", "lr", "warmup", "T",
                "beta_1", "beta_T", "var_type", "dropout", "grad_clip", "ch", "ch_mult",
                "attn", "num_res_blocks", "ema_decay", "conditional", "cfg", "amp",
                "checkpoint_prefix",
            )
            common_eval_keys = (
                "checkpoint_step", "num_images", "sample_batch_size", "guidance_scale",
                "sample_method", "ddim_skip_step", "uniform_labels", "sampler_family",
                "metric_protocol", "artifact_namespace",
            )
            for task in imagenet_tasks:
                prefix = f"{task.method}/seed{task.seed}"
                expected_adapter = "t2h_unified" if common_host else {"ddpm": "cm", "ccua": "ccua"}.get(task.method)
                if task.adapter != expected_adapter: setting_errors.append(f"{prefix}: adapter={task.adapter}")
                if int(task.dataset.get("img_size", -1)) != 64: setting_errors.append(f"{prefix}: img_size")
                if int(task.dataset.get("num_class", task.dataset.get("num_classes", -1))) != 1000: setting_errors.append(f"{prefix}: num_classes")
                if int(task.train.get("total_steps", -1)) != 300000: setting_errors.append(f"{prefix}: total_steps")
                if int(task.train.get("batch_size", -1)) != 256: setting_errors.append(f"{prefix}: batch_size")
                if int(task.eval.get("checkpoint_step", -1)) != 300000: setting_errors.append(f"{prefix}: checkpoint_step")
                if int(task.eval.get("num_images", -1)) != 50000: setting_errors.append(f"{prefix}: num_images")
                if not task.eval.get("uniform_labels", False): setting_errors.append(f"{prefix}: uniform_labels")
                if common_host:
                    for key in common_train_keys:
                        if not same_setting(task.train.get(key), reference_task.train.get(key)):
                            setting_errors.append(f"{prefix}: train.{key} differs from common host")
                    for key in common_eval_keys:
                        if not same_setting(task.eval.get(key), reference_task.eval.get(key)):
                            setting_errors.append(f"{prefix}: eval.{key} differs from common host")
            if setting_errors:
                checks.append(Check("ERROR", "imagenet-lt-controls", "; ".join(setting_errors)))
            else:
                checks.append(Check("PASS", "imagenet-lt-controls", "64x64, batch target 256, 300k endpoint, 50k uniform samples"))
            resolved: Dict[str, Path] = {}
            for key in ("root", "manifest", "reference_manifest"):
                values = {str(task.dataset.get(key, "")).strip() for task in imagenet_tasks}
                if not values or "" in values:
                    checks.append(Check("ERROR", f"imagenet-lt-{key}", f"set valid ImageNet-LT dataset.{key} / .env.local"))
                    continue
                if len(values) != 1:
                    checks.append(Check("ERROR", f"imagenet-lt-{key}", f"ImageNet-LT tasks disagree on dataset.{key}: {sorted(values)}"))
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
    needs_cm_native_metrics = any(
        task.adapter == "cm" and task.eval.get("metric_protocol") != "unified_cifar_v1"
        for task in campaign.tasks
    ) or (
        campaign.raw.get("campaign", {}).get("paper_protocol") == "imagenet_lt_secondary_fid_kid"
        and not any(task.adapter == "t2h_unified" for task in campaign.tasks)
    )
    if cm_dir.exists() and needs_cm_native_metrics:
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
    if "oc" in used_adapters and oc_dir.exists() and not (oc_dir / "stats").exists():
        checks.append(Check("WARN", "oc-stats", "OC stats symlink missing"))
    ids = [t.id for t in campaign.tasks]
    checks.append(Check("PASS" if len(ids) == len(set(ids)) else "ERROR", "task-ids", f"{len(ids)} unique tasks" if len(ids) == len(set(ids)) else "duplicate task IDs"))
    checks.extend(_check_coral_metric_assets(campaign))
    checks.extend(_check_coral2025_metric_protocol(campaign, repo_root))
    checks.extend(_check_native_cifar_contract(campaign, repo_root))
    checks.extend(_check_unified_cifar_contract(campaign, repo_root))
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
        if not (oc / ".ltx_oc_compiled_ckpt_patch_v1").exists():
            checks.append(Check("ERROR", "paper-t2h-eval-ckpt", "T2H evaluator cannot load a torch.compile'd checkpoint; run bootstrap"))
    return checks


def _check_unified_cifar_contract(campaign: LoadedCampaign, repo_root: Path) -> List[Check]:
    """Validate the new comparison before any expensive source-native run.

    This intentionally validates equality of the controllable factors, rather
    than pretending the five independently released method implementations are
    byte-identical.  Any later YAML edit that reintroduces a hidden fourth seed,
    different training budget, random label support, or a duplicate OC/T2H row
    fails at launch time.
    """
    if campaign.raw.get("campaign", {}).get("protocol") != "unified_cifar_v1":
        return []
    checks: List[Check] = []
    contract = campaign.raw.get("fairness_contract", {})
    expected_cells = set(contract.get("cells", ()))
    expected_methods = set(contract.get("methods", ()))
    expected_seeds = sorted(map(int, contract.get("seeds", ())))
    # The IP-SVT arms deliberately share the ddpm baseline's adapter: they are
    # the same trainer with auxiliary flags, which is what makes the comparison
    # a comparison of objectives rather than of codebases.
    common_t2h_host = any(task.adapter == "t2h_unified" for task in campaign.tasks)
    if common_t2h_host:
        expected_adapters = {method: "t2h_unified" for method in expected_methods}
    else:
        expected_adapters = {"ddpm": "coral", "cbdm": "coral", "coral": "coral", "t2h": "oc",
                             "cm": "cm", "ccua": "ccua",
                             "ipsvt": "coral", "ipsvt_twin": "coral", "ipsvt_clean": "coral"}

    methods = {task.method for task in campaign.tasks}
    cells = {str(task.dataset.get("name")) for task in campaign.tasks}
    if methods != expected_methods:
        checks.append(Check("ERROR", "unified-methods", f"expected {sorted(expected_methods)}, found {sorted(methods)}"))
    elif "oc" in methods:
        checks.append(Check("ERROR", "unified-methods", "OC is an alias of T2H and must not appear as a second row"))
    else:
        checks.append(Check("PASS", "unified-methods",
                            f"{'/'.join(m.upper() for m in sorted(expected_methods))} exactly once per cell"))
    if cells != expected_cells:
        checks.append(Check("ERROR", "unified-cells", f"expected {sorted(expected_cells)}, found {sorted(cells)}"))

    errors: list[str] = []
    by_cell: Dict[str, list] = {}
    for cell in sorted(cells):
        cell_tasks = [task for task in campaign.tasks if task.dataset.get("name") == cell]
        by_cell[cell] = cell_tasks
        if {task.method for task in cell_tasks} != expected_methods:
            errors.append(f"{cell}: methods={sorted({task.method for task in cell_tasks})}")
        for method in expected_methods:
            seeds = sorted(task.seed for task in cell_tasks if task.method == method)
            if seeds != expected_seeds:
                errors.append(f"{cell}/{method}: seeds={seeds}")
    if errors:
        checks.append(Check("ERROR", "unified-matrix", "; ".join(errors)))
    else:
        checks.append(Check("PASS", "unified-matrix", f"{len(expected_cells)} cells × {len(expected_methods)} methods × {len(expected_seeds)} seeds = {len(campaign.tasks)} tasks"))

    settings_errors: list[str] = []
    updates = int(contract.get("train_updates", -1))
    batch = int(contract.get("train_batch_size", -1))
    lr = float(contract.get("learning_rate", -1))
    diffusion_steps = int(contract.get("diffusion_steps", -1))
    generated = int(contract.get("generated_images", -1))
    expected_host_revision = str(contract.get("host_revision", "")).strip()
    expected_guidance = contract.get("guidance_scale")
    expected_checkpoint_prefix = str(contract.get("checkpoint_prefix", "")).strip()
    expected_artifact_namespace = str(contract.get("artifact_namespace", "")).strip()

    def effective(section: dict, key: str, default):
        return section[key] if key in section else default

    reference_task = campaign.tasks[0] if campaign.tasks else None
    common_train_keys = {
        "warmup": 5000, "dropout": 0.1, "grad_clip": 1.0,
        "beta_1": 0.0001, "beta_T": 0.02, "var_type": "fixedlarge",
        "conditional": True, "cfg": True, "amp": False,
        "inclusive_final_step": True, "checkpoint_prefix": expected_checkpoint_prefix,
    }
    common_eval_keys = {
        "sample_batch_size": batch, "guidance_scale": expected_guidance,
        "uniform_labels": True, "sample_method": "ddim", "ddim_skip_step": 10,
        "artifact_namespace": expected_artifact_namespace,
        "metrics_file": "metrics.unified.v2.json",
        "per_class_metrics_file": "metrics.per_class.v2.json",
        "longtail_groups": "cm_three_way",
    }
    for task in campaign.tasks:
        prefix = f"{task.dataset.get('name')}/{task.method}"
        if task.adapter != expected_adapters.get(task.method): settings_errors.append(f"{prefix}: adapter={task.adapter}")
        if int(task.train.get("total_steps", -1)) != updates: settings_errors.append(f"{prefix}: updates")
        if int(task.train.get("batch_size", -1)) != batch: settings_errors.append(f"{prefix}: batch")
        if not np.isclose(float(task.train.get("lr", -1)), lr): settings_errors.append(f"{prefix}: lr")
        if int(task.train.get("T", -1)) != diffusion_steps: settings_errors.append(f"{prefix}: T")
        if int(task.eval.get("num_images", -1)) != generated: settings_errors.append(f"{prefix}: num_images")
        if not task.eval.get("uniform_labels", False): settings_errors.append(f"{prefix}: uniform_labels")
        if expected_guidance is not None and not np.isclose(float(task.eval.get("guidance_scale", -1)), float(expected_guidance)):
            settings_errors.append(f"{prefix}: guidance_scale")
        if task.eval.get("sampler_family") != contract.get("sampler_family"): settings_errors.append(f"{prefix}: sampler")
        if task.eval.get("metric_protocol") != contract.get("metric_protocol"): settings_errors.append(f"{prefix}: metrics")
        if int(task.dataset.get("split_seed", -1)) != 0: settings_errors.append(f"{prefix}: split_seed")
        # The backbone was previously matched only because all four repos
        # happened to share the same flag defaults; nothing detected a drift.
        for contract_key, train_key in (("unet_ch", "ch"), ("unet_ch_mult", "ch_mult"),
                                        ("unet_attn", "attn"), ("unet_num_res_blocks", "num_res_blocks"),
                                        ("ema_decay", "ema_decay")):
            expected_value = contract.get(contract_key)
            if expected_value is not None and task.train.get(train_key) != expected_value:
                settings_errors.append(f"{prefix}: {train_key}={task.train.get(train_key)} != {expected_value}")
        if int(task.train.get("save_step", 0)) <= 0: settings_errors.append(f"{prefix}: save_step")
        if not task.eval.get("kid", False): settings_errors.append(f"{prefix}: KID")
        if not task.eval.get("per_class_metrics_file"): settings_errors.append(f"{prefix}: per-class FID")
        if task.eval.get("longtail_groups") != "cm_three_way": settings_errors.append(f"{prefix}: Many/Medium/Few grouping")
        if expected_checkpoint_prefix and task.train.get("checkpoint_prefix", expected_checkpoint_prefix) != expected_checkpoint_prefix:
            settings_errors.append(f"{prefix}: checkpoint namespace")
        if expected_artifact_namespace and task.eval.get("artifact_namespace", expected_artifact_namespace) != expected_artifact_namespace:
            settings_errors.append(f"{prefix}: artifact namespace")
        for key, default in common_train_keys.items():
            if reference_task is not None and effective(task.train, key, default) != effective(reference_task.train, key, default):
                settings_errors.append(f"{prefix}: train {key} differs from common host")
        for key, default in common_eval_keys.items():
            if reference_task is not None and effective(task.eval, key, default) != effective(reference_task.eval, key, default):
                settings_errors.append(f"{prefix}: eval {key} differs from common host")
        if task.adapter == "cm" and not task.train.get("inclusive_final_step", False): settings_errors.append(f"{prefix}: CM inclusive endpoint")
        # The contract fixes ONE sampler for every method; which one is a
        # protocol choice, and it is now DDIM-100. cm/oc/ccua express that as
        # skip = T/steps, the coral-family repos as an explicit step count, so
        # both spellings are checked against the same contract value.
        family = str(contract.get("sampler_family", ""))
        if family.startswith("ddim_"):
            want_steps = int(family.split("_", 1)[1])
            if task.adapter in {"cm", "oc", "ccua", "t2h_unified"}:
                skip = int(task.eval.get("ddim_skip_step", -1))
                got = diffusion_steps // skip if skip > 0 else -1
                if task.eval.get("sample_method") != "ddim" or got != want_steps:
                    settings_errors.append(f"{prefix}: sampler steps={got} != {want_steps}")
            else:
                if int(task.eval.get("ddim_steps", -1)) != want_steps:
                    settings_errors.append(f"{prefix}: ddim_steps={task.eval.get('ddim_steps')} != {want_steps}")
        elif task.adapter in {"cm", "oc", "ccua", "t2h_unified"} and int(task.eval.get("ddim_skip_step", -1)) != 1:
            settings_errors.append(f"{prefix}: ancestral step")
    if settings_errors:
        checks.append(Check("ERROR", "unified-controls", "; ".join(settings_errors)))
    else:
        checks.append(Check("PASS", "unified-controls", f"{updates} updates, batch={batch}, lr={lr:g}, T={diffusion_steps}, N={generated}, common evaluator + KID + tail FID"))

    evaluator = campaign.root / "tools" / "evaluate_coral2025.py"
    if evaluator.is_file():
        checks.append(Check("PASS", "unified-evaluator", "shared FID/IS/PRD/improved-PRD evaluator present"))
    else:
        checks.append(Check("ERROR", "unified-evaluator", f"missing {evaluator}"))
    if common_t2h_host:
        host = repo_root / campaign.raw["repositories"]["t2h_unified"].get("directory", "T2H-unified")
        required = ("unified_main.py", "unified_objectives.py", "ipsvt_aux.py", "model/model_cm.py")
        missing = [item for item in required if not (host / item).is_file()]
        if missing:
            checks.append(Check("ERROR", "unified-t2h-host", f"missing host files: {missing}"))
        else:
            source = (host / "unified_main.py").read_text(encoding="utf-8")
            if expected_host_revision and expected_host_revision not in source:
                checks.append(Check("ERROR", "unified-t2h-host",
                                    f"host revision {expected_host_revision!r} is not present in unified_main.py"))
            else:
                checks.append(Check("PASS", "unified-t2h-host",
                                    "all methods dispatch through the pinned T2H host revision"))
    elif any(task.adapter == "oc" for task in campaign.tasks):
        oc = repo_root / campaign.raw["repositories"]["oc"].get("directory", "OC_LT")
        if not (oc / ".ltx_oc_sample_export_v1").exists():
            checks.append(Check("ERROR", "unified-t2h-export", "T2H generated-array export patch missing"))
        if not (oc / ".ltx_oc_compiled_ckpt_patch_v1").exists():
            checks.append(Check("ERROR", "unified-t2h-eval-ckpt",
                                "T2H evaluator cannot load a torch.compile'd checkpoint; run bootstrap"))
    if any(task.adapter == "ccua" for task in campaign.tasks):
        ccua = repo_root / campaign.raw.get("repositories", {}).get("ccua", {}).get("directory", "CCUA-DDPM")
        if not (ccua / ".ltx_ccua_sample_export_v1").exists():
            checks.append(Check("ERROR", "unified-ccua-export",
                                "CCUA sample-output/uniform-label patch missing; run bootstrap"))
        else:
            checks.append(Check("PASS", "unified-ccua-export",
                                "CCUA exports arrays to the shared evaluator with class-uniform labels"))
    return checks


def _check_native_cifar_contract(campaign: LoadedCampaign, repo_root: Path | None = None) -> List[Check]:
    """Validate the CCUA-DDPM-backed native CIFAR baseline before GPU work.

    The native contract deliberately has one adapter/repository for all three
    objectives.  Objective-specific behavior is dispatched by
    :class:`CCUAAdapter`; this gate keeps a YAML edit from silently restoring
    the old Coral/CBDM split or changing the official CCUA sampler for only one
    row.
    """
    if campaign.raw.get("campaign", {}).get("protocol") != "native_cifar_v1":
        return []

    checks: List[Check] = []
    contract = campaign.raw.get("native_contract", {})
    expected_methods = set(contract.get("methods", ("ddpm", "cbdm", "ccua")))
    expected_seeds = sorted(map(int, contract.get("seeds", campaign.raw.get("campaign", {}).get("paired_seeds", [0, 1, 2]))))
    tasks = campaign.tasks
    actual_methods = {task.method for task in tasks}
    actual_seeds = sorted({int(task.seed) for task in tasks})
    expected_pairs = {(method, seed) for method in expected_methods for seed in expected_seeds}
    actual_pairs = {(task.method, int(task.seed)) for task in tasks}
    if len(tasks) != len(expected_pairs) or actual_pairs != expected_pairs:
        checks.append(Check("ERROR", "native-cifar-matrix", f"expected {sorted(expected_pairs)}, found {sorted(actual_pairs)}"))
    else:
        checks.append(Check("PASS", "native-cifar-matrix", f"{len(expected_methods)} methods × {len(expected_seeds)} seeds = {len(tasks)} tasks"))
    if actual_methods != expected_methods or actual_seeds != expected_seeds:
        checks.append(Check("ERROR", "native-cifar-methods", f"methods={sorted(actual_methods)} seeds={actual_seeds}"))

    expected_adapter = str(contract.get("adapter", "ccua")).strip().lower()
    expected_repository = str(contract.get("repository", "ccua")).strip().lower()
    adapter_errors = []
    if expected_adapter != "ccua":
        adapter_errors.append(f"native_contract.adapter={expected_adapter!r}, expected='ccua'")
    if expected_repository != "ccua":
        adapter_errors.append(f"native_contract.repository={expected_repository!r}, expected='ccua'")
    for task in tasks:
        if task.adapter != expected_adapter:
            adapter_errors.append(f"{task.method}/seed{task.seed}: adapter={task.adapter}, expected={expected_adapter}")
        if task.repository.get("directory") != "CCUA-DDPM":
            adapter_errors.append(
                f"{task.method}/seed{task.seed}: repository={task.repository.get('directory')!r}, expected='CCUA-DDPM'"
            )
    checks.append(Check("ERROR" if adapter_errors else "PASS", "native-cifar-adapters",
                        "; ".join(adapter_errors) if adapter_errors else "all objectives use the CCUA adapter and CCUA-DDPM repository"))

    objective_errors: list[str] = []
    for task in tasks:
        prefix = f"{task.method}/seed{task.seed}"
        objective = str(task.method_config.get("objective", task.method)).strip().lower()
        if objective not in {"ddpm", "cbdm", "ccua"} or objective != task.method:
            objective_errors.append(f"{prefix}: objective={objective!r}, expected {task.method!r}")
            continue
        if objective == "cbdm":
            try:
                cb_tau = float(task.method_config.get("cb_tau", task.method_config.get("tau", 1.0)))
            except (TypeError, ValueError):
                cb_tau = -1.0
            if not np.isfinite(cb_tau) or cb_tau <= 0:
                objective_errors.append(f"{prefix}: cb_tau must be finite and > 0")
        elif objective in {"ddpm", "cbdm"}:
            for key in ("ccua_al", "ccua_ucl"):
                try:
                    value = float(task.method_config.get(key, 0.0))
                except (TypeError, ValueError):
                    value = 1.0
                if value != 0.0:
                    objective_errors.append(f"{prefix}: {key} must be zero for {objective}")
        if objective == "ccua":
            for key in ("ccua_al", "ccua_ucl"):
                try:
                    value = float(task.method_config.get(key, 1.0))
                except (TypeError, ValueError):
                    value = 0.0
                if not np.isfinite(value) or value <= 0:
                    objective_errors.append(f"{prefix}: {key} must be finite and > 0 for ccua")
    checks.append(Check("ERROR" if objective_errors else "PASS", "native-cifar-objectives",
                        "; ".join(objective_errors) if objective_errors else "DDPM/CBDM/CCUA objective dispatch is explicit"))

    if repo_root is None:
        runtime_root = campaign.server.get("runtime", {}).get("repos_root")
        repo_root = Path(runtime_root) if runtime_root else None
    ccua_cfg = campaign.raw.get("repositories", {}).get("ccua", {})
    ccua_directory = str(ccua_cfg.get("directory", "CCUA-DDPM"))
    if "ccua" not in campaign.raw.get("repositories", {}):
        checks.append(Check("ERROR", "native-cifar-repository", "native contract must declare repositories.ccua"))
    elif ccua_directory != "CCUA-DDPM":
        checks.append(Check("ERROR", "native-cifar-repository",
                            f"repositories.ccua.directory={ccua_directory!r}, expected 'CCUA-DDPM'"))
    elif repo_root is not None and (repo_root / ccua_directory).is_dir():
        ccua_source = repo_root / ccua_directory / "main.py"
        try:
            source = ccua_source.read_text(encoding="utf-8")
        except OSError as exc:
            checks.append(Check("ERROR", "native-cifar-repository", f"cannot read {ccua_source}: {exc}"))
        else:
            required_markers = (
                "GaussianDiffusionSamplerDDIM",
                "flags.DEFINE_bool('sample'",
                "if FLAGS.sample:",
            )
            missing = [marker for marker in required_markers if marker not in source]
            sample_export = repo_root / ccua_directory / ".ltx_ccua_sample_export_v1"
            if missing:
                checks.append(Check("ERROR", "native-cifar-repository",
                                    f"CCUA source is missing official sample entrypoint markers: {missing}"))
            elif not sample_export.is_file():
                checks.append(Check("ERROR", "native-cifar-repository",
                                    "CCUA sample-output/uniform-label patch marker missing; run bootstrap"))
            else:
                checks.append(Check("PASS", "native-cifar-repository",
                                    "CCUA-DDPM main.py exposes the official DDIM sampler and array export"))

    def close(value: object, expected: float) -> bool:
        try:
            return bool(np.isclose(float(value), expected))
        except (TypeError, ValueError):
            return False

    setting_errors: list[str] = []
    updates = int(contract.get("train_updates", 300000))
    batch = int(contract.get("train_batch_size", 64))
    lr = float(contract.get("learning_rate", 2e-4))
    diffusion_steps = int(contract.get("diffusion_steps", 1000))
    generated = int(contract.get("generated_images", 50000))
    guidance = float(contract.get("guidance_scale", 1.5))
    wanted_family = str(contract.get("sampler_family", "ddim_100"))
    wanted_steps = int(wanted_family.rsplit("_", 1)[-1]) if wanted_family.startswith("ddim_") else -1
    for task in tasks:
        prefix = f"{task.method}/seed{task.seed}"
        if task.dataset.get("name") != contract.get("dataset", "cifar100lt_if100"):
            setting_errors.append(f"{prefix}: dataset.name")
        if task.dataset.get("data_type") != "cifar100lt":
            setting_errors.append(f"{prefix}: dataset.data_type")
        if int(task.dataset.get("num_class", task.dataset.get("num_classes", -1))) != 100:
            setting_errors.append(f"{prefix}: dataset.num_class")
        if int(task.dataset.get("img_size", -1)) != 32:
            setting_errors.append(f"{prefix}: dataset.img_size")
        if not close(task.dataset.get("imbalance_factor"), 0.01):
            setting_errors.append(f"{prefix}: dataset.imbalance_factor")
        if int(task.dataset.get("split_seed", -1)) != 0:
            setting_errors.append(f"{prefix}: dataset.split_seed")
        if int(task.train.get("total_steps", -1)) != updates:
            setting_errors.append(f"{prefix}: train.total_steps")
        if int(task.train.get("batch_size", -1)) != batch:
            setting_errors.append(f"{prefix}: train.batch_size")
        if not close(task.train.get("lr"), lr):
            setting_errors.append(f"{prefix}: train.lr")
        if int(task.train.get("T", -1)) != diffusion_steps:
            setting_errors.append(f"{prefix}: train.T")
        if int(task.eval.get("checkpoint_step", -1)) != updates:
            setting_errors.append(f"{prefix}: eval.checkpoint_step")
        if int(task.eval.get("num_images", -1)) != generated:
            setting_errors.append(f"{prefix}: eval.num_images")
        if not close(task.eval.get("guidance_scale"), guidance):
            setting_errors.append(f"{prefix}: eval.guidance_scale")
        if not task.eval.get("uniform_labels", False):
            setting_errors.append(f"{prefix}: eval.uniform_labels")
        if task.eval.get("sampler_family") != wanted_family:
            setting_errors.append(f"{prefix}: eval.sampler_family")
        if task.eval.get("metric_protocol") != "native_cifar_v1":
            setting_errors.append(f"{prefix}: eval.metric_protocol")
        skip = int(task.eval.get("ddim_skip_step", -1))
        if task.eval.get("sample_method") != "ddim" or skip <= 0 or diffusion_steps // skip != wanted_steps:
            setting_errors.append(f"{prefix}: CCUA official DDIM stride")
    checks.append(Check("ERROR" if setting_errors else "PASS", "native-cifar-controls",
                        "; ".join(setting_errors) if setting_errors else f"{updates} updates, batch={batch}, T={diffusion_steps}, DDIM-{wanted_steps}, N={generated}"))
    return checks


def _check_coral_metric_assets(campaign: LoadedCampaign) -> List[Check]:
    """Fail before training when a requested paper metric cannot be computed."""
    tasks = [t for t in campaign.tasks
             if t.adapter in {"coral", "ccua", "t2h_unified"}
             and (t.eval.get("standard_metrics", True) or t.eval.get("paper_metrics", False))]
    if not tasks:
        return []
    common_host = any(t.adapter == "t2h_unified" for t in tasks)
    native_ccua = not common_host and all(t.adapter == "ccua" for t in tasks)
    repo_name = "t2h_unified" if common_host else ("ccua" if native_ccua else "coral")
    repo_cfg = campaign.raw.get("repositories", {}).get(repo_name, {})
    repo = Path(campaign.server["runtime"]["repos_root"]) / repo_cfg.get("directory", repo_name)
    if common_host:
        required_source = (
            "dataset.py", "score/inception.py", "score/fid.py", "score/prd_score.py",
            "metrics.py", "unified_main.py",
        )
    elif native_ccua:
        required_source = ("dataset.py", "diffusion.py", "main.py", "model/model.py")
    else:
        required_source = ("dataset.py", "score/both.py", "score/improved_prd.py", "utils/augmentation.py", "loss_tracker.py")
    missing_source = [item for item in required_source if not (repo / item).exists()]
    checks: List[Check] = []
    if missing_source:
        source_check_name = "t2h-source-overlay" if common_host else ("ccua-source" if native_ccua else "coral-source-overlay")
        checks.append(Check(
            "ERROR", source_check_name,
            f"missing required {'T2H host' if common_host else ('CCUA-DDPM' if native_ccua else 'CORAL/CBDM compatibility')} files: {missing_source}"
        ))
        return checks
    source_check_name = "t2h-source-overlay" if common_host else ("ccua-source" if native_ccua else "coral-source-overlay")
    checks.append(Check(
        "PASS", source_check_name,
        "all objectives and shared metrics are backed by the T2H host" if common_host
        else ("all native objectives are backed by the pinned CCUA-DDPM runner" if native_ccua
              else "CORAL imports are backed by the pinned CBDM compatibility overlay")
    ))

    configured_metrics_root = (
        os.environ.get("LTX_T2H_METRICS_ROOT", "").strip()
        if common_host else os.environ.get("LTX_METRICS_ROOT", "").strip()
    )
    if configured_metrics_root:
        metrics_root = Path(configured_metrics_root).expanduser()
        if not metrics_root.is_absolute():
            metrics_root = campaign.root / metrics_root
        metrics_root = metrics_root.resolve()
    else:
        if common_host:
            metrics_root = (repo / "stats").resolve()
        else:
            cbdm_cfg = campaign.raw.get("repositories", {}).get("cbdm", {})
            metrics_root = (
                Path(campaign.server["runtime"]["repos_root"])
                / cbdm_cfg.get("directory", "CBDM-pytorch")
                / "stats"
            ).resolve()
    wanted = {str(t.dataset.get("data_type", "")) for t in tasks}
    names = set()
    if "cifar10" in wanted or "cifar10lt" in wanted:
        names.update(("cifar10.train.npz", "cifar10_feats.npy", "cifar10_labels.npy", "cifar10_vgg16_fc2.npy", "cifar10_vgg16_fc2_k3_radii.npy"))
    if "cifar100" in wanted or "cifar100lt" in wanted:
        names.update(("cifar100.train.npz", "cifar100_feats.npy", "cifar100_labels.npy", "cifar100_vgg16_fc2.npy", "cifar100_vgg16_fc2_k3_radii.npy"))
    if "imagenet_lt" in wanted or "imagenet200lt" in wanted:
        names.add("pt_inception-2015-12-05-6726825d.pth")
    missing = sorted(name for name in names if not (metrics_root / name).is_file())
    if missing:
        checks.append(Check("ERROR", "paper-metric-assets", f"{metrics_root}: missing balanced-reference assets {missing}"))
    else:
        checks.append(Check("PASS", "paper-metric-assets", f"balanced FID/PRD references present in {metrics_root}"))
    return checks
