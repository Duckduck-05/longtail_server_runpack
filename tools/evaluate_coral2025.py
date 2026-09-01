#!/usr/bin/env python3
"""Evaluate one generated CIFAR array with a pinned common metric protocol.

``--mode headline`` is the early-result protocol: it reports the common
Inception FID/IS/PRD columns and deterministic KID, but deliberately omits
the exact VGG16 50k-by-50k improved-PRD and every per-class FID. The default
``detailed`` mode keeps the legacy final metric set; it writes an atomic
headline snapshot before starting those optional expensive calculations.
``--mode quick`` accepts a smaller class-uniform smoke sample and reports only
the common fast metrics; its values are not for paper reporting or direct
comparison with the 50k protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ltx.paper_metrics import improved_prd_vgg16_k3, polynomial_mmd_kid


_IS_SPLITS = 10


def _sample_provenance_path(samples: Path) -> Path:
    """Return the sidecar written by the common T2H sampler."""
    return samples.with_suffix(".provenance.json")


def _load_expected_sample_provenance(args, samples: Path) -> dict | None:
    """Validate the sample/checkpoint identity when called by the common host.

    Native paper evaluators do not pass the ``expected_*`` flags and therefore
    retain their old input contract.  The unified benchmark passes them all;
    missing or mismatched sidecars then fail before an expensive Inception
    pass can publish a number for the wrong checkpoint.
    """
    expected = {
        "host_revision": getattr(args, "expected_host_revision", None),
        "checkpoint_schema": getattr(args, "expected_checkpoint_schema", None),
        "objective": getattr(args, "expected_objective", None),
        "checkpoint_step": getattr(args, "expected_checkpoint_step", None),
        "num_images": getattr(args, "expected_num_images", None),
        "sample_method": getattr(args, "expected_sample_method", None),
        "sampler_method": getattr(args, "expected_sampler_method", None),
        "ddim_skip_step": getattr(args, "expected_ddim_skip_step", None),
        "omega": getattr(args, "expected_omega", None),
        "seed": getattr(args, "expected_seed", None),
        "artifact_namespace": getattr(args, "expected_artifact_namespace", None),
        "T": getattr(args, "expected_T", None),
        "beta_1": getattr(args, "expected_beta_1", None),
        "beta_T": getattr(args, "expected_beta_T", None),
        "var_type": getattr(args, "expected_var_type", None),
        "img_size": getattr(args, "expected_img_size", None),
        "num_class": getattr(args, "expected_num_class", None),
        "uniform_labels": getattr(args, "expected_uniform_labels", None),
    }
    expected = {key: value for key, value in expected.items() if value is not None}
    if not expected:
        return None
    path = _sample_provenance_path(samples)
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            f"unified evaluation requires a valid sampler provenance sidecar: {path}"
        ) from exc
    if not isinstance(actual, dict):
        raise ValueError(f"sampler provenance must be a JSON object: {path}")
    mismatches = []
    for key, wanted in expected.items():
        got = actual.get(key)
        if isinstance(wanted, float):
            try:
                equal = bool(np.isclose(float(got), wanted, rtol=0.0, atol=1e-12))
            except (TypeError, ValueError):
                equal = False
        else:
            equal = got == wanted
        if not equal:
            mismatches.append(f"{key}={got!r} (expected {wanted!r})")
    if mismatches:
        raise ValueError(
            f"sample provenance mismatch for {samples}: " + "; ".join(mismatches)
        )
    return actual


def _metric_provenance(args) -> dict | None:
    sample = getattr(args, "sample_provenance", None)
    if sample is None:
        return None
    return {"metric_host": "common_cifar_metrics_v2", "sample": sample}


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON without exposing a partially written metric artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _inception_features_and_is(repo: Path, images: np.ndarray, batch_size: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Extract pinned 2048-d Inception features once and calculate IS online.

    The released ``score.both`` helper hides the activation array after it
    computes FID/IS/PRD. Keeping that array here lets KID and optional
    class-level FIDs reuse the same model pass. IS is accumulated by split,
    avoiding a second 50k-by-1008 host array while preserving its formula.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("common CIFAR evaluation requires CUDA")
    sys.path.insert(0, str(repo))
    from score.inception import InceptionV3

    num_images = len(images)
    feature_dim = 2048
    model = InceptionV3([
        InceptionV3.BLOCK_INDEX_BY_DIM[feature_dim],
        InceptionV3.BLOCK_INDEX_BY_DIM["prob"],
    ]).cuda().eval()
    features = np.empty((num_images, feature_dim), dtype=np.float32)
    split_sizes = np.array(
        [((index + 1) * num_images // _IS_SPLITS) - (index * num_images // _IS_SPLITS)
         for index in range(_IS_SPLITS)],
        dtype=np.int64,
    )
    probability_sums: np.ndarray | None = None
    probability_log_sums = np.zeros(_IS_SPLITS, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size])).float().cuda(non_blocking=True)
            end = start + len(batch)
            prediction = model(batch)
            features[start:end] = prediction[0].flatten(1).cpu().numpy().astype(np.float32, copy=False)
            probabilities = prediction[1].cpu().numpy().astype(np.float64, copy=False)
            if probability_sums is None:
                probability_sums = np.zeros((_IS_SPLITS, probabilities.shape[1]), dtype=np.float64)

            cursor = start
            while cursor < end:
                split = min(_IS_SPLITS - 1, cursor * _IS_SPLITS // num_images)
                split_end = min(end, (split + 1) * num_images // _IS_SPLITS)
                chunk = probabilities[cursor - start:split_end - start]
                probability_sums[split] += np.sum(chunk, axis=0, dtype=np.float64)
                probability_log_sums[split] += np.sum(chunk * np.log(chunk), dtype=np.float64)
                cursor = split_end

    if probability_sums is None:
        raise ValueError("Inception Score requires at least one image")
    scores = []
    for split, split_size in enumerate(split_sizes):
        if split_size <= 0:
            raise ValueError(f"Inception Score requires at least {_IS_SPLITS} images")
        average_probability = probability_sums[split] / split_size
        scores.append(np.exp(
            probability_log_sums[split] / split_size
            - np.sum(average_probability * np.log(average_probability), dtype=np.float64)
        ))
    return features, (float(np.mean(scores)), float(np.std(scores)))


def _validate_feature_pair(features: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    generated = np.asarray(features)
    real = np.asarray(reference)
    if generated.ndim != 2 or real.ndim != 2 or generated.shape[1] != real.shape[1]:
        raise ValueError(
            "FID expects generated/reference arrays with matching [N, feature_dim] shapes, got "
            f"{generated.shape} and {real.shape}"
        )
    if len(generated) < 2 or len(real) < 2:
        raise ValueError(f"FID requires at least two samples on each side, got {len(generated)} and {len(real)}")
    return generated, real


def _fid_low_rank(features: np.ndarray, reference: np.ndarray) -> float:
    """Exact sample-covariance FID from the nuclear norm of a small Gram matrix.

    For centred feature matrices A and B scaled by ``sqrt(n - 1)``, their
    sample covariances are A.T @ A and B.T @ B. Therefore the covariance cross
    term is ``||A @ B.T||_*``. This is the same Frechet expression as SciPy's
    ``sqrtm(C1 @ C2)`` but, for small class slices, SVDs an
    ``n_generated`` by ``n_reference`` matrix instead of a 2048 by 2048 one.
    """
    generated, real = _validate_feature_pair(features, reference)
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    generated_mean = np.mean(generated, axis=0, dtype=np.float64)
    real_mean = np.mean(real, axis=0, dtype=np.float64)
    generated_factor = (generated - generated_mean) / np.sqrt(len(generated) - 1)
    real_factor = (real - real_mean) / np.sqrt(len(real) - 1)
    covariance_trace = (
        np.sum(generated_factor * generated_factor, dtype=np.float64)
        + np.sum(real_factor * real_factor, dtype=np.float64)
    )
    covariance_cross_trace = np.sum(
        np.linalg.svd(generated_factor @ real_factor.T, compute_uv=False), dtype=np.float64
    )
    mean_distance = np.sum((generated_mean - real_mean) ** 2, dtype=np.float64)
    return float(mean_distance + covariance_trace - 2.0 * covariance_cross_trace)


def _should_use_low_rank_fid(features: np.ndarray, reference: np.ndarray) -> bool:
    """Use the Gram formulation only when its dense SVD is actually smaller."""
    generated, real = _validate_feature_pair(features, reference)
    dimension = generated.shape[1]
    return len(generated) * len(real) <= dimension * dimension


def _fid(features: np.ndarray, reference: np.ndarray, calculate_frechet_distance) -> float:
    generated, real = _validate_feature_pair(features, reference)
    if _should_use_low_rank_fid(generated, real):
        return _fid_low_rank(generated, real)
    return float(calculate_frechet_distance(
        np.mean(generated, axis=0, dtype=np.float64), np.cov(generated, rowvar=False),
        np.mean(real, axis=0, dtype=np.float64), np.cov(real, rowvar=False),
    ))


def _fid_to_statistics(features: np.ndarray, reference_mean: np.ndarray, reference_covariance: np.ndarray,
                       calculate_frechet_distance) -> float:
    generated = np.asarray(features)
    if generated.ndim != 2 or len(generated) < 2:
        raise ValueError(f"FID requires at least two generated feature vectors, got {generated.shape}")
    return float(calculate_frechet_distance(
        np.mean(generated, axis=0, dtype=np.float64), np.cov(generated, rowvar=False),
        reference_mean, reference_covariance,
    ))


def _standard_prd_scores(generated_features: np.ndarray, reference_features: np.ndarray,
                         nclass: int, *, legacy_repeat: bool = False) -> tuple[float, float]:
    """Run the existing 10-run Inception PRD protocol on reused activations.

    The released helper accidentally invoked the stochastic 10-run estimator
    twice and returned its second result. Detailed mode retains that sequence
    for backward-compatible values; explicit headline mode performs one
    estimator call because it is an intentionally new fast protocol.
    """
    from score.prd_score import compute_prd_from_embedding, prd_to_max_f_beta_pair

    prd_data = compute_prd_from_embedding(
        eval_data=generated_features,
        ref_data=reference_features,
        num_clusters=nclass * 20,
        num_angles=1001,
        num_runs=10,
        enforce_balance=True,
    )
    if legacy_repeat:
        prd_data = compute_prd_from_embedding(
            eval_data=generated_features,
            ref_data=reference_features,
            num_clusters=nclass * 20,
            num_angles=1001,
            num_runs=10,
            enforce_balance=True,
        )
    f_8, f_1_8 = prd_to_max_f_beta_pair(prd_data[0], prd_data[1], beta=8)
    return float(f_8), float(f_1_8)


def _three_way_groups(nclass: int) -> dict[str, np.ndarray]:
    if nclass == 10:
        return {
            "Many": np.arange(0, 3),
            "Medium": np.arange(3, 6),
            "Few": np.arange(6, 10),
        }
    if nclass == 100:
        return {
            "Many": np.arange(0, 33),
            "Medium": np.arange(33, 66),
            "Few": np.arange(66, 100),
        }
    raise ValueError(f"CM three-way grouping only supports 10 or 100 classes, got {nclass}")


def _base_protocol(nclass: int, args, sample_count: int = 50000) -> dict:
    return {
        "samples": sample_count,
        "labels": "uniform support across classes",
        "real_reference": "balanced CIFAR train",
        "standard_prd": f"InceptionV3, {nclass * 20} clusters",
        "improved_prd": "VGG16 fc2, exact k-NN manifold k=3",
        "inception_batch_size": args.inception_batch_size,
    }


def _headline_protocol(protocol: dict, mode: str = "headline") -> dict:
    headline = dict(protocol)
    if mode == "quick":
        caveat = (
            f"Quick smoke uses {protocol['samples']} class-uniform samples instead of the paper's 50k; "
            "it is not for paper reporting and must not be compared directly with 50k results. "
            f"{protocol['prd_reference']}. "
            "It omits VGG16 fc2 exact improved-PRD (ImprovedPrecision/Recall) "
            "from the overall quick report; optional per-class FIDs are written separately."
        )
    else:
        caveat = (
            "Headline mode omits VGG16 fc2 exact improved-PRD (ImprovedPrecision/Recall) "
            "and all per-class/split FIDs. F_8 and F_1_8 remain the standard Inception PRD protocol."
        )
    headline.update({"evaluation_mode": mode, "important_caveat": caveat})
    return headline


def _quick_prd_reference_subset(
    args, reference_features: np.ndarray, nclass: int, sample_count: int, dataset: str
) -> np.ndarray:
    """Select the deterministic balanced real subset used by quick PRD."""
    labels_path = args.metrics_root / f"{dataset}_labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"quick PRD requires the balanced-reference labels asset {labels_path}; "
            "rerun tools/prepare_cifar_metric_assets.py"
        )
    reference_labels = np.load(labels_path, mmap_mode="r")
    if reference_labels.ndim != 1 or reference_labels.shape[0] != reference_features.shape[0]:
        raise ValueError(
            "quick PRD requires reference features and labels with matching [N] lengths, got "
            f"features={reference_features.shape} labels={reference_labels.shape}"
        )
    per_class, remainder = divmod(sample_count, nclass)
    if remainder:
        raise ValueError(
            f"quick PRD requires a sample count divisible by {nclass}, got N={sample_count}"
        )
    selected = []
    reference_labels_array = np.asarray(reference_labels)
    for class_id in range(nclass):
        class_indices = np.flatnonzero(reference_labels_array == class_id)
        if len(class_indices) < per_class:
            raise ValueError(
                f"quick PRD reference has only {len(class_indices)} samples for class {class_id}; "
                f"requires {per_class}"
            )
        selected.append(class_indices[:per_class])
    return np.asarray(reference_features[np.concatenate(selected)])


def _collect_headline_metrics(args, images: np.ndarray, nclass: int, dataset: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute every shared headline metric from one generated feature array."""
    generated_features, (is_score, is_std) = _inception_features_and_is(
        args.repo, images, batch_size=args.inception_batch_size
    )
    reference_features = np.load(args.metrics_root / f"{dataset}_feats.npy", mmap_mode="r")
    prd_reference_features = reference_features
    if args.mode == "quick":
        prd_reference_features = _quick_prd_reference_subset(
            args, reference_features, nclass, len(generated_features), dataset
        )
    sys.path.insert(0, str(args.repo))
    from score.fid import calculate_frechet_distance

    with np.load(args.metrics_root / f"{dataset}.train.npz") as statistics:
        fid = _fid_to_statistics(
            generated_features, statistics["mu"][:], statistics["sigma"][:], calculate_frechet_distance
        )
    f_8, f_1_8 = _standard_prd_scores(
        generated_features,
        prd_reference_features,
        nclass,
        legacy_repeat=args.mode == "detailed",
    )
    metrics = {
        "FID": float(fid),
        "IS": float(is_score),
        "IS_std": float(is_std),
        "F_8": float(f_8),
        "F_1_8": float(f_1_8),
    }
    if args.kid or args.mode in ("headline", "quick"):
        metrics["KID"] = polynomial_mmd_kid(
            generated_features,
            reference_features,
            num_subsets=args.kid_subsets,
            max_subset_size=args.kid_subset_size,
            rng=np.random.default_rng(args.kid_seed),
        )
    return metrics, generated_features, reference_features


def _per_class_payload(args, generated_features: np.ndarray, reference_features: np.ndarray,
                       labels: np.ndarray, nclass: int, dataset: str) -> dict:
    labels_path = args.metrics_root / f"{dataset}_labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"per-class FID requires the balanced-reference labels asset {labels_path}; "
            "rerun tools/prepare_cifar_metric_assets.py"
        )
    reference_labels = np.load(labels_path, mmap_mode="r")
    sys.path.insert(0, str(args.repo))
    from score.fid import calculate_frechet_distance

    per_class = {}
    for class_id in range(nclass):
        generated_mask = labels == class_id
        reference_mask = np.asarray(reference_labels) == class_id
        generated_slice = generated_features[generated_mask]
        reference_slice = reference_features[reference_mask]
        per_class[str(class_id)] = {
            "FID": _fid(generated_slice, reference_slice, calculate_frechet_distance),
            "generated": int(generated_mask.sum()),
            "reference": int(reference_mask.sum()),
        }
    groups = {}
    if args.longtail_groups == "cm_three_way":
        for name, classes in _three_way_groups(nclass).items():
            generated_mask = np.isin(labels, classes)
            reference_mask = np.isin(reference_labels, classes)
            groups[name] = {
                "FID": _fid(generated_features[generated_mask], reference_features[reference_mask], calculate_frechet_distance),
                "classes": classes.tolist(),
                "generated": int(generated_mask.sum()),
                "reference": int(reference_mask.sum()),
            }
    sample_count = int(labels.shape[0])
    return {
        "protocol": {
            "feature_extractor": "pinned CBDM InceptionV3 (2048-d)",
            "reference": "balanced CIFAR train",
            "sample_count": sample_count,
            "sample_schedule": (
                f"same {'50k' if sample_count == 50000 else sample_count} "
                "class-uniform samples as overall table"
            ),
            "fid_computation": (
                "exact sample-covariance FID; class slices whose generated-by-reference Gram matrix "
                "is no larger than 2048-by-2048 use an equivalent low-rank nuclear-norm formulation, "
                "while larger slices retain the release SciPy sqrtm formulation"
            ),
            "important_caveat": (
                "CM's published split FIDs use separately sampled 20k generated images per split. "
                f"These values use the common table's class-uniform "
                f"{'50k' if sample_count == 50000 else sample_count} sample and are only comparable within this campaign."
            ),
        },
        "per_class": per_class,
        "groups": groups,
        **({"provenance": _metric_provenance(args)} if _metric_provenance(args) is not None else {}),
    }


def _evaluate(args, images: np.ndarray, labels: np.ndarray, counts: np.ndarray, nclass: int) -> dict:
    """Run one protocol, publishing the headline result before detailed work."""
    dataset = "cifar100" if nclass == 100 else "cifar10"
    headline_metrics, generated_features, reference_features = _collect_headline_metrics(args, images, nclass, dataset)
    sample_count = int(images.shape[0]) if args.mode == "quick" else 50000
    protocol = _base_protocol(nclass, args, sample_count=sample_count)
    if args.mode == "quick":
        protocol["prd_reference"] = (
            "F_8/F_1_8 use the deterministic class-balanced real subset "
            f"with the first {sample_count // nclass} reference samples per class "
            f"({sample_count} total)"
        )
    if "KID" in headline_metrics:
        protocol["kid"] = {
            "feature_extractor": "pinned CBDM InceptionV3 (2048-d)",
            "estimator": "unbiased cubic polynomial MMD, CM release formula",
            "subsets": args.kid_subsets,
            "max_subset_size": args.kid_subset_size,
            "subset_rng_seed": args.kid_seed,
        }
    headline_mode = "quick" if args.mode == "quick" else "headline"
    headline_payload = {
        "metrics": headline_metrics,
        "protocol": _headline_protocol(protocol, mode=headline_mode),
        "label_histogram": counts.tolist(),
    }
    metric_provenance = _metric_provenance(args)
    if metric_provenance is not None:
        headline_payload["provenance"] = metric_provenance
    # The primary path becomes available as a valid headline report immediately;
    # detailed mode atomically replaces it only after all optional work succeeds.
    _atomic_write_json(args.output, headline_payload)
    if args.headline_output is not None:
        _atomic_write_json(args.headline_output, headline_payload)
    if args.mode == "quick":
        if args.per_class_output is not None:
            _atomic_write_json(
                args.per_class_output,
                _per_class_payload(args, generated_features, reference_features, labels, nclass, dataset),
            )
        return headline_payload
    if args.mode == "headline":
        return headline_payload

    metrics = dict(headline_metrics)
    reference_vgg_features = np.load(args.metrics_root / f"{dataset}_vgg16_fc2.npy", mmap_mode="r")
    reference_radii = np.load(args.metrics_root / f"{dataset}_vgg16_fc2_k3_radii.npy", mmap_mode="r")
    precision, recall = improved_prd_vgg16_k3(
        np.asarray(images),
        np.asarray(reference_vgg_features),
        np.asarray(reference_radii),
        batch_size=args.vgg_batch_size,
        query_batch=args.knn_query_batch,
    )
    metrics["ImprovedPrecision"] = float(precision)
    metrics["Recall"] = float(recall)
    if args.per_class_output is not None:
        _atomic_write_json(
            args.per_class_output,
            _per_class_payload(args, generated_features, reference_features, labels, nclass, dataset),
        )
    payload = {"metrics": metrics, "protocol": protocol, "label_histogram": counts.tolist()}
    if metric_provenance is not None:
        payload["provenance"] = metric_provenance
    _atomic_write_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-type", choices=("cifar10lt", "cifar100lt"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("detailed", "headline", "quick"), default="detailed",
        help="detailed preserves legacy 50k VGG/per-class behavior; headline emits fast common metrics plus KID; quick accepts a smaller class-uniform smoke sample",
    )
    parser.add_argument(
        "--headline-output", type=Path, default=None,
        help="optional permanent atomic headline JSON; --output is also written early then replaced by the detailed result",
    )
    parser.add_argument(
        "--inception-batch-size", type=int, default=16,
        help="GPU micro-batch for the pinned Inception FID/IS/PRD/KID feature pass (default: 16)",
    )
    parser.add_argument("--vgg-batch-size", type=int, default=128)
    parser.add_argument("--knn-query-batch", type=int, default=1024)
    parser.add_argument("--kid", action="store_true", help="append deterministic CM-style KID in detailed mode (always included in headline and quick modes)")
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--kid-seed", type=int, default=2026)
    parser.add_argument("--per-class-output", type=Path, default=None,
                        help="optional JSON with per-class FID and CM Many/Medium/Few FIDs (detailed and quick modes)")
    parser.add_argument("--longtail-groups", choices=("none", "cm_three_way"), default="none")
    parser.add_argument("--expected-host-revision", default=None)
    parser.add_argument("--expected-checkpoint-schema", type=int, default=None)
    parser.add_argument("--expected-objective", default=None)
    parser.add_argument("--expected-checkpoint-step", type=int, default=None)
    parser.add_argument("--expected-num-images", type=int, default=None)
    parser.add_argument("--expected-sample-method", default=None)
    parser.add_argument("--expected-sampler-method", default=None)
    parser.add_argument("--expected-ddim-skip-step", type=int, default=None)
    parser.add_argument("--expected-omega", type=float, default=None)
    parser.add_argument("--expected-seed", type=int, default=None)
    parser.add_argument("--expected-artifact-namespace", default=None)
    parser.add_argument("--expected-T", type=int, default=None)
    parser.add_argument("--expected-beta-1", type=float, default=None)
    parser.add_argument("--expected-beta-T", type=float, default=None)
    parser.add_argument("--expected-var-type", default=None)
    parser.add_argument("--expected-img-size", type=int, default=None)
    parser.add_argument("--expected-num-class", type=int, default=None)
    parser.add_argument("--expected-uniform-labels", action="store_true", default=None)
    args = parser.parse_args()
    if args.inception_batch_size <= 0:
        parser.error("--inception-batch-size must be positive")
    if args.vgg_batch_size <= 0:
        parser.error("--vgg-batch-size must be positive")
    if args.knn_query_batch <= 0:
        parser.error("--knn-query-batch must be positive")
    if args.kid_subsets <= 0:
        parser.error("--kid-subsets must be positive")
    if args.kid_subset_size <= 1:
        parser.error("--kid-subset-size must be at least two")
    if args.mode == "headline" and args.per_class_output is not None:
        parser.error("--mode headline intentionally omits per-class FID; rerun with --mode detailed for --per-class-output")
    if args.mode == "detailed" and args.headline_output is not None and args.headline_output == args.output:
        parser.error("--headline-output must differ from --output in detailed mode so the headline result remains available")

    args.samples = Path(args.samples).resolve()
    args.sample_provenance = _load_expected_sample_provenance(args, args.samples)
    images = np.load(args.samples, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    nclass = 100 if args.data_type == "cifar100lt" else 10
    if args.mode in ("detailed", "headline"):
        if images.shape != (50000, 3, 32, 32) or labels.shape != (50000,):
            raise ValueError(f"paper protocol requires 50k CIFAR arrays, got images={images.shape} labels={labels.shape}")
    else:
        if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
            raise ValueError(f"quick protocol requires images with shape [N, 3, 32, 32], got {images.shape}")
        if labels.ndim != 1 or labels.shape != (images.shape[0],):
            raise ValueError(f"quick protocol requires labels with shape [N], got {labels.shape} for N={images.shape[0]}")
        if images.shape[0] <= 0 or images.shape[0] % nclass != 0:
            raise ValueError(
                f"quick protocol requires a positive sample count divisible by {nclass}, got N={images.shape[0]}"
            )
    labels_array = np.asarray(labels)
    counts = np.bincount(np.asarray(labels_array, dtype=np.int64), minlength=nclass)
    expected_count = images.shape[0] // nclass
    if len(counts) != nclass or np.any(counts != expected_count):
        raise ValueError(f"labels are not exactly class-uniform: {counts.tolist()}")
    os.environ["LTX_METRICS_ROOT"] = str(args.metrics_root.resolve())
    payload = _evaluate(args, images, labels_array, counts, nclass)
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
