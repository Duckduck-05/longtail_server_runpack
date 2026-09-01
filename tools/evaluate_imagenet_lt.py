#!/usr/bin/env python3
"""Evaluate array exports from the secondary ImageNet-LT campaign.

DDPM and CCUA emit float32 NCHW arrays rather than 50,000 individual files.
This evaluator keeps the paper-facing ImageNet-LT FID/KID contract used by the
CM helper, while validating the generated class schedule before loading the
metric model.  The reference split is the published balanced 20-images/class
validation manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


NUM_CLASSES = 1000
REFERENCE_IMAGES_PER_CLASS = 20


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Publish a complete metric payload, never a half-written JSON file."""
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


def _load_expected_sample_provenance(args, samples: Path) -> dict | None:
    """Fail closed when the common host gives us an unexpected artifact.

    The native evaluator path does not pass expected fields and remains
    backwards-compatible.  A common-host caller may supply the fields below so
    a metric process cannot accidentally score a sample array from another checkpoint,
    sampler, seed, or objective.
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
    path = samples.with_suffix(".provenance.json")
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


def manifest_paths(root: Path, manifest: Path, num_classes: int = NUM_CLASSES) -> list[Path]:
    paths: list[Path] = []
    counts = np.zeros(num_classes, dtype=np.int64)
    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or raw.lstrip().startswith("#"):
            continue
        if len(fields) != 2:
            raise ValueError(f"invalid ImageNet-LT manifest row {lineno}: {raw!r}")
        relative, label_raw = fields
        try:
            label = int(label_raw)
        except ValueError as exc:
            raise ValueError(f"invalid ImageNet-LT label at line {lineno}: {label_raw!r}") from exc
        if not 0 <= label < num_classes:
            raise ValueError(f"ImageNet-LT label {label} at line {lineno} is outside 0..{num_classes - 1}")
        path = Path(relative) if Path(relative).is_absolute() else root / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest image missing at line {lineno}: {path}")
        paths.append(path)
        counts[label] += 1
    if len(paths) != num_classes * REFERENCE_IMAGES_PER_CLASS:
        raise ValueError(
            f"reference manifest must contain {num_classes * REFERENCE_IMAGES_PER_CLASS} images, found {len(paths)}"
        )
    if np.any(counts != REFERENCE_IMAGES_PER_CLASS):
        raise ValueError(f"reference manifest must contain exactly {REFERENCE_IMAGES_PER_CLASS} images/class: {counts.tolist()}")
    return paths


def load_generated(samples_path: Path, labels_path: Path, num_images: int, image_size: int,
                   num_classes: int = NUM_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    images = np.load(samples_path, mmap_mode="r")
    labels = np.asarray(np.load(labels_path, allow_pickle=False), dtype=np.int64)
    if images.ndim != 4 or images.shape[1:] != (3, image_size, image_size):
        raise ValueError(f"generated samples must have shape [N,3,{image_size},{image_size}], got {images.shape}")
    if images.shape[0] != num_images or labels.shape != (num_images,):
        raise ValueError(f"generated sample/label count mismatch: samples={images.shape}, labels={labels.shape}, expected={num_images}")
    if not np.issubdtype(images.dtype, np.number):
        raise ValueError(f"generated samples must be numeric, got {images.dtype}")
    minimum = float(np.min(images))
    maximum = float(np.max(images))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum < -1e-5 or maximum > 1.00001:
        raise ValueError(f"generated samples must be finite in [0,1], got min={minimum}, max={maximum}")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("generated labels contain a value outside the ImageNet-LT class range")
    if num_images % num_classes:
        raise ValueError(f"num_images={num_images} is not divisible by num_classes={num_classes}")
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts != num_images // num_classes):
        raise ValueError(f"generated labels must be exactly class-uniform: {counts.tolist()}")
    return images, labels


def image_batches(paths: Iterable[Path], batch_size: int):
    to_tensor = transforms.ToTensor()
    batch: list[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as image:
            batch.append(to_tensor(image.convert("RGB")))
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    if batch:
        yield torch.stack(batch)


def array_batches(images: np.ndarray, batch_size: int):
    for start in range(0, len(images), batch_size):
        # Copy only one micro-batch: the full generated array remains memory
        # mapped, while torch receives writable storage on every backend.
        yield torch.from_numpy(np.asarray(images[start:start + batch_size], dtype=np.float32).copy())


def features(batches: Iterable[torch.Tensor], total: int, batch_size: int,
             device: torch.device, repo: Path, desc: str) -> np.ndarray:
    sys.path.insert(0, str(repo))
    previous = Path.cwd()
    os.chdir(repo)
    try:
        if (repo / "score" / "inception.py").is_file():
            from score.inception import InceptionV3
        else:
            # Backward-compatible path for the old source-native CM launcher.
            from imbdiff_cm.score.inception import InceptionV3

        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        output = np.empty((total, 2048), dtype=np.float32)
        start = 0
        with torch.no_grad():
            for batch in tqdm(batches, total=(total + batch_size - 1) // batch_size, desc=desc):
                value = model(batch.to(device)).flatten(1).cpu().numpy()
                output[start:start + len(value)] = value
                start += len(value)
        if start != total:
            raise RuntimeError(f"feature extraction stopped early: {start}/{total}")
        return output
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="metric host root containing FID Inception")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--samples", required=True, help="float32 NCHW .npy array in [0,1]")
    parser.add_argument("--labels", required=True, help="int64 labels aligned with --samples")
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kid-repeats", type=int, default=2)
    parser.add_argument("--weights", default="",
                        help="optional pinned FID-Inception checkpoint; defaults to <repo>/stats/... for legacy runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
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
    if args.batch_size <= 0 or args.kid_repeats <= 0:
        raise ValueError("--batch-size and --kid-repeats must be positive")

    repo = Path(args.repo).resolve()
    root = Path(args.image_root).resolve()
    args.samples = Path(args.samples).resolve()
    args.sample_provenance = _load_expected_sample_provenance(args, args.samples)
    reference = manifest_paths(root, Path(args.reference_manifest).resolve(), args.num_classes)
    images, labels = load_generated(
        args.samples, Path(args.labels).resolve(), args.num_images, args.image_size, args.num_classes
    )
    weights = Path(args.weights).expanduser().resolve() if args.weights else (
        repo / "stats" / "pt_inception-2015-12-05-6726825d.pth"
    )
    if not weights.is_file():
        raise FileNotFoundError(f"FID Inception weights missing: {weights}")
    os.environ["LTX_FID_WEIGHTS_PATH"] = str(weights)

    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    real = features(image_batches(reference, args.batch_size), len(reference), args.batch_size, device, repo, "ImageNet-LT reference features")
    generated = features(array_batches(images, args.batch_size), len(images), args.batch_size, device, repo, "ImageNet-LT generated features")

    if (repo / "metrics.py").is_file():
        from metrics import calculate_frechet_distance, polynomial_mmd_kid
    else:
        # Backward-compatible path for the old source-native CM launcher.
        from imbdiff_cm.metrics import calculate_frechet_distance, polynomial_mmd_kid

    fid = float(calculate_frechet_distance(
        np.mean(generated, axis=0), np.cov(generated, rowvar=False),
        np.mean(real, axis=0), np.cov(real, rowvar=False),
    ))
    kid_values = [float(polynomial_mmd_kid(generated, real)) for _ in range(args.kid_repeats)]
    payload = {
        "protocol": "ImageNet-LT 64x64: pinned FID-Inception/KID, generated balanced grid",
        "FID": fid,
        "KID": {"mean": float(np.mean(kid_values)), "std": float(np.std(kid_values)), "all": kid_values},
        "num_generated": int(len(generated)),
        "num_reference": int(len(real)),
        "num_classes": int(args.num_classes),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
    }
    if args.sample_provenance is not None:
        payload["provenance"] = {
            "metric_host": "common_imagenet_metrics_v2",
            "sample": args.sample_provenance,
        }
    output = Path(args.output).resolve()
    _atomic_write_json(output, payload)
    print(f"FID: {fid:.6f}")
    print(f"KID: {payload['KID']['mean']:.8f}")


if __name__ == "__main__":
    main()
