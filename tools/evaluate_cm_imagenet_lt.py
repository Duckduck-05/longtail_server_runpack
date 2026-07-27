#!/usr/bin/env python3
"""CM-paper metric evaluator for an ImageNet-LT manifest and generated PNGs.

The CM paper's ImageNet-LT/iNaturalist table reports FID and KID.  Both sides
use the same licensed ImageNet image root; generated samples are the balanced
50k class-conditional grid emitted by the released CM sampler.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def manifest_paths(root: Path, manifest: Path) -> list[Path]:
    paths: list[Path] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"invalid ImageNet-LT manifest row: {line!r}")
        path = root / fields[0]
        if not path.is_file():
            raise FileNotFoundError(f"manifest image missing: {path}")
        paths.append(path)
    if len(paths) < 2:
        raise ValueError("reference manifest must contain at least two images")
    return paths


_CLASS_FILENAME = re.compile(r"^\d+_class_(\d+)\.png$")


def generated_paths(image_dir: Path, limit: int, num_classes: int = 1000) -> list[Path]:
    if limit % num_classes:
        raise ValueError(f"num_images={limit} is not divisible by num_classes={num_classes}")
    paths = sorted(p for p in image_dir.glob("*.png") if p.is_file())
    if len(paths) != limit:
        raise ValueError(f"paper protocol requires exactly {limit} generated PNGs in {image_dir}, found {len(paths)}")
    labels = []
    for path in paths:
        match = _CLASS_FILENAME.match(path.name)
        if not match:
            raise ValueError(f"cannot certify class-uniform sampling from filename: {path.name}")
        labels.append(int(match.group(1)))
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts != limit // num_classes):
        raise ValueError(f"generated labels must be exactly uniform: {counts.tolist()}")
    return paths


def batches(paths: Iterable[Path], batch_size: int):
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


def features(paths: list[Path], batch_size: int, device: torch.device, repo: Path) -> np.ndarray:
    sys.path.insert(0, str(repo))
    # The upstream FID Inception implementation resolves its weights relative
    # to CM's repository (stats/...).  Make that contract explicit here.
    previous = Path.cwd()
    os.chdir(repo)
    try:
        from imbdiff_cm.score.inception import InceptionV3
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        out = np.empty((len(paths), 2048), dtype=np.float32)
        start = 0
        with torch.no_grad():
            for batch in tqdm(batches(paths, batch_size), total=(len(paths) + batch_size - 1) // batch_size,
                              desc="CM Inception features"):
                value = model(batch.to(device)).flatten(1).cpu().numpy()
                out[start:start + len(value)] = value
                start += len(value)
        return out
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="vendored ImbDiff-CM root")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--reference-manifest", required=True,
                        help="balanced ImageNet-LT reference split used for FID/KID")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kid-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo, root, manifest, image_dir = map(Path, (args.repo, args.image_root, args.reference_manifest, args.generated_dir))
    if not (repo / "stats" / "pt_inception-2015-12-05-6726825d.pth").is_file():
        raise FileNotFoundError("CM FID Inception weights missing; run scripts/prepare_cm_metric_assets.sh first")
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    real = features(manifest_paths(root, manifest), args.batch_size, device, repo)
    generated = features(generated_paths(image_dir, args.num_images), args.batch_size, device, repo)
    sys.path.insert(0, str(repo))
    from imbdiff_cm.metrics import calculate_frechet_distance, polynomial_mmd_kid
    fid = float(calculate_frechet_distance(np.mean(generated, axis=0), np.cov(generated, rowvar=False),
                                           np.mean(real, axis=0), np.cov(real, rowvar=False)))
    kid_values = [float(polynomial_mmd_kid(generated, real)) for _ in range(args.kid_repeats)]
    payload = {
        "protocol": "CM ImageNet-LT: FID/KID, released CM FID-Inception, generated balanced grid",
        "FID": fid,
        "KID": {"mean": float(np.mean(kid_values)), "std": float(np.std(kid_values)), "all": kid_values},
        "num_generated": int(len(generated)), "num_reference": int(len(real)), "seed": args.seed,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"FID: {fid:.6f}")
    print(f"KID: {payload['KID']['mean']:.8f}")


if __name__ == "__main__":
    main()
