#!/usr/bin/env python3
"""FID/KID evaluator for CM Table-baseline CIFAR-LT runs."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm


_CLASS_FILENAME = re.compile(r"^\d+_class_(\d+)\.png$")


def generated_batches(image_dir: Path, count: int, num_classes: int, batch_size: int):
    if count % num_classes:
        raise ValueError(f"num_images={count} is not divisible by num_classes={num_classes}")
    paths = sorted(image_dir.glob("*.png"))
    if len(paths) != count:
        raise ValueError(f"paper protocol requires exactly {count} generated images in {image_dir}, found {len(paths)}")
    labels = []
    for path in paths:
        match = _CLASS_FILENAME.match(path.name)
        if not match:
            raise ValueError(f"cannot certify class-uniform sampling from filename: {path.name}")
        labels.append(int(match.group(1)))
    histogram = np.bincount(labels, minlength=num_classes)
    if len(histogram) != num_classes or np.any(histogram != count // num_classes):
        raise ValueError(f"generated labels must be exactly uniform: {histogram.tolist()}")
    to_tensor = transforms.ToTensor()
    for start in range(0, count, batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                images.append(to_tensor(image.convert("RGB")))
        yield torch.stack(images)


def real_batches(dataset, batch_size: int):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    yield from (images for images, _ in loader)


def extract(batches, total: int, device: torch.device, repo: Path, label: str) -> np.ndarray:
    sys.path.insert(0, str(repo))
    old_cwd = Path.cwd(); os.chdir(repo)
    try:
        from imbdiff_cm.score.inception import InceptionV3
        model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        values = np.empty((total, 2048), dtype=np.float32); start = 0
        with torch.no_grad():
            for batch in tqdm(batches, total=None, desc=label):
                feature = model(batch.to(device)).flatten(1).cpu().numpy()
                values[start:start + len(feature)] = feature; start += len(feature)
        if start != total:
            raise ValueError(f"{label}: expected {total} features, got {start}")
        return values
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kid-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / "stats" / "pt_inception-2015-12-05-6726825d.pth").is_file():
        raise FileNotFoundError("CM FID Inception weights missing; run scripts/prepare_cm_metric_assets.sh")
    Dataset = CIFAR10 if args.dataset == "cifar10" else CIFAR100
    real_dataset = Dataset(args.data_root, train=True, download=True, transform=transforms.ToTensor())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    real = extract(real_batches(real_dataset, args.batch_size), len(real_dataset), device, repo, "real CM features")
    generated = extract(generated_batches(Path(args.generated_dir), args.num_images, 10 if args.dataset == "cifar10" else 100,
                                          args.batch_size), args.num_images, device, repo, "generated CM features")
    sys.path.insert(0, str(repo))
    from imbdiff_cm.metrics import calculate_frechet_distance, polynomial_mmd_kid
    fid = float(calculate_frechet_distance(np.mean(generated, 0), np.cov(generated, rowvar=False),
                                           np.mean(real, 0), np.cov(real, rowvar=False)))
    kid = [float(polynomial_mmd_kid(generated, real)) for _ in range(args.kid_repeats)]
    payload = {"protocol": f"CM CIFAR-LT {args.dataset}: released CM FID-Inception/FID/KID",
               "FID": fid, "KID": {"mean": float(np.mean(kid)), "std": float(np.std(kid)), "all": kid},
               "num_generated": args.num_images, "num_reference": len(real), "seed": args.seed}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"FID: {fid:.6f}\nKID: {payload['KID']['mean']:.8f}")


if __name__ == "__main__":
    main()
