#!/usr/bin/env python3
"""Evaluate one 50k generated array using the Table-1 metric protocol."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from ltx.paper_metrics import improved_prd_vgg16_k3

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-type", choices=("cifar10lt", "cifar100lt"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vgg-batch-size", type=int, default=64)
    parser.add_argument("--knn-query-batch", type=int, default=128)
    args = parser.parse_args()
    images = np.load(args.samples, mmap_mode="r"); labels = np.load(args.labels, mmap_mode="r")
    nclass = 100 if args.data_type == "cifar100lt" else 10
    if images.shape != (50000, 3, 32, 32) or labels.shape != (50000,):
        raise ValueError(f"paper protocol requires 50k CIFAR arrays, got images={images.shape} labels={labels.shape}")
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=nclass)
    if len(counts) != nclass or np.any(counts != 50000 // nclass):
        raise ValueError(f"labels are not exactly class-uniform: {counts.tolist()}")
    os.environ["LTX_METRICS_ROOT"] = str(args.metrics_root.resolve())
    sys.path.insert(0, str(args.repo))
    from score.both import get_inception_and_fid_score
    dataset = "cifar100" if nclass == 100 else "cifar10"
    flags = SimpleNamespace(prd=True, improved_prd=False, data_type=args.data_type)
    (is_score, is_std), fid, prd, _ = get_inception_and_fid_score(
        np.asarray(images), np.asarray(labels), str(args.metrics_root / f"{dataset}.train.npz"),
        num_images=50000, use_torch=False, FLAGS=flags)
    reference_features = np.load(args.metrics_root / f"{dataset}_vgg16_fc2.npy", mmap_mode="r")
    reference_radii = np.load(args.metrics_root / f"{dataset}_vgg16_fc2_k3_radii.npy", mmap_mode="r")
    precision, recall = improved_prd_vgg16_k3(np.asarray(images), np.asarray(reference_features), np.asarray(reference_radii),
                                               batch_size=args.vgg_batch_size, query_batch=args.knn_query_batch)
    payload = {"metrics": {"FID": float(fid), "IS": float(is_score), "IS_std": float(is_std), "F_8": float(prd[0]),
                           "F_1_8": float(prd[1]), "ImprovedPrecision": float(precision), "Recall": float(recall)},
               "protocol": {"samples": 50000, "labels": "uniform support across classes", "real_reference": "balanced CIFAR train",
                            "standard_prd": f"InceptionV3, {nclass * 20} clusters", "improved_prd": "VGG16 fc2, exact k-NN manifold k=3"},
               "label_histogram": counts.tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], sort_keys=True))

if __name__ == "__main__": main()
