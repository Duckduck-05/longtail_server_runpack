#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def ids_hash(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(ids.astype(str).tolist()).encode("utf-8")).hexdigest()


def load_labels(args):
    if args.frozen_manifest:
        manifest = Path(args.frozen_manifest).expanduser().resolve()
        with np.load(manifest, allow_pickle=False) as payload:
            labels = np.asarray(payload["train_labels"], dtype=np.int64)
            ids = np.asarray(payload["sample_ids"]).astype(str) if "sample_ids" in payload else np.arange(len(labels)).astype(str)
        return labels, ids, manifest.stem, hashlib.sha256(manifest.read_bytes()).hexdigest()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from torchvision import transforms
    from dataset import ImbalanceCIFAR10, ImbalanceCIFAR100

    cls = ImbalanceCIFAR100 if args.data_type == "cifar100lt" else ImbalanceCIFAR10
    ds = cls(root=args.root, imb_type="exp", imb_factor=args.imb_factor, rand_number=0,
             train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    labels = np.asarray(ds.targets, dtype=np.int64)
    ids = np.array([
        hashlib.sha256(np.asarray(ds.data[i]).tobytes() + labels[i].tobytes()).hexdigest()
        for i in range(len(labels))
    ])
    fingerprint = hashlib.sha256(np.asarray(ds.data).tobytes() + labels.tobytes()).hexdigest()
    return labels, ids, args.data_type, fingerprint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-type", required=True, choices=["cifar10lt", "cifar100lt"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--imb-factor", type=float, default=0.01)
    ap.add_argument("--frozen-manifest", default="")
    ap.add_argument("--mode", required=True, choices=[
        "uniform_manifest", "inverse_class_frequency", "sqrt_inverse_class_frequency"
    ])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    labels, ids, dataset_name, fingerprint = load_labels(args)
    if labels.ndim != 1 or np.any(labels < 0):
        raise ValueError("labels must be a non-negative 1D array")
    counts = np.bincount(labels)
    if np.any(counts == 0):
        raise ValueError("labels must be contiguous without empty classes")
    if args.mode == "uniform_manifest":
        weights = np.ones(len(labels), dtype=np.float64)
    elif args.mode == "inverse_class_frequency":
        weights = 1.0 / counts[labels]
    else:
        weights = 1.0 / np.sqrt(counts[labels])
    weights = weights / weights.mean()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, weights.astype(np.float64))
    payload = {
        "dataset_name": dataset_name,
        "dataset_fingerprint": fingerprint,
        "sample_ids_sha256": ids_hash(ids),
        "num_samples": int(len(labels)),
        "weights_file": str(out.resolve()),
        "weights_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "method": args.mode,
        "normalization": "mean_one",
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
        "fine_labels_used_for_training": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Generated in exact upstream/frozen-manifest ordering; sampler uses replacement=True.",
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
