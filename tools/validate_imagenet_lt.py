#!/usr/bin/env python3
"""Fail early on mismatched licensed ImageNet-LT roots/manifests."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx.preflight import _validate_imagenet_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--reference-manifest", required=True)
    args = parser.parse_args()
    root = Path(args.image_root).resolve()
    for manifest, name, balanced in (
        (Path(args.train_manifest).resolve(), "train", False),
        (Path(args.reference_manifest).resolve(), "reference", True),
    ):
        counts = _validate_imagenet_manifest(root, manifest, name, require_balanced=balanced)
        print(f"{name}: n={sum(counts.values())}, classes=1000, min/class={min(counts.values())}, max/class={max(counts.values())}")


if __name__ == "__main__":
    main()
