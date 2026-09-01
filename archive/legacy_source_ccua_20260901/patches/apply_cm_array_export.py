#!/usr/bin/env python3
"""Add CM sample-array export for the shared CIFAR evaluator only."""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = ".ltx_cm_array_export_patch_v1"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"CM array-export patch anchor missing: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("repo", type=Path); args = parser.parse_args()
    repo = args.repo.resolve(); path = repo / "tools" / "sample_images.py"; marker = repo / MARKER
    if marker.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "import numpy as np" not in text:
        text = once(text, "import torch\n", "import torch\nimport numpy as np\n", "numpy import")
    flag = '    parser.add_argument("--device", default=None)\n'
    if 'parser.add_argument("--samples_output"' not in text:
        text = once(text, flag, flag + '    parser.add_argument("--samples_output", default=None, help="optional float32 NCHW .npy export")\n    parser.add_argument("--labels_output", default=None, help="optional int64 .npy export aligned with samples_output")\n', "flags")
    num = "    num_images = num_per_class * num_classes\n\n    output_dir.mkdir(parents=True, exist_ok=True)\n"
    new_num = "    num_images = num_per_class * num_classes\n    if bool(args.samples_output) != bool(args.labels_output):\n        raise ValueError(\"--samples_output and --labels_output must be supplied together\")\n\n    output_dir.mkdir(parents=True, exist_ok=True)\n    sample_output = None\n    labels_output = None\n    if args.samples_output:\n        sample_path = Path(args.samples_output)\n        labels_path = Path(args.labels_output)\n        sample_path.parent.mkdir(parents=True, exist_ok=True)\n        labels_path.parent.mkdir(parents=True, exist_ok=True)\n        sample_output = np.lib.format.open_memmap(sample_path, mode=\"w+\", dtype=np.float32, shape=(num_images, 3, config[\"dataset\"][\"img_size\"], config[\"dataset\"][\"img_size\"]))\n        labels_output = np.empty(num_images, dtype=np.int64)\n"
    if "sample_output = None" not in text:
        text = once(text, num, new_num, "array initialization")
    image = "            images = (images + 1) / 2\n"
    new_image = image + "            if sample_output is not None:\n                sample_output[start:start + current] = images.numpy().astype(np.float32, copy=False)\n                labels_output[start:start + current] = np.asarray(labels, dtype=np.int64)\n"
    if "sample_output[start:start + current]" not in text:
        text = once(text, image, new_image, "array write")
    end = "                for future in futures:\n                    future.result()\n\n\nif __name__ == \"__main__\":\n"
    new_end = "                for future in futures:\n                    future.result()\n    if sample_output is not None:\n        sample_output.flush()\n        np.save(args.labels_output, labels_output)\n\n\nif __name__ == \"__main__\":\n"
    if "sample_output.flush()" not in text:
        text = once(text, end, new_end, "array finalize")
    path.write_text(text, encoding="utf-8")
    marker.write_text("optional direct float32 sample/label export; no sampling change\n", encoding="utf-8")


if __name__ == "__main__":
    main()
