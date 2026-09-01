#!/usr/bin/env python3
"""Idempotent, fail-closed patch for the official CORAL repository.

Adds deterministic seeding, immutable-manifest loading, per-example replacement
sampling, and an evaluation-only sample_only switch. It does not alter the U-Net,
diffusion target, optimizer, losses, EMA, or reverse sampler.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = ".ltx_weighted_sampler_patch_v2"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Patch anchor missing: {label}. Upstream changed; inspect before running.")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    main_py = repo / "main.py"
    marker = repo / MARKER
    if marker.exists():
        print("CORAL v2 patch already applied")
        return
    text = main_py.read_text(encoding="utf-8")

    text = replace_once(
        text, "import os\n", "import os\nimport random\nfrom ltx_manifest_dataset import FrozenManifestDataset\n",
        "imports",
    )
    flag_anchor = "flags.DEFINE_bool('amp', False, help='Use Automatic Mixed Precision for training')\n"
    flag_block = flag_anchor + (
        "flags.DEFINE_integer('seed', 0, help='global random seed')\n"
        "flags.DEFINE_string('sample_weights', '', help='optional .npy per-example sampling weights aligned to dataset order')\n"
        "flags.DEFINE_string('frozen_manifest', '', help='optional immutable NPZ dataset in exact sample order')\n"
        "flags.DEFINE_integer('num_class_override', 0, help='override number of conditional classes')\n"
        "flags.DEFINE_bool('sample_only', False, help='generate/save arrays without built-in CIFAR FID/PRD evaluation')\n"
    )
    text = replace_once(text, flag_anchor, flag_block, "flags")

    device_anchor = "device = torch.device('cuda')\n\n\n"
    helper_block = device_anchor + (
        "def set_seed(seed):\n"
        "    random.seed(seed)\n"
        "    np.random.seed(seed)\n"
        "    torch.manual_seed(seed)\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.manual_seed_all(seed)\n"
        "    torch.backends.cudnn.deterministic = True\n"
        "    torch.backends.cudnn.benchmark = False\n\n\n"
        "def ltx_num_classes():\n"
        "    if FLAGS.num_class_override > 0:\n"
        "        return int(FLAGS.num_class_override)\n"
        "    if FLAGS.frozen_manifest:\n"
        "        with np.load(FLAGS.frozen_manifest, allow_pickle=False) as payload:\n"
        "            labels = np.asarray(payload['train_labels'], dtype=np.int64)\n"
        "        return int(np.max(labels)) + 1\n"
        "    return 100 if 'cifar100' in FLAGS.data_type else 10\n\n\n"
    )
    text = replace_once(text, device_anchor, helper_block, "helpers")

    text = replace_once(text, "def train():\n", "def train():\n    set_seed(FLAGS.seed)\n", "train seed")
    text = replace_once(text, "def eval():\n", "def eval():\n    set_seed(FLAGS.seed)\n", "eval seed")

    dataset_anchor = "    if FLAGS.data_type == 'cifar10':\n        dataset = CIFAR10(\n"
    dataset_block = (
        "    if FLAGS.frozen_manifest:\n"
        "        dataset = FrozenManifestDataset(FLAGS.frozen_manifest, transform=tran_transform, target_transform=None)\n"
        "    elif FLAGS.data_type == 'cifar10':\n"
        "        dataset = CIFAR10(\n"
    )
    text = replace_once(text, dataset_anchor, dataset_block, "frozen dataset")

    old_loader = (
        "    dataloader = torch.utils.data.DataLoader(\n"
        "        dataset, batch_size=FLAGS.batch_size,\n"
        "        shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)\n"
    )
    new_loader = (
        "    ltx_sampler = None\n"
        "    ltx_generator = torch.Generator()\n"
        "    ltx_generator.manual_seed(FLAGS.seed)\n"
        "    if FLAGS.sample_weights:\n"
        "        ltx_weights = np.load(FLAGS.sample_weights, allow_pickle=False).astype(np.float64)\n"
        "        if ltx_weights.ndim != 1 or len(ltx_weights) != len(dataset):\n"
        "            raise ValueError(f'sample_weights shape {ltx_weights.shape} does not match dataset length {len(dataset)}')\n"
        "        if not np.all(np.isfinite(ltx_weights)) or np.any(ltx_weights < 0) or ltx_weights.sum() <= 0:\n"
        "            raise ValueError('sample_weights must be finite, non-negative, and have positive total mass')\n"
        "        ltx_weights = ltx_weights / ltx_weights.mean()\n"
        "        ltx_sampler = torch.utils.data.WeightedRandomSampler(\n"
        "            torch.as_tensor(ltx_weights, dtype=torch.double), num_samples=len(dataset),\n"
        "            replacement=True, generator=ltx_generator)\n"
        "        ltx_ess = float(ltx_weights.sum() ** 2 / np.square(ltx_weights).sum())\n"
        "        print(f'LTX weighted empirical measure: file={FLAGS.sample_weights} n={len(ltx_weights)} ESS={ltx_ess:.3f} min={ltx_weights.min():.6g} max={ltx_weights.max():.6g}')\n"
        "    dataloader = torch.utils.data.DataLoader(\n"
        "        dataset, batch_size=FLAGS.batch_size, shuffle=(ltx_sampler is None), sampler=ltx_sampler,\n"
        "        generator=ltx_generator, num_workers=FLAGS.num_workers, drop_last=True)\n"
    )
    text = replace_once(text, old_loader, new_loader, "dataloader")

    old_num = "FLAGS.num_class = 100 if 'cifar100' in FLAGS.data_type else 10"
    if old_num not in text:
        raise RuntimeError("Patch anchor missing: num_class assignments")
    text = text.replace(old_num, "FLAGS.num_class = ltx_num_classes()")

    metric_anchor = (
        "    (IS, IS_std), FID, prd_score, ipr = get_inception_and_fid_score(\n"
        "        images, labels, FLAGS.fid_cache, num_images=FLAGS.num_images,\n"
        "        use_torch=FLAGS.fid_use_torch, FLAGS=FLAGS)\n"
    )
    metric_block = (
        "    if FLAGS.sample_only:\n"
        "        print('LTX sample_only: arrays and visual grid saved; built-in CIFAR metrics intentionally skipped')\n"
        "        nan_pair = (float('nan'), float('nan'))\n"
        "        return nan_pair, float('nan'), nan_pair, nan_pair\n"
        + metric_anchor
    )
    text = replace_once(text, metric_anchor, metric_block, "sample-only metric guard")

    main_py.write_text(text, encoding="utf-8")
    marker.write_text(
        "deterministic seed + frozen manifest + replacement WeightedRandomSampler + sample_only; no model/loss change\n",
        encoding="utf-8",
    )
    # Remove old marker so preflight cannot mistake a stale patch for v2.
    old_marker = repo / ".ltx_weighted_sampler_patch"
    if old_marker.exists():
        old_marker.unlink()
    print(f"Patched {main_py}")


if __name__ == "__main__":
    main()
