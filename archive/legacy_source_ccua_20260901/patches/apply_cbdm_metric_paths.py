#!/usr/bin/env python3
"""Make the official CBDM/CORAL metric code use a portable feature cache path.

The checked-in source hard-codes one author's /mnt/workspace path.  This patch
changes only the lookup location to LTX_METRICS_ROOT (default ./stats); the
Inception features, PRD algorithm, and reported values are untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = ".ltx_metric_paths_patch_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    path = repo / "score" / "both.py"
    marker = repo / MARKER
    if marker.exists():
        return
    text = path.read_text(encoding="utf-8")
    old10 = "np.load('/mnt/workspace/dlly/ucm3/stats/cifar10_feats.npy')"
    old100 = "np.load('/mnt/workspace/dlly/ucm3/stats/cifar100_feats.npy')"
    new10 = "np.load(os.path.join(os.environ.get('LTX_METRICS_ROOT', './stats'), 'cifar10_feats.npy'))"
    new100 = "np.load(os.path.join(os.environ.get('LTX_METRICS_ROOT', './stats'), 'cifar100_feats.npy'))"
    if old10 not in text or old100 not in text:
        raise RuntimeError("CBDM metric-path patch anchor missing; inspect source before running")
    path.write_text(text.replace(old10, new10).replace(old100, new100), encoding="utf-8")
    marker.write_text("portable LTX_METRICS_ROOT feature-cache lookup only\n", encoding="utf-8")


if __name__ == "__main__":
    main()
