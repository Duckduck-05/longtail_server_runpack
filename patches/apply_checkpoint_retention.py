#!/usr/bin/env python3
"""Stop the vendored training loops deleting the previous checkpoint.

All three upstream loops keep only the newest checkpoint, so after a run only
the final step is observable. That makes the training budget unauditable: you
cannot show that a ranking is stable at 150k as well as 300k, and you cannot
re-evaluate an earlier step without retraining from scratch.

This is an I/O-retention change only. The optimisation, the loss, the sampler
and the saved tensors are untouched — the deletion call is replaced by an
opt-out guarded on LTX_KEEP_CHECKPOINTS so the original behaviour is still
one environment variable away when disk is tight.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = ".ltx_checkpoint_retention_patch_v1"

ABSL_OLD = """                prev_ckpt = os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step - FLAGS.save_step))
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
"""
ABSL_NEW = """                # LTX: retain every checkpoint so the training budget stays
                # auditable. Set LTX_KEEP_CHECKPOINTS=0 for upstream behaviour.
                if os.environ.get('LTX_KEEP_CHECKPOINTS', '1') == '0':
                    prev_ckpt = os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step - FLAGS.save_step))
                    if os.path.exists(prev_ckpt):
                        os.remove(prev_ckpt)
"""

CM_OLD = """                prev_ckpt = output_dir / f"ckpt_{step - save_step}.pt"
                if prev_ckpt.exists():
                    prev_ckpt.unlink()
"""
CM_NEW = """                # LTX: retain every checkpoint so the training budget stays
                # auditable. Set LTX_KEEP_CHECKPOINTS=0 for upstream behaviour.
                if os.environ.get("LTX_KEEP_CHECKPOINTS", "1") == "0":
                    prev_ckpt = output_dir / f"ckpt_{step - save_step}.pt"
                    if prev_ckpt.exists():
                        prev_ckpt.unlink()
"""


def patch_file(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"checkpoint-retention anchor missing in {path}")
    text = text.replace(old, new, 1)
    if "import os" not in text:
        raise RuntimeError(f"{path} does not import os; cannot guard the deletion")
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repos_root", type=Path)
    args = parser.parse_args()
    root = args.repos_root.resolve()

    targets = [
        (root / "coral-lt-diffusion" / "main.py", ABSL_OLD, ABSL_NEW),
        (root / "OC_LT" / "main.py", ABSL_OLD, ABSL_NEW),
        (root / "ImbDiff-CM" / "tools" / "train.py", CM_OLD, CM_NEW),
    ]
    changed = 0
    for path, old, new in targets:
        if not path.is_file():
            raise FileNotFoundError(f"expected vendored file missing: {path}")
        if patch_file(path, old, new):
            changed += 1
    (root / MARKER).write_text("checkpoint retention enabled for all training loops\n", encoding="utf-8")
    print(f"[patch] checkpoint retention: {changed} file(s) updated, {len(targets) - changed} already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
