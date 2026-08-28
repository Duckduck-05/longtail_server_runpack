#!/usr/bin/env python3
"""Let specific checkpoints survive the vendored trainer's pruning.

`coral-lt-diffusion` keeps exactly one checkpoint: each save deletes the
previous one. That is the right default for a long run on a shared disk, but it
makes intermediate steps unusable as measurement points -- a 300k run silently
destroys its own 200k checkpoint at step 250k, and there is no way to compare a
method against a baseline that stopped at 200k without training a second time.

`PRESERVE_CKPT_STEPS` (comma-separated step numbers) exempts those steps from
pruning. Unset, behaviour is byte-identical to upstream. The same mechanism and
variable name is used by the runpack's training entrypoints, so the retention
policy can be configured without changing the trainer command.

Idempotent; the anchor must match exactly once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "ltx_preserve_ckpt_v1"

ANCHOR = """                prev_ckpt = os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step - FLAGS.save_step))
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)"""

REPLACEMENT = """                prev_ckpt = os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step - FLAGS.save_step))
                # ltx_preserve_ckpt_v1: steps named in PRESERVE_CKPT_STEPS are
                # kept, so a run can be measured at an intermediate budget
                # without being trained twice. Unset -> upstream behaviour.
                _preserve = {
                    int(s) for s in os.environ.get('PRESERVE_CKPT_STEPS', '').split(',') if s.strip()
                }
                if os.path.exists(prev_ckpt) and (step - FLAGS.save_step) not in _preserve:
                    os.remove(prev_ckpt)"""


def apply(repo: Path) -> int:
    main = repo / "main.py"
    if not main.is_file():
        raise SystemExit(f"trainer not found: {main}")
    text = main.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"{main}: already patched")
        return 0
    found = text.count(ANCHOR)
    if found != 1:
        raise SystemExit(
            f"{main}: prune anchor matched {found} times, expected 1 -- "
            "refusing to write a half-patched trainer")
    main.write_text(text.replace(ANCHOR, REPLACEMENT), encoding="utf-8")
    print(f"{main}: patched")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path)
    sys.exit(apply(p.parse_args().repo))
