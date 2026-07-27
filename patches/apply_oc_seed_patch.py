#!/usr/bin/env python3
"""Fail-closed deterministic seed and resume-loop patch for official OC."""
from __future__ import annotations
import argparse
from pathlib import Path

MARKER = ".ltx_seed_resume_patch_v2"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("repo"); args = ap.parse_args()
    repo = Path(args.repo).resolve(); path = repo / "main.py"; marker = repo / MARKER
    if marker.exists():
        print("OC v2 patch already applied"); return
    s = path.read_text(encoding="utf-8")
    if "flags.DEFINE_integer('seed'" not in s:
        anchor = "FLAGS = flags.FLAGS\n"
        if anchor not in s: raise RuntimeError("OC flags anchor changed")
        s = s.replace(anchor, anchor + "flags.DEFINE_integer('seed', 0, help='global random seed')\n", 1)
    if "import random\n" not in s:
        if "import os\n" not in s: raise RuntimeError("OC import anchor changed")
        s = s.replace("import os\n", "import os\nimport random\n", 1)
    if "def ltx_set_seed" not in s:
        idx = s.find("def train():")
        if idx < 0: raise RuntimeError("OC train anchor changed")
        helper = (
            "def ltx_set_seed(seed):\n"
            "    random.seed(seed)\n"
            "    np.random.seed(seed)\n"
            "    torch.manual_seed(seed)\n"
            "    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)\n"
            "    torch.backends.cudnn.deterministic = True\n"
            "    torch.backends.cudnn.benchmark = False\n\n"
        )
        s = s[:idx] + helper + s[idx:]
    if "def train():\n    ltx_set_seed(FLAGS.seed)" not in s:
        if "def train():\n" not in s: raise RuntimeError("OC train definition changed")
        s = s.replace("def train():\n", "def train():\n    ltx_set_seed(FLAGS.seed)\n", 1)
    # Upstream resumes weights but restarts the loop at zero; fix only the loop origin.
    if "with trange(FLAGS.ckpt_step, FLAGS.total_steps" not in s:
        anchor = "with trange(0, FLAGS.total_steps, dynamic_ncols=True) as pbar:"
        if anchor not in s: raise RuntimeError("OC training-loop anchor changed")
        s = s.replace(anchor, "with trange(FLAGS.ckpt_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:", 1)
    path.write_text(s, encoding="utf-8")
    marker.write_text("deterministic seed + true checkpoint-step resume\n", encoding="utf-8")
    old = repo / ".ltx_seed_patch"
    if old.exists(): old.unlink()
    print(f"Patched {path}")


if __name__ == "__main__": main()
