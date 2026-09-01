#!/usr/bin/env python3
"""Replace an author-local OC FID checkpoint path with the public upstream URL."""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = ".ltx_oc_metric_weights_patch_v1"
UPSTREAM = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    path = repo / "score" / "inception.py"
    marker = repo / MARKER
    if marker.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "from torch.hub import load_state_dict_from_url" not in text:
        if "import torch\n" not in text:
            raise RuntimeError("OC metric patch import anchor missing")
        text = text.replace("import torch\n", "import torch\nfrom torch.hub import load_state_dict_from_url\n", 1)
    old_url = "FID_WEIGHTS_URL = '/remote-home/tianjiaozhang/LT_diffusion/yiming_baseline/CBDM-pytorch-main/pt_inception-2015-12-05-6726825d.pth'"
    if old_url not in text or "state_dict = torch.load(FID_WEIGHTS_URL)" not in text:
        raise RuntimeError("OC metric patch anchor missing; inspect source before running")
    text = text.replace(old_url, f"FID_WEIGHTS_URL = '{UPSTREAM}'")
    text = text.replace("state_dict = torch.load(FID_WEIGHTS_URL)", "state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)")
    path.write_text(text, encoding="utf-8")
    marker.write_text("public FID-Inception checkpoint URL only\n", encoding="utf-8")


if __name__ == "__main__":
    main()
