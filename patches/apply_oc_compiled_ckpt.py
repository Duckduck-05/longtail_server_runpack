#!/usr/bin/env python3
"""Let the official OC/T2H evaluator load a torch.compile'd training checkpoint.

``main.py`` wraps the UNet in ``torch.compile`` before training, so every key in
the saved ``net_model`` state dict carries an ``_orig_mod.`` prefix.  ``ddpm_gen.py``
instantiates a plain ``UNet``, so the load raises ``RuntimeError: Error(s) in
loading state_dict for UNet`` and the run dies at the eval phase after training
has already been paid for.  The upstream training code already strips the prefix
in ``ema()`` and this patch does the same thing on the evaluation path; it does
not change weights, sampling, or metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = ".ltx_oc_compiled_ckpt_patch_v1"

HELPER = '''def _strip_compile_prefix(state_dict):
    """torch.compile prefixes every key with ``_orig_mod.``; the evaluator builds
    a plain UNet, so drop the prefix before loading a training checkpoint."""
    return {str(key).replace("_orig_mod.", ""): value for key, value in state_dict.items()}


'''


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"OC compiled-checkpoint anchor missing: {label}")
    return text.replace(old, new, 1)


def patch(repo: Path) -> None:
    path = repo / "ddpm_gen.py"
    text = path.read_text(encoding="utf-8")
    if "_strip_compile_prefix" not in text:
        text = once(text, "def eval():\n", HELPER + "def eval():\n", "eval definition")
    text = once(
        text,
        "    model.load_state_dict(ckpt['net_model'])\n",
        "    model.load_state_dict(_strip_compile_prefix(ckpt['net_model']))\n",
        "net_model load",
    )
    text = once(
        text,
        "    model.load_state_dict(ckpt['ema_model'])\n",
        "    model.load_state_dict(_strip_compile_prefix(ckpt['ema_model']))\n",
        "ema_model load",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    marker = repo / MARKER
    if marker.exists():
        print("OC compiled-checkpoint patch already applied")
        return
    patch(repo)
    marker.write_text("evaluator accepts torch.compile'd net_model/ema_model keys\n", encoding="utf-8")
    print("patched OC evaluator for torch.compile'd checkpoints")


if __name__ == "__main__":
    main()
