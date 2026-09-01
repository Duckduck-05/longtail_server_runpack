#!/usr/bin/env python3
"""Make CORAL and T2H evaluation labels exactly class-uniform.

The official evaluators draw conditional labels iid.  That is fine for visual
sampling, but it is not a valid input to a class-balanced 50k metric table:
each method must be scored on exactly the same label support.  This patch only
adds an opt-in evaluation flag; it leaves training and the reverse process
unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = ".ltx_uniform_eval_labels_patch_v1"


def once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"uniform-label patch anchor missing: {label}")
    return text.replace(old, new, 1)


def patch_coral(repo: Path) -> None:
    main = repo / "main.py"
    diffusion = repo / "diffusion.py"
    text = main.read_text(encoding="utf-8")
    flag = "flags.DEFINE_bool('sample_only', False, help='generate/save arrays without built-in CIFAR FID/PRD evaluation')\n"
    if "flags.DEFINE_bool('uniform_labels'" not in text:
        text = once(text, flag, flag + "flags.DEFINE_bool('uniform_labels', False, help='sample an exact class-uniform label schedule during evaluation')\n", "CORAL flag")
    old_call = (
        "                batch_images, batch_labels = sampler(x_T.to(device),\n"
        "                                                     omega=FLAGS.omega,\n"
        "                                                     method=FLAGS.sample_method)\n"
    )
    new_call = (
        "                forced_labels = None\n"
        "                if FLAGS.uniform_labels and FLAGS.sample_method != 'uncond':\n"
        "                    forced_labels = torch.arange(i, i + batch_size, device=device) % FLAGS.num_class\n"
        "                batch_images, batch_labels = sampler(x_T.to(device),\n"
        "                                                     omega=FLAGS.omega,\n"
        "                                                     method=FLAGS.sample_method,\n"
        "                                                     labels=forced_labels)\n"
    )
    if "forced_labels = None" not in text:
        text = once(text, old_call, new_call, "CORAL evaluator call")
    main.write_text(text, encoding="utf-8")

    text = diffusion.read_text(encoding="utf-8")
    if "def forward(self, x_T, omega=0.0, method='cfg', labels=None):" not in text:
        text = once(text, "def forward(self, x_T, omega=0.0, method='cfg'):", "def forward(self, x_T, omega=0.0, method='cfg', labels=None):", "CORAL sampler signature")
    old_labels = (
        "        if method == 'uncond':\n"
        "            y = None\n"
        "        else:\n"
        "            y = torch.randint(0, self.num_class, (len(x_t),)).to(x_t.device)\n"
    )
    new_labels = (
        "        if method == 'uncond':\n"
        "            if labels is not None:\n"
        "                raise ValueError('unconditional sampling cannot accept labels')\n"
        "            y = None\n"
        "        elif labels is not None:\n"
        "            if labels.shape != (len(x_t),):\n"
        "                raise ValueError(f'labels must have shape ({len(x_t)},), got {tuple(labels.shape)}')\n"
        "            y = labels.to(x_t.device, dtype=torch.long)\n"
        "        else:\n"
        "            y = torch.randint(0, self.num_class, (len(x_t),)).to(x_t.device)\n"
    )
    if "unconditional sampling cannot accept labels" not in text:
        text = once(text, old_labels, new_labels, "CORAL sampler labels")
    diffusion.write_text(text, encoding="utf-8")


def patch_oc(repo: Path) -> None:
    path = repo / "ddpm_gen.py"
    text = path.read_text(encoding="utf-8")
    flag = "flags.DEFINE_string('sample_output', '', help='absolute .npy path for generated arrays')\n"
    if "flags.DEFINE_bool('uniform_labels'" not in text:
        text = once(text, flag, flag + "flags.DEFINE_bool('uniform_labels', False, help='sample an exact class-uniform label schedule')\n", "T2H flag")
    old_labels = "            y = torch.randint(FLAGS.num_class, size=(x_T.shape[0], ),device=device)\n"
    new_labels = (
        "            if FLAGS.uniform_labels:\n"
        "                y = torch.arange(i, i + batch_size, device=device) % FLAGS.num_class\n"
        "            else:\n"
        "                y = torch.randint(FLAGS.num_class, size=(x_T.shape[0], ),device=device)\n"
    )
    if "if FLAGS.uniform_labels:" not in text:
        text = once(text, old_labels, new_labels, "T2H evaluator labels")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repos_root", type=Path)
    args = parser.parse_args()
    root = args.repos_root.resolve()
    marker = root / MARKER
    if marker.exists():
        print("uniform evaluation-label patch already applied")
        return
    patch_coral(root / "coral-lt-diffusion")
    patch_oc(root / "OC_LT")
    marker.write_text("opt-in exact class-uniform conditional evaluation labels\n", encoding="utf-8")
    print("patched exact class-uniform evaluation labels")


if __name__ == "__main__":
    main()
