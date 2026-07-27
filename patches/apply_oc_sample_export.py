#!/usr/bin/env python3
"""Add source-native generated-array export to the official OC/T2H evaluator."""
from __future__ import annotations
import argparse
from pathlib import Path
MARKER = ".ltx_oc_sample_export_v1"
def once(text, old, new, label):
    if new in text: return text
    if old not in text: raise RuntimeError(f"OC sample-export anchor missing: {label}")
    return text.replace(old, new, 1)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("repo"); args=parser.parse_args()
    repo=Path(args.repo).resolve(); path=repo/"ddpm_gen.py"; marker=repo/MARKER
    if marker.exists(): return
    text=path.read_text(encoding="utf-8")
    anchor="flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')"
    if "flags.DEFINE_bool('sample_only'" not in text:
        if anchor not in text: raise RuntimeError("OC sample-export anchor missing: flags")
        pos=text.index(anchor)+len(anchor)
        text=text[:pos]+"\nflags.DEFINE_bool('sample_only', False, help='save generated arrays without upstream metrics')\nflags.DEFINE_string('sample_output', '', help='absolute .npy path for generated arrays')"+text[pos:]
    text=once(text,"            images.append((batch_images + 1) / 2)\n","            images.append((batch_images + 1) / 2)\n            labels.append(y.cpu())\n","labels")
    anchor="        images = torch.cat(images, dim=0).numpy(); #labels = torch.cat(labels, dim=0).cpu().numpy()\n"
    replacement="        images = torch.cat(images, dim=0).numpy()\n        labels = torch.cat(labels, dim=0).numpy()\n        if FLAGS.sample_output:\n            np.save(FLAGS.sample_output, images)\n            np.save(FLAGS.sample_output + '.labels.npy', labels)\n        if FLAGS.sample_only:\n            return (float('nan'), float('nan')), float('nan'), images\n"
    text=once(text,anchor,replacement,"sample output")
    path.write_text(text,encoding="utf-8"); marker.write_text("generated arrays/labels export only\n",encoding="utf-8")
if __name__ == "__main__": main()
