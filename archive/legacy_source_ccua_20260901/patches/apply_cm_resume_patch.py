#!/usr/bin/env python3
"""Operational-only resume patch for the official CM training script.

No model, data, loss, optimizer hyperparameter, or evaluation code is changed.
"""
from __future__ import annotations
import argparse
from pathlib import Path

MARKER = ".ltx_resume_patch_v1"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("repo"); args=ap.parse_args()
    repo=Path(args.repo).resolve(); path=repo/"tools"/"train.py"; marker=repo/MARKER
    if marker.exists(): print("CM resume patch already applied"); return
    s=path.read_text(encoding="utf-8")
    parser_anchor='    parser.add_argument("--device", default=None)\n'
    if 'parser.add_argument("--resume_ckpt"' not in s:
        if parser_anchor not in s: raise RuntimeError("CM parser anchor changed")
        s=s.replace(parser_anchor, parser_anchor+'    parser.add_argument("--resume_ckpt", default=None)\n',1)
    fixed_anchor='''    fixed_x_T = torch.randn(
        min(config["training"]["sample_size"], 100),
        3,
        config["dataset"]["img_size"],
        config["dataset"]["img_size"],
        device=device,
    )
'''
    resume_block=fixed_anchor+'''    start_step = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        net_model.load_state_dict(ckpt["net_model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])
        if "fixed_x_T" in ckpt:
            fixed_x_T = ckpt["fixed_x_T"].to(device)
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"Resumed CM from {args.resume_ckpt} at step {start_step}")
'''
    if 'start_step = 0' not in s:
        if fixed_anchor not in s: raise RuntimeError("CM fixed_x_T anchor changed")
        s=s.replace(fixed_anchor,resume_block,1)
    loop='    with trange(0, total_steps, dynamic_ncols=True) as pbar:\n'
    if 'with trange(start_step, total_steps' not in s:
        if loop not in s: raise RuntimeError("CM loop anchor changed")
        s=s.replace(loop,'    with trange(start_step, total_steps, dynamic_ncols=True) as pbar:\n',1)
    path.write_text(s,encoding="utf-8"); marker.write_text("operational checkpoint resume only\n")
    print(f"Patched {path}")

if __name__=="__main__": main()
