#!/usr/bin/env python3
"""Install the IP-SVT auxiliary objective into the vendored coral-lt-diffusion trainer.

IP-SVT is compared against the ``ddpm`` baseline, and in this campaign that
baseline is this repository's ``main.py``. So IP-SVT runs the *same* file with
extra flags rather than a fork: the arms then differ by the auxiliary objective
and nothing else -- not the data pipeline, not the schedule, not the sampler,
not the metric path.

Idempotent, and anchored on exact unique strings: a missing or ambiguous anchor
aborts instead of writing a half-patched trainer.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FLAGS_ANCHOR = "flags.DEFINE_bool('amp', False, help='Use Automatic Mixed Precision for training')"
FLAGS_BLOCK = """

# --- IP-SVT auxiliary objective (tools/ipsvt_aux.py in the Longtail repo)
flags.DEFINE_bool('ipsvt', False, help='enable the IP-SVT class-uniform auxiliary branch')
flags.DEFINE_enum('ipsvt_mode', 'full', ['full', 'twin', 'clean'],
                  help='full=twin+SVT, twin=twin only, clean=plain class-uniform DDPM control')
flags.DEFINE_float('ipsvt_lambda_aux', 1.0, help='weight of the whole auxiliary objective')
flags.DEFINE_float('ipsvt_lambda_svt', 1.0, help='weight of L_SVT inside the auxiliary objective')
flags.DEFINE_integer('ipsvt_K', 4, help='stochastic response directions per anchor')
flags.DEFINE_float('ipsvt_s', 0.05, help='dimensionless condition-perturbation radius')
flags.DEFINE_float('ipsvt_delta', 0.1, help='twin noise radius; must satisfy delta < 1/sqrt(2)')
flags.DEFINE_integer('ipsvt_every', 4, help='run the auxiliary branch every q ordinary updates')
flags.DEFINE_integer('ipsvt_batch', 16, help='class-uniform auxiliary batch size')"""

INIT_ANCHOR = "    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)"
INIT_BLOCK = '''    ipsvt_aux_branch = None
    if FLAGS.ipsvt:
        if FLAGS.amp:
            # The auxiliary branch is not wired into the AMP path. Failing here
            # is the point: silently skipping it would produce an "IP-SVT" run
            # that trained the baseline objective.
            raise ValueError('--ipsvt is not supported with --amp')
        from ipsvt_aux import IPSVTAuxiliary
        _raw = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float().div_(255.0)
        ipsvt_aux_branch = IPSVTAuxiliary(
            images=_raw.mul_(2.0).sub_(1.0), targets=dataset.targets,
            num_class=FLAGS.num_class, T=FLAGS.T,
            beta_1=FLAGS.beta_1, beta_T=FLAGS.beta_T,
            K=FLAGS.ipsvt_K, s=FLAGS.ipsvt_s, delta=FLAGS.ipsvt_delta,
            batch_size=FLAGS.ipsvt_batch, lambda_svt=FLAGS.ipsvt_lambda_svt,
            lambda_aux=FLAGS.ipsvt_lambda_aux, every=FLAGS.ipsvt_every,
            mode=FLAGS.ipsvt_mode, device=device, seed=FLAGS.seed)
        del _raw

'''

LOSS_ANCHOR = """                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                optim.step()"""
LOSS_BLOCK = """                ipsvt_stats = None
                if ipsvt_aux_branch is not None:
                    _aux = ipsvt_aux_branch(net_model, step)
                    if _aux is not None:
                        _twin, _svt, ipsvt_stats = _aux
                        loss = loss + FLAGS.ipsvt_lambda_aux * (
                            _twin + FLAGS.ipsvt_lambda_svt * _svt)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                optim.step()"""

LOG_ANCHOR = "            writer.add_scalar('loss_ddpm', loss_ddpm, step)"
LOG_BLOCK = """
            if ipsvt_aux_branch is not None and ipsvt_stats is not None:
                for _k, _v in ipsvt_stats.items():
                    writer.add_scalar(_k, _v, step)"""


def apply(repo: Path, module: Path) -> int:
    main = repo / "main.py"
    if not main.is_file():
        raise SystemExit(f"trainer not found: {main}")
    if not module.is_file():
        raise SystemExit(f"auxiliary module not found: {module}")

    # The module is copied, never symlinked: the run must not depend on a path
    # outside the repo that a later checkout could break.
    shutil.copyfile(module, repo / "ipsvt_aux.py")

    text = main.read_text(encoding="utf-8")
    if "IP-SVT auxiliary objective" in text:
        print(f"{main}: already patched")
        return 0

    for label, anchor in (("flags", FLAGS_ANCHOR), ("init", INIT_ANCHOR),
                          ("loss", LOSS_ANCHOR), ("log", LOG_ANCHOR)):
        found = text.count(anchor)
        if found != 1:
            raise SystemExit(
                f"{main}: {label} anchor matched {found} times, expected 1 -- "
                "refusing to write a half-patched trainer")

    text = text.replace(FLAGS_ANCHOR, FLAGS_ANCHOR + FLAGS_BLOCK)
    text = text.replace(INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR)
    text = text.replace(LOSS_ANCHOR, LOSS_BLOCK)
    text = text.replace(LOG_ANCHOR, LOG_ANCHOR + LOG_BLOCK)
    main.write_text(text, encoding="utf-8")
    print(f"{main}: patched")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path, help="path to the coral-lt-diffusion checkout")
    p.add_argument("--module", type=Path,
                   default=Path(__file__).resolve().parent / "ipsvt_aux.py",
                   help="canonical auxiliary module to install")
    args = p.parse_args()
    sys.exit(apply(args.repo, args.module))
