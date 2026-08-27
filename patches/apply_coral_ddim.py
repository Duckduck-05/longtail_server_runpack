#!/usr/bin/env python3
"""Give the vendored coral-lt-diffusion sampler a DDIM path.

The unified campaign evaluates every method under one shared sampler, so that
the table compares methods rather than the samplers their papers happened to
ship with. CM/OC/CCUA all reach that sampler through their own `ddim_skip_step`
flag; CBDM has `forward_ddim`; coral-lt-diffusion had no DDIM at all, which is
why the campaign previously had to normalise everyone onto the 1000-step
ancestral sampler -- twenty times the cost of what any of these papers actually
run.

Two implementation choices matter:

* the branch lives *inside* ``GaussianDiffusionSampler.forward``, selected by a
  constructor argument, so no call site changes. That is what lets this patch
  compose with ``apply_uniform_eval_labels.py``, which rewrites the same
  method's signature to take an explicit ``labels`` tensor;
* the update rule is copied from CBDM's ``forward_ddim`` (eta = 0, evenly spaced
  subsequence, x0 clipped to [-1,1]) so both repositories run the *same*
  sampler, not two independent implementations that happen to share a name.

Idempotent; anchors must match exactly once or the patch aborts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "ltx_coral_ddim_v1"

INIT_ANCHOR = ("    def __init__(self, model, beta_1, beta_T, T, num_class, "
               "img_size=32, var_type='fixedlarge'):")
INIT_NEW = ("    def __init__(self, model, beta_1, beta_T, T, num_class, "
            "img_size=32, var_type='fixedlarge', ddim_steps=0):")

STORE_ANCHOR = """        self.var_type = var_type
"""
STORE_NEW = """        self.var_type = var_type
        # ltx_coral_ddim_v1: 0 keeps the original ancestral sampler.
        self.ddim_steps = int(ddim_steps)
"""

FORWARD_ANCHOR = """        with torch.no_grad():
            for time_step in tqdm(reversed(range(0, self.T)), total=self.T):
                t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y,
                                                     omega=omega, method=method)

                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                
                x_t = mean + torch.exp(0.5 * log_var) * noise

        return torch.clip(x_t, -1, 1), y"""

FORWARD_NEW = '''        if self.ddim_steps > 0:
            # DDIM (Song et al. 2021), eta = 0. Same update as CBDM's
            # forward_ddim so the two repositories share one sampler.
            seq = torch.linspace(0, self.T - 1, steps=self.ddim_steps).long()
            seq = torch.unique(seq, sorted=True).flip(0).tolist()
            augm = torch.zeros((x_t.shape[0], 9)).to(x_t.device)
            with torch.no_grad():
                for i, time_step in enumerate(tqdm(seq, total=len(seq), desc='DDIM')):
                    t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                    eps, _, _ = self.model(x_t, t, y=y, augm=augm)
                    if omega > 0 and method == 'cfg':
                        unc_eps, _, _ = self.model(x_t, t, y=None, augm=None)
                        eps = eps + omega * (eps - unc_eps)
                    x_0 = torch.clip(self.predict_xstart_from_eps(x_t, t, eps=eps), -1., 1.)
                    if i + 1 < len(seq):
                        abar_next = self.alphas_bar[seq[i + 1]]
                    else:
                        abar_next = torch.tensor(1.0, device=x_t.device)  # t = -1
                    abar_next = abar_next.to(x_t.dtype)
                    x_t = torch.sqrt(abar_next) * x_0 + torch.sqrt(1 - abar_next) * eps
            return torch.clip(x_t, -1, 1), y

        with torch.no_grad():
            for time_step in tqdm(reversed(range(0, self.T)), total=self.T):
                t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y,
                                                     omega=omega, method=method)

                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                
                x_t = mean + torch.exp(0.5 * log_var) * noise

        return torch.clip(x_t, -1, 1), y'''

FLAG_ANCHOR = "flags.DEFINE_string('sample_method', 'cfg', help='sampling method, must be in [cfg, cond, uncond]')"
FLAG_NEW = (FLAG_ANCHOR + "\nflags.DEFINE_integer('ddim_steps', 0, "
            "help='ltx_coral_ddim_v1: if >0, sample with this many DDIM steps instead of the full T-step ancestral chain')\n")

# The whole call, not just its opening line: the arguments sit on the following
# line, so inserting a keyword right after "GaussianDiffusionSampler(" puts it
# ahead of the positional arguments and the file stops parsing.
CTOR_ANCHOR = ("    sampler = GaussianDiffusionSampler(\n"
               "        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, "
               "FLAGS.img_size, FLAGS.var_type).to(device)")
CTOR_NEW = ("    sampler = GaussianDiffusionSampler(\n"
            "        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, "
            "FLAGS.img_size, FLAGS.var_type,\n"
            "        ddim_steps=FLAGS.ddim_steps).to(device)")


def _once(text: str, old: str, new: str, label: str) -> str:
    found = text.count(old)
    if found != 1:
        raise SystemExit(f"coral DDIM patch: {label} anchor matched {found} times, expected 1")
    return text.replace(old, new)


def apply(repo: Path) -> int:
    diffusion, main = repo / "diffusion.py", repo / "main.py"
    for path in (diffusion, main):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    # Each file is checked and patched independently. A single shared marker
    # would let a run that patched diffusion.py and then failed on main.py
    # report "already applied" forever, leaving the sampler unreachable.
    text = diffusion.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"{diffusion}: already patched")
    else:
        text = _once(text, INIT_ANCHOR, INIT_NEW, "sampler __init__")
        text = _once(text, STORE_ANCHOR, STORE_NEW, "sampler state")
        text = _once(text, FORWARD_ANCHOR, FORWARD_NEW, "sampler forward")
        diffusion.write_text(text, encoding="utf-8")
        print(f"{diffusion}: patched")

    text = main.read_text(encoding="utf-8")
    if "ddim_steps" in text:
        print(f"{main}: already patched")
        return 0
    text = _once(text, FLAG_ANCHOR, FLAG_NEW, "ddim_steps flag")
    # Every sampler construction gets the argument. During training the flag is
    # absent from the command line and defaults to 0, so the ancestral sampler
    # is what the training-time preview uses, exactly as before.
    count = text.count(CTOR_ANCHOR)
    if count == 0:
        raise SystemExit("coral DDIM patch: no eval-side GaussianDiffusionSampler construction found")
    text = text.replace(CTOR_ANCHOR, CTOR_NEW)
    main.write_text(text, encoding="utf-8")
    print(f"{repo}: coral DDIM patch applied ({count} sampler constructions)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path)
    sys.exit(apply(p.parse_args().repo))
