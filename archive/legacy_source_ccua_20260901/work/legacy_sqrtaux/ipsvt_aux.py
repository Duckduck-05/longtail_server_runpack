"""IP-SVT auxiliary objective: twin-condition DDPM + stochastic-response transfer.

Spec: ``Plan/IPSVT_PREEXPERIMENT_FINAL.md`` sections 5.1-5.4.

The full objective this plugs into is

    L = L_DDPM^natural  +  lambda_aux * ( L_twin + lambda * L_SVT )^class-uniform

so the ordinary DDPM branch keeps its natural long-tailed sampling and only the
sparse auxiliary branch is class-uniform. Nothing here touches the architecture
or the sampler, and inference cost is unchanged.

Three implementation choices are load-bearing and are fixed here so they cannot
drift:

* **One global perturbation radius.** ``eta ~ N(0, s^2 r_e^2 I / d)`` with
  ``r_e`` the mean embedding norm over *all* classes, stop-gradient, tracked by a
  slow EMA. A per-class radius would make the perturbation itself encode class
  frequency -- the method would then "work" for a reason that has nothing to do
  with the condition Jacobian.

* **Dropout is disabled for the auxiliary branch.** The two condition views must
  differ *only* by ``eta``. With dropout live, the clean and perturbed forwards
  would also differ by their dropout masks, and ``L_SVT`` would spend its
  gradient making the model robust to dropout rather than to the condition. The
  module is restored to its previous mode afterwards, so the ordinary branch is
  untouched. (The UNet uses GroupNorm, which has no running statistics, so
  ``eval()`` changes nothing else.)

* **Activations are recomputed, not stored.** The branch holds
  ``2 * (K+1)`` forward graphs at once, which does not fit beside the ordinary
  branch on a shared GPU. Each forward is therefore gradient-checkpointed. This
  is safe *because* dropout is already disabled: recomputation in the backward
  pass reproduces the forward exactly, which would not be true with live
  dropout.

* **The clean geometry is a stopped teacher.** ``L_SVT`` matches the perturbed
  Gram to ``sg(G(e_y))``. Without the stop-gradient both views could meet at a
  shared degenerate geometry, which is exactly the collapse the term exists to
  prevent.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

TAU_R_DEFAULT = 1e-6


def forward_with_rows(model, x: Tensor, t: Tensor, e_rows: Tensor) -> Tensor:
    """``UNet.forward`` with the class embedding replaced by explicit rows.

    Mirrors the vendored forward exactly; the only substitution is at the single
    fan-out point ``temb = time_embedding(t) + label_embedding(y)``. The module
    is never mutated, so this stays safe alongside the ordinary branch.

    ``augm`` is not supported: every IP-SVT run trains ``--noaugm``, and silently
    dropping an augmentation embedding would change the objective.
    """
    from model.model import ResBlock

    if getattr(model, "augm_embedding", None) is not None:
        raise RuntimeError("IP-SVT auxiliary branch does not support --augm")

    temb = model.time_embedding(t) + e_rows
    h = model.head(x)
    hs = [h]
    for layer in model.downblocks:
        h = layer(h, temb)
        hs.append(h)
    for layer in model.middleblocks:
        h = layer(h, temb)
    for layer in model.upblocks:
        if isinstance(layer, ResBlock):
            h = torch.cat([h, hs.pop()], dim=1)
        h = layer(h, temb)
    h = model.tail(h)
    if hs:
        raise RuntimeError(f"skip-connection stack not consumed: {len(hs)} left")
    return h


def response_gram(responses, tau_r: float = TAU_R_DEFAULT):
    """Normalised response Gram ``G = U U^T`` per anchor, plus a validity mask.

    An anchor whose response norm fell below ``tau_r`` carries numerical noise,
    not geometry, so it is masked out rather than clamped.
    """
    stacked = torch.stack(responses, dim=1)              # (N, K, D)
    norms = stacked.norm(dim=2)                          # (N, K)
    valid = (norms >= tau_r).all(dim=1)
    units = stacked / norms.clamp_min(tau_r).unsqueeze(2)
    return units @ units.transpose(1, 2), valid


def off_diagonal(gram: Tensor) -> Tensor:
    k = gram.shape[1]
    mask = ~torch.eye(k, dtype=torch.bool, device=gram.device)
    return gram * mask


def gram_discrepancy(gram_a: Tensor, gram_b: Tensor) -> Tensor:
    """``||Off(G_a - G_b)||_F^2 / (K(K-1))`` -- the optimised IP-SVT quantity."""
    k = gram_a.shape[1]
    return off_diagonal(gram_a - gram_b).pow(2).sum(dim=(1, 2)) / (k * (k - 1))


class IPSVTAuxiliary:
    """Class-uniform auxiliary branch producing the auxiliary DDPM term and ``L_SVT``.

    ``mode`` selects which auxiliary objective is built on the class-uniform batch:

    * ``"full"``  -- twin-condition DDPM plus the response-geometry term;
    * ``"twin"``  -- twin-condition DDPM only (caller sets ``lambda_svt = 0``);
    * ``"clean"`` -- the *ordinary* DDPM loss on the clean condition and the anchor
      state alone. This is the attribution control: it gives the auxiliary branch
      exactly the same class-uniform data at exactly the same cadence, with no
      condition perturbation at all. If tail gains survive against it, they come
      from the twin/SVT mechanism rather than from tail classes simply receiving
      extra auxiliary exposure.

    ``clean`` deliberately matches the *data* the auxiliary branch touches, not its
    FLOPs: it runs one forward where ``twin`` runs ``2(K+1)``. Exposure to tail
    examples is the confound being removed here; compute is not.
    """

    MODES = ("full", "twin", "clean")

    def __init__(self, *, images: Tensor, targets, num_class: int, T: int,
                 beta_1: float, beta_T: float, K: int = 4, s: float = 0.05,
                 delta: float = 0.1, batch_size: int = 32,
                 lambda_svt: float = 1.0, lambda_aux: float = 1.0,
                 every: int = 4, r_e_ema: float = 0.999,
                 tau_r: float = TAU_R_DEFAULT, device=None, seed: int = 0,
                 use_checkpoint: bool = True, mode: str = "full",
                 sampler: str = "uniform", warmup_steps: int = 0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}; got {mode!r}")
        if sampler not in ("uniform", "sqrt"):
            raise ValueError(f"sampler must be 'uniform' or 'sqrt'; got {sampler!r}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative; got {warmup_steps}")
        if not 0 < delta < 2 ** -0.5:
            raise ValueError(f"delta must satisfy 0 < delta < 1/sqrt(2); got {delta}")
        if K < 2:
            raise ValueError(f"K >= 2 is needed for a response geometry; got {K}")
        self.device = device or images.device
        self.images = images.to(self.device)
        self.num_class = num_class
        self.K, self.s, self.delta = K, s, delta
        self.batch_size = batch_size
        self.lambda_svt, self.lambda_aux = lambda_svt, lambda_aux
        self.every, self.tau_r = every, tau_r
        self.use_checkpoint = use_checkpoint
        self.mode = mode
        self.sampler = sampler
        self.warmup_steps = int(warmup_steps)
        self.r_e_ema = r_e_ema
        self._r_e = None

        targets = np.asarray(targets)
        self.class_index = [
            torch.as_tensor(np.flatnonzero(targets == c), dtype=torch.long, device=self.device)
            for c in range(num_class)
        ]
        empty = [c for c, idx in enumerate(self.class_index) if len(idx) == 0]
        if empty:
            raise ValueError(f"class-uniform sampling needs every class present; missing {empty}")
        counts = torch.tensor([len(idx) for idx in self.class_index],
                              dtype=torch.float32, device=self.device)
        if sampler == "uniform":
            self.class_probs = torch.full_like(counts, 1.0 / self.num_class)
        else:
            self.class_probs = counts.sqrt()
            self.class_probs = self.class_probs / self.class_probs.sum()

        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64, device=self.device)
        alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_alphas_bar = alphas_bar.sqrt().float()
        self.sqrt_one_minus_alphas_bar = (1.0 - alphas_bar).sqrt().float()
        self.T = T
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    # ------------------------------------------------------------------ data

    def sample_batch(self):
        """Class-uniform: each slot draws a class first, then an image from it."""
        if self.sampler == "uniform":
            y = torch.randint(self.num_class, (self.batch_size,),
                              generator=self.generator, device=self.device)
        else:
            y = torch.multinomial(self.class_probs, self.batch_size,
                                  replacement=True, generator=self.generator)
        pick = torch.empty(self.batch_size, dtype=torch.long, device=self.device)
        for i, c in enumerate(y.tolist()):
            pool = self.class_index[c]
            j = torch.randint(len(pool), (1,), generator=self.generator, device=self.device)
            pick[i] = pool[j]
        x0 = self.images[pick]
        flip = torch.rand(self.batch_size, generator=self.generator, device=self.device) < 0.5
        x0 = torch.where(flip.view(-1, 1, 1, 1), x0.flip(-1), x0)
        return x0, y

    # ------------------------------------------------------------- condition

    def embedding_scale(self, model) -> float:
        """Global ``r_e``, stop-gradient, smoothed by a slow EMA (spec 5.1)."""
        with torch.no_grad():
            current = float(model.label_embedding.weight[:self.num_class]
                            .detach().pow(2).sum(dim=1).mean().sqrt())
        if self._r_e is None:
            self._r_e = current
        else:
            self._r_e = self.r_e_ema * self._r_e + (1.0 - self.r_e_ema) * current
        return self._r_e

    # ------------------------------------------------------------------ loss

    def __call__(self, model, step: int):
        """Return ``(loss_twin, loss_svt, stats)``, or ``None`` on skipped steps."""
        if step < self.warmup_steps:
            return None
        if self.every > 1 and step % self.every != 0:
            return None

        x0, y = self.sample_batch()
        n = len(y)
        r_e = self.embedding_scale(model)
        emb = model.label_embedding.weight
        d = emb.shape[1]

        t = torch.randint(self.T, (n,), generator=self.generator, device=self.device)
        a_t = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
        sig_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)

        if self.mode == "clean":
            eps0 = torch.randn(x0.shape, generator=self.generator, device=self.device)
            x_t = a_t * x0 + sig_t * eps0
            was_training = model.training
            model.eval()
            try:
                pred = forward_with_rows(model, x_t, t, emb[y])
            finally:
                if was_training:
                    model.train()
            loss_clean = ((pred.float() - eps0) ** 2).mean(dim=(1, 2, 3)).mean()
            zero = torch.zeros((), device=self.device, dtype=loss_clean.dtype)
            return loss_clean, zero, {
                "ipsvt_r_e": r_e,
                "ipsvt_valid_fraction": 1.0,
                "ipsvt_twin": float(loss_clean.detach()),
                "ipsvt_svt": 0.0,
            }

        # K+1 forward-valid states sharing one instance: an anchor plus K twins
        # eps^(k) = sqrt(1-delta^2) eps^(0) + delta xi_k, each with a KNOWN noise,
        # so every branch is trained against an exact DDPM target (spec 5.2).
        eps = [torch.randn(x0.shape, generator=self.generator, device=self.device)]
        for _ in range(self.K):
            xi = torch.randn(x0.shape, generator=self.generator, device=self.device)
            eps.append((1.0 - self.delta ** 2) ** 0.5 * eps[0] + self.delta * xi)
        states = [a_t * x0 + sig_t * e for e in eps]

        e_clean = emb[y]
        eta = torch.randn(n, d, generator=self.generator, device=self.device) * (self.s * r_e / d ** 0.5)
        views = [e_clean, e_clean + eta]

        was_training = model.training
        model.eval()                       # see module docstring: dropout must not
        try:                               # differ between the two condition views
            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint

                def _eval_forward(x, e):
                    # The eval() switch must live INSIDE the checkpointed
                    # function, not around the forward pass. Recomputation
                    # happens during backward, long after the enclosing
                    # try/finally has restored train(); with dropout live at
                    # that moment the recomputed graph saves different tensors
                    # than the forward did, and torch raises a metadata
                    # mismatch. Re-asserting eval() here makes the two passes
                    # identical whenever they run.
                    previous = model.training
                    model.eval()
                    try:
                        return forward_with_rows(model, x, t, e)
                    finally:
                        if previous:
                            model.train()

                def run(xk, view):
                    # Only tensors are passed as checkpoint arguments; the
                    # module and the timestep are captured.
                    return checkpoint(_eval_forward, xk, view, use_reentrant=False)
            else:
                def run(xk, view):
                    return forward_with_rows(model, xk, t, view)

            preds = [[run(xk, view) for xk in states] for view in views]
        finally:
            if was_training:
                model.train()

        # L_twin: both views solve the same exact DDPM task on all K+1 states
        # fp32 for every reduction: the ordinary branch runs under bf16 autocast,
        # and ||Off(G_pert - G_clean)||_F^2 is a squared norm of small differences,
        # which bf16 cannot represent without losing most of the signal.
        twin_terms = [((preds[j][k].float() - eps[k]) ** 2).mean(dim=(1, 2, 3))
                      for j in range(2) for k in range(self.K + 1)]
        loss_twin = torch.stack(twin_terms, dim=0).mean(dim=0)

        # L_SVT: perturbed response geometry matched to the STOPPED clean one
        grams = []
        for j in range(2):
            base = (states[0] - sig_t * preds[j][0].float()) / a_t
            resp = [((states[k] - sig_t * preds[j][k].float()) / a_t - base).flatten(1)
                    for k in range(1, self.K + 1)]
            grams.append(response_gram(resp, self.tau_r))
        (g_clean, valid_c), (g_pert, valid_p) = grams
        valid = valid_c & valid_p
        svt_per_anchor = gram_discrepancy(g_pert, g_clean.detach())
        loss_svt = (svt_per_anchor * valid).sum() / valid.sum().clamp_min(1)

        stats = {
            "ipsvt_r_e": r_e,
            "ipsvt_valid_fraction": float(valid.float().mean()),
            "ipsvt_twin": float(loss_twin.mean().detach()),
            "ipsvt_svt": float(loss_svt.detach()),
        }
        return loss_twin.mean(), loss_svt, stats
