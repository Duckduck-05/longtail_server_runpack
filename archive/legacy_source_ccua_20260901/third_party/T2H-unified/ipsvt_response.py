"""One-way IP-SVT response regularizer for the shared T2H host.

This is intentionally independent of :mod:`ipsvt_aux`, which implements the
legacy class-uniform ``full``/``twin``/``clean`` experiment.  The response
variant consumes the ordinary training minibatch directly; it never builds a
class index or draws an auxiliary batch.

For one response direction (``K=1``), with two independently noised views of
the same natural example, it implements

    Twin = mean((f0(c_tilde) - sg(f0(c)))**2)
    SVT  = mean(((f1(c_tilde) - f0(c_tilde))
                 - sg(f1(c) - f0(c)))**2)

where ``c_tilde = sg(c_y) + eta``.  The raw noise-space MSE deliberately
preserves response magnitude and direction: there is no normalization, Gram
matrix, off-diagonal projection, threshold, or second loss weight.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


ConditionedForward = Callable[[Any, Tensor, Tensor, Tensor], Tensor]


def forward_with_condition(model, x: Tensor, t: Tensor, condition: Tensor) -> Tensor:
    """Run T2H's plain U-Net with an explicit condition embedding.

    Response mode intentionally uses the plain shared T2H path.  CM LoRA
    branches and augmentation embeddings have separate conditioning semantics,
    so accepting them here would silently implement a different objective.
    """
    from model.model_cm import ResBlock

    if getattr(model, "augm_embedding", None) is not None:
        raise RuntimeError("IP-SVT response mode does not support augmentation conditioning")
    if tuple(getattr(model, "lora_part", ())) != ():
        raise RuntimeError("IP-SVT response mode requires the plain T2H U-Net (no CM LoRA parts)")

    temb = model.time_embedding(t) + condition
    h = model.head(x)
    skips = [h]
    for layer in model.downblocks:
        h = layer(h, temb)
        skips.append(h)
    for layer in model.middleblocks:
        h = layer(h, temb)
    for layer in model.upblocks:
        if isinstance(layer, ResBlock):
            h = torch.cat([h, skips.pop()], dim=1)
        h = layer(h, temb)
    if skips:
        raise RuntimeError(f"T2H skip stack was not consumed ({len(skips)} tensors remain)")
    return model.tail(h)


@contextmanager
def _eval_mode(model):
    """Make the four condition comparisons deterministic without state drift."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


class IPSVTResponseAuxiliary:
    """Natural-batch, one-way Twin / response-transfer objective.

    ``variant='twin'`` is the corrected Twin-only ablation.  ``variant='full'``
    adds the single K=1 response difference term.  ``lambda_weight`` is the
    only loss coefficient and scales their sum in the caller.
    """

    VARIANTS = frozenset({"twin", "full"})

    def __init__(
        self,
        *,
        T: int,
        beta_1: float,
        beta_T: float,
        eta_std: float,
        lambda_weight: float,
        variant: str,
        conditioned_forward: ConditionedForward = forward_with_condition,
        use_checkpoint: bool = True,
    ) -> None:
        if T <= 0:
            raise ValueError("T must be positive")
        if eta_std < 0:
            raise ValueError("ipsvt_response_eta must be non-negative")
        if lambda_weight < 0:
            raise ValueError("ipsvt_lambda must be non-negative")
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}; got {variant!r}")
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
        self.T = int(T)
        self.eta_std = float(eta_std)
        self.lambda_weight = float(lambda_weight)
        self.variant = variant
        self.conditioned_forward = conditioned_forward
        self.use_checkpoint = bool(use_checkpoint)

    def _schedule(self, t: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        shape = (len(t),) + (1,) * (x.ndim - 1)
        alpha = self.sqrt_alpha_bar.to(device=x.device, dtype=x.dtype)[t].view(shape)
        sigma = self.sqrt_one_minus_alpha_bar.to(device=x.device, dtype=x.dtype)[t].view(shape)
        return alpha, sigma

    def _perturbed_forward(self, model, x: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        def run(x_view: Tensor, c_view: Tensor) -> Tensor:
            # Non-reentrant checkpointing recomputes this closure in backward.
            # Reassert eval mode there so dropout cannot create a second source
            # of disagreement between c and c_tilde.
            with _eval_mode(model):
                return self.conditioned_forward(model, x_view, t, c_view)

        if not self.use_checkpoint:
            return run(x, condition)
        from torch.utils.checkpoint import checkpoint

        return checkpoint(run, x, condition, use_reentrant=False)

    def __call__(self, context: Mapping[str, Any]):
        model = context["model"]
        x0, x_t0, t, noise0, y = (
            context[key] for key in ("x0", "x_t", "t", "noise", "y")
        )
        if getattr(model, "label_embedding", None) is None:
            raise RuntimeError("IP-SVT response mode requires a conditional label embedding")
        if y is None:
            raise RuntimeError("IP-SVT response mode requires labels from the natural training batch")

        c_y = model.label_embedding(y)
        # Interpret eta_std as the dimensionless relative radius s from the
        # protocol, not as an absolute per-coordinate standard deviation.
        # This is sigma_c = s * r_c / sqrt(d_c), where r_c^2 is the mean
        # squared norm over all class embeddings.  The radius is stopped so
        # the auxiliary loss cannot change its own perturbation scale.
        with torch.no_grad():
            all_embeddings = model.label_embedding.weight.float()
            r_c = all_embeddings.square().sum(dim=1).mean().sqrt()
            sigma_c = (r_c * self.eta_std / (c_y.shape[-1] ** 0.5)).to(c_y.dtype)
        eta = torch.randn_like(c_y) * sigma_c
        # This is deliberately one-way: neither the clean target nor the
        # perturbed base may train label_embedding through this auxiliary loss.
        c_tilde = c_y.detach() + eta

        alpha, sigma = self._schedule(t, x0)
        if self.variant == "full":
            noise1 = torch.randn_like(noise0)
            x_t1 = alpha * x0 + sigma * noise1
        else:
            x_t1 = None

        # The stopped c branch is a teacher.  no_grad is stronger than a final
        # detach: it also prevents unused intermediate activations from being
        # retained beside the two student graphs.
        with _eval_mode(model), torch.no_grad():
            f0_clean = self.conditioned_forward(model, x_t0, t, c_y)
            f1_clean = (
                self.conditioned_forward(model, x_t1, t, c_y)
                if x_t1 is not None else None
            )

        f0_perturbed = self._perturbed_forward(model, x_t0, t, c_tilde)
        twin = F.mse_loss(f0_perturbed.float(), f0_clean.detach().float())

        if x_t1 is None:
            svt = twin.new_zeros(())
        else:
            f1_perturbed = self._perturbed_forward(model, x_t1, t, c_tilde)
            response_perturbed = f1_perturbed.float() - f0_perturbed.float()
            response_clean = f1_clean.detach().float() - f0_clean.detach().float()
            # Raw MSE is intentional: no normalisation/Gram/off-diagonal/tau.
            svt = F.mse_loss(response_perturbed, response_clean)

        raw = twin + svt
        total = raw * self.lambda_weight
        return total, {
            "twin": twin.detach(),
            "svt": svt.detach(),
            "raw": raw.detach(),
            "lambda": twin.new_tensor(self.lambda_weight),
            "embedding_radius": r_c.detach().to(twin.dtype),
            "eta_sigma": sigma_c.detach().to(twin.dtype),
            "variant_full": twin.new_tensor(float(self.variant == "full")),
        }


__all__ = ["IPSVTResponseAuxiliary", "forward_with_condition"]
