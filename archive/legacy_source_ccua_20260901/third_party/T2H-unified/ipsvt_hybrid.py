"""Natural-batch hybrid IP-SVT: new Twin plus legacy Gram/SVT transfer.

This module is deliberately separate from both ``ipsvt_aux`` (the legacy
class-uniform ``full``/``twin``/``clean`` experiment) and ``ipsvt_response``
(the K=1 response smoke).  It consumes the ordinary T2H loader minibatch and
creates no class index, resampler, tail gate, transfer target, or routing path.

For K >= 2 forward-valid probes,

    eps_k = sqrt(1 - delta**2) eps_0 + delta xi_k,

the auxiliary objective added by this module is

    lambda_aux * [L_twin + lambda_svt * L_SVT].

The host's ordinary k=0 DDPM loss remains the only noise-target supervision.
``L_twin = mean((f0(c_tilde) - sg(f0(c)))**2)``, and ``L_SVT`` is the legacy
predicted-clean, normalized response Gram/off-diagonal transfer matched to a
stopped clean Gram.  No probe other than the host anchor receives a DDPM
target.  In contrast to response mode, ``c_tilde = c_y + eta``: the embedding
row is intentionally *not* detached while the scalar noise scale is detached.

With K=4 the hook makes 2*(K+1)=10 logical U-Net forwards per natural batch:
five stopped true-condition teachers and five perturbed-condition probes.  The
host additionally makes its standard k=0 DDPM forward, for 11 full-batch
logical forwards total.  Only the five perturbed hook forwards are
gradient-bearing; the clean teachers are no-grad.  ``chunk_size`` limits
activation memory by splitting the hook batch.  For batch 64 and chunk 16,
that is one full-batch host call plus 40 hook calls in forward, with 20
checkpoint recomputations during backward.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor


ConditionedForward = Callable[[Any, Tensor, Tensor, Tensor], Tensor]


def response_gram(responses: list[Tensor], tau: float) -> tuple[Tensor, Tensor]:
    """Legacy normalized Gram and anchor-validity mask for K response vectors."""
    stacked = torch.stack(responses, dim=1)  # (N, K, D)
    norms = stacked.norm(dim=2)
    valid = (norms >= tau).all(dim=1)
    unit = stacked / norms.clamp_min(tau).unsqueeze(2)
    return unit @ unit.transpose(1, 2), valid


def off_diagonal(gram: Tensor) -> Tensor:
    k = gram.shape[1]
    mask = ~torch.eye(k, dtype=torch.bool, device=gram.device)
    return gram * mask


def gram_discrepancy(perturbed: Tensor, clean: Tensor) -> Tensor:
    """Legacy ||Off(G_pert - G_clean)||²_F / (K * (K - 1)) per anchor."""
    k = perturbed.shape[1]
    if k < 2:
        raise ValueError("Gram/SVT requires K >= 2")
    return off_diagonal(perturbed - clean).square().sum(dim=(1, 2)) / (k * (k - 1))


@contextmanager
def _eval_mode(model):
    """Disable dropout only for the auxiliary condition comparisons."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


class IPSVTHybridAuxiliary:
    """Chunked natural-batch hybrid objective used by ``ipsvt_mode=hybrid``."""

    def __init__(
        self,
        *,
        T: int,
        beta_1: float,
        beta_T: float,
        K: int = 4,
        s: float = 0.05,
        delta: float = 0.1,
        tau: float = 1e-6,
        lambda_aux: float = 1.0,
        lambda_svt: float = 1.0,
        chunk_size: int = 16,
        conditioned_forward: ConditionedForward,
        use_checkpoint: bool = True,
    ) -> None:
        if T <= 0:
            raise ValueError("T must be positive")
        if K < 2:
            raise ValueError("IP-SVT hybrid requires K >= 2 for Gram/SVT")
        if not 0.0 < delta < 2 ** -0.5:
            raise ValueError("ipsvt_delta must satisfy 0 < delta < 1/sqrt(2)")
        if s < 0 or tau <= 0 or lambda_aux < 0 or lambda_svt < 0:
            raise ValueError("hybrid s/lambda values must be non-negative and tau positive")
        if chunk_size <= 0:
            raise ValueError("ipsvt_hybrid_chunk must be positive")
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
        self.T = int(T)
        self.K = int(K)
        self.s = float(s)
        self.delta = float(delta)
        self.tau = float(tau)
        self.lambda_aux = float(lambda_aux)
        self.lambda_svt = float(lambda_svt)
        self.chunk_size = int(chunk_size)
        self.conditioned_forward = conditioned_forward
        self.use_checkpoint = bool(use_checkpoint)

    def _schedule(self, t: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        shape = (len(t),) + (1,) * (x.ndim - 1)
        alpha = self.sqrt_alpha_bar.to(device=x.device, dtype=x.dtype)[t].view(shape)
        sigma = self.sqrt_one_minus_alpha_bar.to(device=x.device, dtype=x.dtype)[t].view(shape)
        return alpha, sigma

    def _grad_forward(self, model, x: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        def run(x_view: Tensor, c_view: Tensor) -> Tensor:
            # checkpoint() runs this again during backward after the outer
            # context restored training mode, so set eval for both executions.
            with _eval_mode(model):
                return self.conditioned_forward(model, x_view, t, c_view)

        if not self.use_checkpoint:
            return run(x, condition)
        from torch.utils.checkpoint import checkpoint

        return checkpoint(run, x, condition, use_reentrant=False)

    @staticmethod
    def _per_example_mse(prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction.float() - target.float()).square().flatten(1).mean(dim=1)

    @staticmethod
    def _predicted_clean(x_t: Tensor, prediction: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
        return (x_t - sigma * prediction.float()) / alpha

    def __call__(self, context: Mapping[str, Any]):
        model = context["model"]
        x0, x_t0, t, eps0, y = (
            context[key] for key in ("x0", "x_t", "t", "noise", "y")
        )
        embedding = getattr(model, "label_embedding", None)
        if embedding is None:
            raise RuntimeError("IP-SVT hybrid requires a conditional label embedding")
        if y is None:
            raise RuntimeError("IP-SVT hybrid requires labels from the natural training batch")

        c_y = embedding(y)  # Spec: do not stop-gradient the embedding row.
        with torch.no_grad():
            # r_c^2 = mean_c ||c||_2^2 over the embedding table.  Only sigma is
            # stopped; c_y and c_tilde remain connected to label_embedding.
            r_c = embedding.weight.detach().square().sum(dim=1).mean().sqrt()
            condition_sigma = (self.s * r_c / (c_y.shape[-1] ** 0.5)).detach()
        c_tilde = c_y + torch.randn_like(c_y) * condition_sigma

        alpha, sigma = self._schedule(t, x0)
        states = [x_t0]
        for _ in range(self.K):
            xi = torch.randn_like(eps0)
            eps_k = (1.0 - self.delta ** 2) ** 0.5 * eps0 + self.delta * xi
            states.append(alpha * x0 + sigma * eps_k)

        twin_sum = x0.new_zeros(())
        svt_numerator = x0.new_zeros(())
        svt_denominator = x0.new_zeros(())
        total_examples = 0
        chunk_count = 0

        # The auxiliary comparisons use eval mode so eta is their only
        # condition-view difference.  Grad checkpoint closures reassert this.
        with _eval_mode(model):
            for start in range(0, len(y), self.chunk_size):
                end = min(start + self.chunk_size, len(y))
                sl = slice(start, end)
                chunk_count += 1
                n = end - start
                total_examples += n
                state_chunk = [state[sl] for state in states]
                alpha_chunk, sigma_chunk = alpha[sl], sigma[sl]
                t_chunk = t[sl]
                clean_condition, perturbed_condition = c_y[sl], c_tilde[sl]

                # The host already owns the sole exact DDPM target at k=0.
                # Every clean probe here is a stopped teacher: in particular,
                # k=1 must never create an auxiliary noise-target loss.
                with torch.no_grad():
                    clean_predictions = [
                        self.conditioned_forward(model, state_chunk[k], t_chunk, clean_condition)
                        for k in range(self.K + 1)
                    ]

                perturbed_predictions = [
                    self._grad_forward(model, state_chunk[k], t_chunk, perturbed_condition)
                    for k in range(self.K + 1)
                ]
                # New one-way Twin: no DDPM target on c_tilde.
                twin_sum = twin_sum + self._per_example_mse(
                    perturbed_predictions[0], clean_predictions[0].detach()
                ).sum()

                clean_x0 = [
                    self._predicted_clean(state_chunk[k], clean_predictions[k].detach(), alpha_chunk, sigma_chunk)
                    for k in range(self.K + 1)
                ]
                perturbed_x0 = [
                    self._predicted_clean(state_chunk[k], perturbed_predictions[k], alpha_chunk, sigma_chunk)
                    for k in range(self.K + 1)
                ]
                clean_responses = [
                    (clean_x0[k] - clean_x0[0]).flatten(1) for k in range(1, self.K + 1)
                ]
                perturbed_responses = [
                    (perturbed_x0[k] - perturbed_x0[0]).flatten(1) for k in range(1, self.K + 1)
                ]
                clean_gram, clean_valid = response_gram(clean_responses, self.tau)
                perturbed_gram, perturbed_valid = response_gram(perturbed_responses, self.tau)
                valid = clean_valid & perturbed_valid
                per_anchor_svt = gram_discrepancy(perturbed_gram, clean_gram.detach())
                svt_numerator = svt_numerator + (per_anchor_svt * valid).sum()
                svt_denominator = svt_denominator + valid.sum()

        twin = twin_sum / max(total_examples, 1)
        svt = svt_numerator / svt_denominator.clamp_min(1)
        raw = twin + self.lambda_svt * svt
        loss = self.lambda_aux * raw
        hook_logical_forwards = 2 * (self.K + 1)
        hook_gradient_forwards = self.K + 1
        return loss, {
            "twin": twin.detach(),
            "svt": svt.detach(),
            "raw": raw.detach(),
            "lambda_aux": x0.new_tensor(self.lambda_aux),
            "lambda_svt": x0.new_tensor(self.lambda_svt),
            "valid_fraction": (svt_denominator / max(total_examples, 1)).detach(),
            "hook_logical_forwards": x0.new_tensor(float(hook_logical_forwards)),
            "hook_gradient_forwards": x0.new_tensor(float(hook_gradient_forwards)),
            "total_logical_forwards": x0.new_tensor(float(1 + hook_logical_forwards)),
            "total_gradient_forwards": x0.new_tensor(float(1 + hook_gradient_forwards)),
            "chunk_count": x0.new_tensor(float(chunk_count)),
        }


__all__ = ["IPSVTHybridAuxiliary", "gram_discrepancy", "off_diagonal", "response_gram"]
