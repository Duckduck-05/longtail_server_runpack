"""Stochastic partial-pooling score model for long-tailed diffusion.

The conditional noise predictor is parameterized as

    eps_c(x_t, t) = eps_g(x_t, t) + B(x_t, t) z_{c,k(t)},

where the backbone is label-free, the columns of ``B`` are shared RMS-normalized
spatial basis maps, and ``q(z_{c,k})`` is a diagonal Gaussian posterior.  During
training ``z`` is reparameterized; evaluation uses its posterior mean.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import init

from model.model import UNet


class StochasticPartialPoolingUNet(nn.Module):
    """Label-free UNet plus stochastic class/time random effects."""

    def __init__(
        self,
        *,
        T: int,
        ch: int,
        ch_mult: list[int],
        attn: list[int],
        num_res_blocks: int,
        dropout: float,
        augm: bool,
        num_class: int,
        rank: int = 16,
        time_bins: int = 10,
        posterior_init_std: float = 1e-2,
        basis_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if T <= 0 or num_class <= 0 or rank <= 0 or time_bins <= 0:
            raise ValueError("T, num_class, rank, and time_bins must be positive")
        if posterior_init_std <= 0 or basis_eps <= 0:
            raise ValueError("posterior_init_std and basis_eps must be positive")

        self.T = int(T)
        self.num_class = int(num_class)
        self.rank = int(rank)
        self.time_bins = int(time_bins)
        self.basis_eps = float(basis_eps)

        # ``cond=False`` is the architectural contract: no label enters the
        # shared backbone or the global score branch.
        self.backbone = UNet(
            T=T,
            ch=ch,
            ch_mult=ch_mult,
            attn=attn,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            cond=False,
            augm=augm,
            num_class=num_class,
        )
        feature_channels = self.backbone.tail[-1].in_channels
        self.backbone.tail[-1] = nn.Identity()
        self.global_head = nn.Conv2d(feature_channels, 3, 3, stride=1, padding=1)
        self.basis_head = nn.Conv2d(
            feature_channels, 3 * self.rank, 3, stride=1, padding=1)
        init.xavier_uniform_(self.global_head.weight, gain=1e-5)
        init.zeros_(self.global_head.bias)
        init.xavier_uniform_(self.basis_head.weight)
        init.zeros_(self.basis_head.bias)

        posterior_shape = (self.num_class, self.time_bins, self.rank)
        self.posterior_mean = nn.Parameter(torch.zeros(posterior_shape))
        initial_var = posterior_init_std**2
        self.posterior_logvar = nn.Parameter(
            torch.full(posterior_shape, math.log(initial_var)))
        self.register_buffer("prior_tau2", torch.full((self.time_bins,), initial_var))

    def time_bin(self, t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.long:
            raise TypeError("t must be a LongTensor")
        if bool(((t < 0) | (t >= self.T)).any()):
            raise ValueError("timestep is outside [0, T)")
        return torch.div(t * self.time_bins, self.T, rounding_mode="floor").clamp_max(
            self.time_bins - 1)

    def _basis(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        basis = self.basis_head(features).view(batch, self.rank, 3, height, width)
        rms = basis.float().square().mean(dim=(2, 3, 4), keepdim=True).add(
            self.basis_eps).sqrt()
        return basis / rms.to(dtype=basis.dtype)

    def posterior(self, y: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.dtype != torch.long or y.ndim != 1 or y.shape != t.shape:
            raise ValueError("y and t must be same-shaped rank-one LongTensors")
        if bool(((y < 0) | (y >= self.num_class)).any()):
            raise ValueError("class label is outside [0, num_class)")
        bins = self.time_bin(t)
        mean = self.posterior_mean[y, bins]
        logvar = self.posterior_logvar[y, bins].clamp(min=-20.0, max=10.0)
        return mean, logvar

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        augm: torch.Tensor | None = None,
        *,
        sample_posterior: bool | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.backbone(x, t, y=None, augm=augm)
        global_score = self.global_head(features)
        basis = self._basis(features)

        if y is None:
            prediction = global_score
            local_score = torch.zeros_like(global_score)
            z = torch.zeros((x.shape[0], self.rank), device=x.device, dtype=x.dtype)
        else:
            mean, logvar = self.posterior(y, t)
            if sample_posterior is None:
                sample_posterior = self.training
            if sample_posterior:
                z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
            else:
                z = mean
            local_score = torch.einsum("br,brchw->bchw", z.to(basis.dtype), basis)
            prediction = global_score + local_score

        if not return_components:
            return prediction
        return prediction, {
            "global_score": global_score,
            "local_score": local_score,
            "basis": basis,
            "z": z,
        }

    def kl_per_class_bin(self) -> torch.Tensor:
        """KL(q(z_ck) || N(0, tau_k^2 I)), shape ``[C, K]``."""
        logvar = self.posterior_logvar.clamp(min=-20.0, max=10.0)
        variance = torch.exp(logvar)
        tau2 = self.prior_tau2.clamp_min(1e-12)[None, :, None]
        return 0.5 * (
            torch.log(tau2) - logvar + (variance + self.posterior_mean.square()) / tau2 - 1.0
        ).sum(dim=-1)

    def partial_pooling_regularizer(self, class_counts: torch.Tensor) -> torch.Tensor:
        """Frequency-derived ``(1/CK) sum_ck (n_bar/n_c) KL(q_ck || p_k)``."""
        counts = torch.as_tensor(
            class_counts, device=self.posterior_mean.device, dtype=self.posterior_mean.dtype)
        if counts.shape != (self.num_class,) or not bool((counts > 0).all()):
            raise ValueError("class_counts must contain one positive count per class")
        class_weight = counts.mean() / counts
        return (class_weight[:, None] * self.kl_per_class_bin()).mean()

    @torch.no_grad()
    def empirical_bayes_update(self, *, min_tau2: float = 1e-8) -> torch.Tensor:
        """Balanced M-step for the time-bin prior variance.

        The update averages ``m_ckj^2 + V_ckj`` uniformly over classes and
        basis coordinates, so it is independent of the long-tail sampling law.
        ``min_tau2`` is only a numerical variance floor.
        """
        if min_tau2 <= 0:
            raise ValueError("min_tau2 must be positive")
        second_moment = self.posterior_mean.square() + torch.exp(
            self.posterior_logvar.clamp(min=-20.0, max=10.0))
        estimate = second_moment.mean(dim=(0, 2)).clamp_min(min_tau2)
        self.prior_tau2.copy_(estimate)
        return estimate.clone()

    @torch.no_grad()
    def posterior_summary(self, class_counts: torch.Tensor) -> dict[str, float | list[float]]:
        counts = torch.as_tensor(class_counts, device=self.posterior_mean.device)
        order = torch.argsort(counts, descending=True)
        third = max(1, self.num_class // 3)
        local_energy = self.posterior_mean.square().mean(dim=(1, 2))
        variance = torch.exp(self.posterior_logvar.clamp(min=-20.0, max=10.0))
        return {
            "tau2": self.prior_tau2.detach().float().cpu().tolist(),
            "posterior_variance_mean": float(variance.mean().cpu()),
            "mean_energy_head": float(local_energy[order[:third]].mean().cpu()),
            "mean_energy_medium": float(local_energy[order[third:-third]].mean().cpu())
            if self.num_class > 2 * third else float("nan"),
            "mean_energy_tail": float(local_energy[order[-third:]].mean().cpu()),
        }
