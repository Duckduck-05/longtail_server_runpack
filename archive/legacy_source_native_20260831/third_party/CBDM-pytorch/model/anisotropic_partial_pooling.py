"""Gauge-controlled anisotropic partial pooling for diffusion noise prediction.

This module implements a class-only Gaussian random effect under a balanced DSM
pseudo-objective.  It is not an exact ELBO and its alternating covariance update
is MAP-style empirical Bayes rather than an exact EM procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model.model import UNet


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def _project_spd(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric.float())
    clipped = eigenvalues.clamp_min(floor)
    projected = (eigenvectors * clipped.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    return projected.to(dtype=matrix.dtype)


class APPLinearFactorUNet(nn.Module):
    """Identifiable class-only linear-in-z output factorization over one U-Net.

    This is not a generic z-conditioned U-Net with z injected throughout its
    residual blocks. The basis is whitened separately for every `(x_t, t)`
    prediction, so its remaining gauge is orthogonal and full posterior and
    population covariances transform coherently under that gauge.
    """

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
        posterior_structure: str = "full",
        population_structure: str = "full",
        covariance_floor: float = 1e-5,
        posterior_init_scale: float = 1e-2,
        basis_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if T <= 0 or num_class <= 0 or rank <= 0:
            raise ValueError("T, num_class, and rank must be positive")
        if posterior_structure not in {"full", "diagonal"}:
            raise ValueError("posterior_structure must be 'full' or 'diagonal'")
        if population_structure not in {"full", "isotropic"}:
            raise ValueError("population_structure must be 'full' or 'isotropic'")
        if covariance_floor <= 0 or posterior_init_scale <= covariance_floor or basis_eps <= 0:
            raise ValueError("invalid covariance floor, initial scale, or basis epsilon")

        self.T = int(T)
        self.num_class = int(num_class)
        self.rank = int(rank)
        self.posterior_structure = posterior_structure
        self.population_structure = population_structure
        self.covariance_floor = float(covariance_floor)
        self.basis_eps = float(basis_eps)

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
        self.global_head = nn.Conv2d(feature_channels, 3, kernel_size=3, padding=1)
        self.basis_head = nn.Conv2d(feature_channels, 3 * rank, kernel_size=3, padding=1)
        init.xavier_uniform_(self.global_head.weight, gain=1e-5)
        init.zeros_(self.global_head.bias)
        init.xavier_uniform_(self.basis_head.weight)
        init.zeros_(self.basis_head.bias)

        self.posterior_mean_raw = nn.Parameter(torch.zeros(num_class, rank))
        raw_tril = torch.zeros(num_class, rank, rank)
        diagonal = _inverse_softplus(torch.full((rank,), posterior_init_scale - covariance_floor))
        raw_tril[:, torch.arange(rank), torch.arange(rank)] = diagonal
        self.posterior_tril_raw = nn.Parameter(raw_tril)
        initial_covariance = torch.eye(rank) * posterior_init_scale**2
        self.register_buffer("population_covariance", initial_covariance)

    def centered_means(self) -> torch.Tensor:
        """Return class effects in the μ=0 global/local gauge."""
        return self.posterior_mean_raw - self.posterior_mean_raw.mean(dim=0, keepdim=True)

    def _tril_from_raw(self) -> torch.Tensor:
        raw_lower = torch.tril(self.posterior_tril_raw, diagonal=-1)
        raw_diagonal = torch.diagonal(self.posterior_tril_raw, dim1=-2, dim2=-1)
        diagonal = F.softplus(raw_diagonal) + self.covariance_floor
        return raw_lower + torch.diag_embed(diagonal)

    def posterior_covariance(self) -> torch.Tensor:
        """Return one SPD posterior covariance matrix for each class."""
        tril = self._tril_from_raw()
        if self.posterior_structure == "diagonal":
            diagonal = torch.diagonal(tril, dim1=-2, dim2=-1).square()
            return torch.diag_embed(diagonal)
        return tril @ tril.transpose(-1, -2)

    def _population_covariance(self) -> torch.Tensor:
        covariance = _project_spd(self.population_covariance, self.covariance_floor)
        if self.population_structure == "isotropic":
            variance = torch.diagonal(covariance).mean().clamp_min(self.covariance_floor)
            return torch.eye(self.rank, device=covariance.device, dtype=covariance.dtype) * variance
        return covariance

    def _posterior_cholesky(self) -> torch.Tensor:
        return torch.linalg.cholesky(self.posterior_covariance())

    def _basis(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        raw_basis = self.basis_head(features).view(batch, self.rank, 3, height, width)
        flat = raw_basis.flatten(start_dim=2).transpose(1, 2)
        gram = flat.transpose(1, 2).float() @ flat.float() / flat.shape[1]
        identity = torch.eye(self.rank, device=flat.device, dtype=gram.dtype).expand_as(gram)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram + self.basis_eps * identity)
        inverse_sqrt = (eigenvectors * eigenvalues.rsqrt().unsqueeze(-2)) @ eigenvectors.transpose(1, 2)
        whitened = flat.float() @ inverse_sqrt
        return whitened.transpose(1, 2).reshape(batch, self.rank, 3, height, width).to(raw_basis.dtype)

    def posterior(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.dtype != torch.long or y.ndim != 1:
            raise ValueError("y must be a rank-one LongTensor")
        if bool(((y < 0) | (y >= self.num_class)).any()):
            raise ValueError("class label is outside [0, num_class)")
        return self.centered_means()[y], self._posterior_cholesky()[y]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        augm: torch.Tensor | None = None,
        *,
        sample_posterior: bool | None = None,
        global_only: bool = False,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict noise using one shared-backbone forward for each input sample."""
        features = self.backbone(x, t, y=None, augm=augm)
        global_score = self.global_head(features)
        basis = self._basis(features)
        if y is None or global_only:
            local_score = torch.zeros_like(global_score)
            z = torch.zeros((x.shape[0], self.rank), dtype=x.dtype, device=x.device)
        else:
            mean, chol = self.posterior(y)
            if sample_posterior is None:
                sample_posterior = self.training
            if sample_posterior:
                z = mean + (chol @ torch.randn_like(mean).unsqueeze(-1)).squeeze(-1)
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

    def kl_per_class(self) -> torch.Tensor:
        """Return KL(q_c || N(0, Λ)) for every class in the centered gauge."""
        covariance = self.posterior_covariance()
        population = self._population_covariance()
        population_chol = torch.linalg.cholesky(population)
        solved_covariance = torch.cholesky_solve(covariance, population_chol)
        means = self.centered_means()
        solved_means = torch.cholesky_solve(means.unsqueeze(-1), population_chol).squeeze(-1)
        logdet_population = 2.0 * torch.log(torch.diagonal(population_chol)).sum()
        posterior_chol = self._posterior_cholesky()
        logdet_posterior = 2.0 * torch.log(
            torch.diagonal(posterior_chol, dim1=-2, dim2=-1)).sum(dim=-1)
        return 0.5 * (
            torch.diagonal(solved_covariance, dim1=-2, dim2=-1).sum(dim=-1)
            + (means * solved_means).sum(dim=-1)
            - self.rank
            + logdet_population
            - logdet_posterior
        )

    def frequency_weighted_kl(self, class_counts: torch.Tensor) -> torch.Tensor:
        """Return (1/C)∑_c (n̄/n_c) KL(q_c || p)."""
        counts = torch.as_tensor(
            class_counts, dtype=self.posterior_mean_raw.dtype, device=self.posterior_mean_raw.device)
        if counts.shape != (self.num_class,) or not bool((counts > 0).all()):
            raise ValueError("class_counts must contain one positive count per class")
        return ((counts.mean() / counts) * self.kl_per_class()).mean()

    @torch.no_grad()
    def set_posterior_(self, means: torch.Tensor, covariances: torch.Tensor) -> None:
        """Set posterior parameters from full covariance matrices for tests and initialization."""
        if means.shape != (self.num_class, self.rank):
            raise ValueError("means has an invalid shape")
        if covariances.shape != (self.num_class, self.rank, self.rank):
            raise ValueError("covariances has an invalid shape")
        projected = _project_spd(covariances.to(self.posterior_tril_raw), self.covariance_floor)
        chol = torch.linalg.cholesky(projected)
        diagonal = torch.diagonal(chol, dim1=-2, dim2=-1)
        raw_diagonal = _inverse_softplus((diagonal - self.covariance_floor).clamp_min(1e-12))
        self.posterior_mean_raw.copy_(means.to(self.posterior_mean_raw))
        self.posterior_tril_raw.copy_(torch.tril(chol, diagonal=-1))
        self.posterior_tril_raw.diagonal(dim1=-2, dim2=-1).copy_(raw_diagonal)

    @torch.no_grad()
    def map_update_population_(
        self,
        *,
        prior_strength: float,
        prior_scale2: float,
    ) -> torch.Tensor:
        """Perform a shrinkage-protected alternating-MAP population covariance update."""
        if prior_strength <= 0 or prior_scale2 <= 0:
            raise ValueError("prior_strength and prior_scale2 must be positive")
        means = self.centered_means()
        second_moment = (self.posterior_covariance() + means.unsqueeze(-1) * means.unsqueeze(-2)).sum(dim=0)
        identity = torch.eye(self.rank, device=second_moment.device, dtype=second_moment.dtype)
        updated = (second_moment + prior_strength * prior_scale2 * identity) / (
            self.num_class + prior_strength)
        updated = _project_spd(updated, self.covariance_floor)
        if self.population_structure == "isotropic":
            updated = torch.eye(self.rank, device=updated.device, dtype=updated.dtype) * torch.diagonal(updated).mean()
        self.population_covariance.copy_(updated)
        return updated.clone()

    @torch.no_grad()
    def rotate_orthogonal_gauge_(self, rotation: torch.Tensor) -> None:
        """Apply an equivalent orthogonal basis/latent-coordinate transform."""
        if rotation.shape != (self.rank, self.rank):
            raise ValueError("rotation has an invalid shape")
        identity = torch.eye(self.rank, device=rotation.device, dtype=rotation.dtype)
        if not torch.allclose(rotation.transpose(0, 1) @ rotation, identity, atol=1e-5, rtol=1e-5):
            raise ValueError("rotation must be orthogonal")
        rotation = rotation.to(device=self.posterior_mean_raw.device, dtype=self.posterior_mean_raw.dtype)
        self.posterior_mean_raw.copy_(self.posterior_mean_raw @ rotation)
        covariance = self.posterior_covariance()
        transformed_covariance = rotation.transpose(0, 1) @ covariance @ rotation
        self.set_posterior_(self.posterior_mean_raw, transformed_covariance)
        self.population_covariance.copy_(rotation.transpose(0, 1) @ self.population_covariance @ rotation)
        weight = self.basis_head.weight.view(self.rank, 3, *self.basis_head.weight.shape[1:])
        bias = self.basis_head.bias.view(self.rank, 3)
        self.basis_head.weight.copy_(
            (rotation.transpose(0, 1) @ weight.flatten(start_dim=1)).reshape_as(self.basis_head.weight))
        self.basis_head.bias.copy_((rotation.transpose(0, 1) @ bias).reshape_as(self.basis_head.bias))
