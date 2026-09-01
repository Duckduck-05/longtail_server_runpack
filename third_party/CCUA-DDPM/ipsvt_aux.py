"""The original IP-SVT auxiliary branch on the native CCUA U-Net.

This is intentionally a narrow port of the already evaluated IP-SVT
objective.  It keeps the exact-target Twin, the normalised response Gram
penalty, class-uniform auxiliary sampling, K=4 probes, and the global
embedding perturbation scale.  Only the host/model call has changed: the
branch now calls the CCUA-DDPM U-Net directly, so its checkpoint and optimizer
lineage are native CCUA rather than T2H/Coral.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


TAU_R_DEFAULT = 1e-6


def forward_with_rows(model, x: Tensor, t: Tensor, embedding_rows: Tensor) -> Tensor:
    """Run the CCUA U-Net with explicit class-conditioning rows.

    ``UNet.forward`` adds ``label_embedding(y)`` to the time embedding.  This
    mirrors that forward while replacing only the embedding lookup, which is
    needed for the perturbed condition ``c_y + eta``.
    """
    from model.model import ResBlock

    if getattr(model, "augm_embedding", None) is not None:
        raise RuntimeError("native IP-SVT does not support an augmentation embedding")

    temb = model.time_embedding(t) + embedding_rows
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
    h = model.tail(h)
    if skips:
        raise RuntimeError(f"native CCUA skip stack not consumed: {len(skips)} left")
    return h


def response_gram(responses: list[Tensor], tau_r: float) -> tuple[Tensor, Tensor]:
    stacked = torch.stack(responses, dim=1)
    norms = stacked.norm(dim=2)
    valid = (norms >= tau_r).all(dim=1)
    units = stacked / norms.clamp_min(tau_r).unsqueeze(2)
    return units @ units.transpose(1, 2), valid


def gram_discrepancy(gram_a: Tensor, gram_b: Tensor) -> Tensor:
    k = gram_a.shape[1]
    off = ~torch.eye(k, dtype=torch.bool, device=gram_a.device)
    return (gram_a - gram_b).mul(off).square().sum(dim=(1, 2)) / (k * (k - 1))


class IPSVTAuxiliary:
    """Class-uniform old IP-SVT auxiliary objective.

    The full mode is the paper/draft method that already has empirical
    evidence: both clean and perturbed conditions solve the exact DDPM target
    on K+1 forward-valid probes, and the perturbed response Gram matches the
    stopped clean response Gram.  ``twin`` and ``clean`` remain available as
    attribution controls.
    """

    MODES = ("full", "twin", "clean")

    def __init__(
        self,
        *,
        images,
        targets,
        num_class: int,
        T: int,
        beta_1: float,
        beta_T: float,
        K: int = 4,
        s: float = 0.05,
        delta: float = 0.1,
        batch_size: int = 16,
        lambda_svt: float = 1.0,
        lambda_aux: float = 1.0,
        every: int = 4,
        tau_r: float = TAU_R_DEFAULT,
        device: torch.device | None = None,
        seed: int = 0,
        use_checkpoint: bool = True,
        mode: str = "full",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"IP-SVT mode must be one of {self.MODES}; got {mode!r}")
        if K < 2:
            raise ValueError(f"old Gram-SVT requires K >= 2; got {K}")
        if not 0 < delta < 2 ** -0.5:
            raise ValueError(f"delta must satisfy 0 < delta < 1/sqrt(2); got {delta}")
        if batch_size <= 0 or every <= 0:
            raise ValueError("IP-SVT batch_size and every must be positive")

        raw = np.asarray(images)
        if raw.ndim != 4 or raw.shape[-1] not in (1, 3, 4):
            raise ValueError(f"native IP-SVT expects HWC image data, got {raw.shape}")
        if raw.shape[-1] == 1:
            raw = np.repeat(raw, 3, axis=-1)
        if raw.shape[-1] == 4:
            raw = raw[..., :3]

        self.images = torch.from_numpy(np.ascontiguousarray(raw))
        labels = np.asarray(targets, dtype=np.int64)
        if len(labels) != len(self.images):
            raise ValueError("IP-SVT images and targets have different lengths")
        self.class_index = [
            torch.as_tensor(np.flatnonzero(labels == c), dtype=torch.long)
            for c in range(num_class)
        ]
        empty = [c for c, indices in enumerate(self.class_index) if len(indices) == 0]
        if empty:
            raise ValueError(f"IP-SVT class-uniform sampling has empty classes: {empty}")

        self.num_class = int(num_class)
        self.T = int(T)
        self.K, self.s, self.delta = int(K), float(s), float(delta)
        self.batch_size = int(batch_size)
        self.lambda_svt, self.lambda_aux = float(lambda_svt), float(lambda_aux)
        self.every, self.tau_r = int(every), float(tau_r)
        self.use_checkpoint = bool(use_checkpoint)
        self.mode = mode
        self._r_e: float | None = None
        self._r_e_ema = 0.999
        self.cpu_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)

        betas = torch.linspace(beta_1, beta_T, self.T, dtype=torch.float64, device=self.device)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_alpha_bar = alpha_bar.sqrt().float()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt().float()

    def sample_batch(self) -> tuple[Tensor, Tensor]:
        """Sample a class-uniform batch, then apply CCUA's CIFAR transform."""
        y_cpu = torch.randint(
            self.num_class, (self.batch_size,), generator=self.cpu_generator, device="cpu"
        )
        picks = torch.empty(self.batch_size, dtype=torch.long)
        for i, class_id in enumerate(y_cpu.tolist()):
            pool = self.class_index[class_id]
            picks[i] = pool[torch.randint(len(pool), (1,), generator=self.cpu_generator)]

        x0 = self.images.index_select(0, picks).float()
        if float(x0.max()) > 1.5:
            x0 = x0 / 127.5 - 1.0
        elif float(x0.min()) >= 0.0:
            x0 = x0 * 2.0 - 1.0
        x0 = x0.permute(0, 3, 1, 2).contiguous()
        flip = torch.rand(self.batch_size, generator=self.cpu_generator) < 0.5
        x0[flip] = x0[flip].flip(-1)
        return x0.to(self.device, non_blocking=True), y_cpu.to(self.device, non_blocking=True)

    def embedding_scale(self, model) -> float:
        with torch.no_grad():
            current = float(
                model.label_embedding.weight[: self.num_class]
                .detach()
                .square()
                .sum(dim=1)
                .mean()
                .sqrt()
            )
        if self._r_e is None:
            self._r_e = current
        else:
            self._r_e = self._r_e_ema * self._r_e + (1.0 - self._r_e_ema) * current
        return self._r_e

    def __call__(self, model, step: int):
        if step % self.every != 0:
            return None

        model_device = next(model.parameters()).device
        if model_device != self.device:
            self.device = model_device
            self.noise_generator = torch.Generator(device=self.device).manual_seed(1)
            self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(self.device)
            self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(self.device)

        x0, y = self.sample_batch()
        n = len(y)
        r_e = self.embedding_scale(model)
        embedding = model.label_embedding.weight
        d = embedding.shape[1]
        t = torch.randint(self.T, (n,), generator=self.noise_generator, device=self.device)
        a_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sigma_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        eps = [torch.randn(x0.shape, generator=self.noise_generator, device=self.device)]
        for _ in range(self.K):
            xi = torch.randn(x0.shape, generator=self.noise_generator, device=self.device)
            eps.append((1.0 - self.delta**2) ** 0.5 * eps[0] + self.delta * xi)
        states = [a_t * x0 + sigma_t * noise for noise in eps]

        e_clean = embedding[y]
        eta = torch.randn(
            (n, d), generator=self.noise_generator, device=self.device
        ) * (self.s * r_e / d**0.5)
        views = [e_clean, e_clean + eta]

        was_training = model.training
        model.eval()
        try:
            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint

                def run(xk, view):
                    def forward_checkpointed(x_arg, view_arg):
                        previous = model.training
                        model.eval()
                        try:
                            return forward_with_rows(model, x_arg, t, view_arg)
                        finally:
                            if previous:
                                model.train()

                    return checkpoint(forward_checkpointed, xk, view, use_reentrant=False)
            else:
                def run(xk, view):
                    return forward_with_rows(model, xk, t, view)

            predictions = [[run(state, view) for state in states] for view in views]
        finally:
            if was_training:
                model.train()

        # Old Twin: both conditions regress every known probe to its exact noise.
        twin_terms = [
            (prediction.float() - eps[k]).square().mean(dim=(1, 2, 3))
            for prediction_set in predictions
            for k, prediction in enumerate(prediction_set)
        ]
        loss_twin = torch.stack(twin_terms, dim=0).mean(dim=0).mean()

        grams = []
        for prediction_set in predictions:
            anchor = (states[0] - sigma_t * prediction_set[0].float()) / a_t
            responses = [
                ((states[k] - sigma_t * prediction_set[k].float()) / a_t - anchor).flatten(1)
                for k in range(1, self.K + 1)
            ]
            grams.append(response_gram(responses, self.tau_r))
        (clean_gram, clean_valid), (perturbed_gram, perturbed_valid) = grams
        valid = clean_valid & perturbed_valid
        loss_svt = (
            gram_discrepancy(perturbed_gram, clean_gram.detach()) * valid
        ).sum() / valid.sum().clamp_min(1)
        if self.mode == "twin":
            loss_svt = loss_svt.detach() * 0.0
        elif self.mode == "clean":
            loss_twin = ((predictions[0][0].float() - eps[0]).square().mean(dim=(1, 2, 3))).mean()
            loss_svt = loss_svt.detach() * 0.0

        stats = {
            "ipsvt/r_e": r_e,
            "ipsvt/valid_fraction": float(valid.float().mean().detach()),
            "ipsvt/twin": float(loss_twin.detach()),
            "ipsvt/svt": float(loss_svt.detach()),
        }
        return loss_twin, loss_svt, stats

