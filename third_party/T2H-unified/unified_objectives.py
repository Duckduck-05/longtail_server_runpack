"""Standalone long-tail diffusion objectives for the shared T2H/OC_LT host.

The module deliberately owns no training loop or model architecture.  Its sole
contract with a U-Net is::

    model(x_t, t, y=..., augm=..., return_mid=..., use_cm=...)

``UnifiedObjective`` returns ``(loss, diagnostics)`` where ``loss`` is a
scalar tensor.  It therefore can be used by a native trainer without changing
its optimizer, data loader, or checkpoint protocol.

The implementation mirrors the vendored sources where the objective is
available: OC_LT's transferred DSM target, CBDM's stopped two-sided balanced
prediction penalty, CORAL's timestep-scaled supervised contrastive penalty,
and CCUA's unconditional contrastive/alignment penalties.  CM and IP-SVT are
explicit protocol hooks because their released implementations respectively
need a CM-capable U-Net and a class-uniform auxiliary data/embedding path.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


METHODS = frozenset({"ddpm", "t2h", "cbdm", "coral", "ccua", "ipsvt", "cm"})


@dataclass(frozen=True)
class ObjectiveResult:
    """Named alternative for callers that prefer an object over tuple unpacking."""

    loss: Tensor
    diagnostics: Mapping[str, Tensor]


def _extract(values: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
    """Gather a schedule at ``t`` and reshape it for arbitrary image ranks."""
    return values.gather(0, t).to(dtype=torch.float32).view(
        (t.shape[0],) + (1,) * (len(shape) - 1)
    )


def _as_scalar(value: Any, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _prediction_and_feature(output: Any) -> tuple[Tensor, Tensor | None]:
    """Accept the source repos' tensor, ``(eps, mid)``, and CORAL tuple forms."""
    if isinstance(output, Tensor):
        return output, None
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        feature = output[1] if len(output) > 1 and isinstance(output[1], Tensor) else None
        return output[0], feature
    raise TypeError("model must return an epsilon Tensor or a tuple whose first item is epsilon")


def _per_example_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction.float() - target.float()).square().flatten(1).mean(dim=1)


def _unconditional_info_nce(features: Tensor, temperature: float) -> Tensor:
    """The CCUA ``uncond_info_nce(f, f)`` calculation without third-party deps.

    CCUA flattens each latent, treats the same sample as its positive key, and
    every other sample as a negative paired key.  For identical query/key
    tensors this is exactly cross entropy over their normalized Gram matrix.
    """
    if features.shape[0] < 2:
        return features.new_zeros((features.shape[0],), dtype=torch.float32)
    vectors = F.normalize(features.float().flatten(1), dim=1)
    # CCUA's released ``info_nce`` receives the same latent tensor as both
    # query and key.  Keep gradients through both sides; detaching the key
    # changes the objective to a one-sided surrogate and makes this port
    # systematically weaker/different from the native loss.
    logits = vectors @ vectors.T / temperature
    return F.cross_entropy(logits, torch.arange(len(vectors), device=features.device), reduction="none")


def _supervised_contrastive(features: Tensor, labels: Tensor, temperature: float) -> Tensor:
    """SupCon loss used by CORAL, returning one scalar.

    Anchors without another member of their class are excluded, matching the
    conventional supervised-contrastive reduction and avoiding a fake positive
    from the anchor itself.
    """
    if features.shape[0] < 2:
        return features.new_zeros((), dtype=torch.float32)
    vectors = F.normalize(features.float().flatten(1), dim=1)
    logits = vectors @ vectors.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    non_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & non_self
    log_prob = logits - torch.logsumexp(logits.masked_fill(~non_self, float("-inf")), dim=1, keepdim=True)
    positive_count = positives.sum(dim=1)
    per_anchor = -(log_prob * positives).sum(dim=1) / positive_count.clamp_min(1)
    usable = positive_count > 0
    return per_anchor[usable].mean() if usable.any() else per_anchor.new_zeros(())


class UnifiedObjective(nn.Module):
    """Dispatch a faithful, self-contained loss for one long-tail method.

    ``class_probs`` is the source repositories' empirical class-frequency
    vector: CBDM samples balanced labels from it and T2H uses it to retain only
    source-compatible transfer directions.  It is normalized defensively.

    Hooks receive a context dictionary and may return either a loss tensor or
    ``(loss, diagnostics)``.  ``transfer_target_hook`` replaces only T2H's
    target construction, so it can implement an experimental transfer protocol
    while preserving the common corruption/model call path.
    """

    def __init__(
        self,
        method: str,
        *,
        T: int = 1000,
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,
        num_classes: int | None = None,
        class_probs: Tensor | None = None,
        cfg_dropout: float = 0.0,
        cbdm_tau: float = 1.0,
        t2h_mode: str = "t2h",
        t2h_cut_time: int = -1,
        transfer_target_hook: Callable[[Mapping[str, Any]], Any] | None = None,
        coral_weight: float = 0.5,
        coral_temperature: float = 0.1,
        coral_time_scale: float = 0.3,
        ccua_ucl_weight: float = 0.1,
        ccua_alignment_weight: float = 0.1,
        ccua_temperature: float = 0.1,
        ipsvt_hook: Callable[[Mapping[str, Any]], Any] | None = None,
        cm_hook: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        super().__init__()
        method = method.lower()
        if method not in METHODS:
            raise ValueError(f"unknown objective {method!r}; expected one of {sorted(METHODS)}")
        if T <= 0:
            raise ValueError("T must be positive")
        if not 0.0 <= cfg_dropout <= 1.0:
            raise ValueError("cfg_dropout must be in [0, 1]")
        if t2h_mode not in {"t2h", "h2t", "full"}:
            raise ValueError("t2h_mode must be one of 't2h', 'h2t', or 'full'")
        if coral_temperature <= 0 or coral_time_scale <= 0 or ccua_temperature <= 0:
            raise ValueError("contrastive temperatures/scales must be positive")
        if class_probs is not None:
            probs = torch.as_tensor(class_probs, dtype=torch.float64).flatten()
            if probs.numel() == 0 or not torch.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
                raise ValueError("class_probs must be a non-empty, finite, non-negative vector with positive sum")
            probs = probs / probs.sum()
            if num_classes is not None and probs.numel() != num_classes:
                raise ValueError("class_probs length must equal num_classes")
            num_classes = probs.numel()
        elif num_classes is not None:
            if num_classes <= 0:
                raise ValueError("num_classes must be positive")
            probs = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float64)
        else:
            probs = torch.empty(0, dtype=torch.float64)

        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float64)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.method, self.T, self.num_classes = method, T, num_classes
        self.cfg_dropout, self.cbdm_tau = cfg_dropout, cbdm_tau
        self.t2h_mode, self.t2h_cut_time = t2h_mode, t2h_cut_time
        self.transfer_target_hook = transfer_target_hook
        self.coral_weight, self.coral_temperature = coral_weight, coral_temperature
        self.coral_time_scale = coral_time_scale
        self.ccua_ucl_weight, self.ccua_alignment_weight = ccua_ucl_weight, ccua_alignment_weight
        self.ccua_temperature = ccua_temperature
        self.ipsvt_hook, self.cm_hook = ipsvt_hook, cm_hook
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.register_buffer("sigma_squared", 1.0 / alpha_bar - 1.0)
        self.register_buffer("class_probs", probs)

    def _call_model(self, model: Callable[..., Any], x_t: Tensor, t: Tensor, *, y: Tensor | None,
                    augm: Any, return_mid: bool, use_cm: bool) -> tuple[Tensor, Tensor | None]:
        return _prediction_and_feature(model(x_t, t, y=y, augm=augm, return_mid=return_mid, use_cm=use_cm))

    def _model_labels(self, y: Tensor) -> Tensor | None:
        if self.cfg_dropout and torch.rand((), device=y.device) < self.cfg_dropout:
            return None
        return y

    def _t2h_target(self, x0: Tensor, x_t: Tensor, t: Tensor, noise: Tensor, y: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if self.class_probs.numel() == 0:
            raise ValueError("t2h needs num_classes or empirical class_probs")
        probs = self.class_probs.to(device=y.device, dtype=torch.float32)
        if y.min() < 0 or y.max() >= probs.numel():
            raise ValueError("labels are outside class_probs")
        sigma_sq = _extract(self.sigma_squared, t, x0.shape)
        # OC_LT evaluates distance from c(x_t)=x0+sigma_t*epsilon to every x0.
        cxt = x0 + sigma_sq.sqrt() * noise
        squared_distance = (cxt.flatten(1)[:, None] - x0.flatten(1)[None, :]).square().sum(dim=-1)
        logits = -squared_distance / (2.0 * sigma_sq.flatten())[:, None]
        weights = torch.softmax(logits, dim=1)
        sampled = torch.multinomial(weights, 1).squeeze(1)
        identity = torch.arange(x0.shape[0], device=x0.device)
        old_prob, new_prob = probs[y], probs[y[sampled]]
        allowed = torch.ones_like(y, dtype=torch.bool)
        if self.t2h_mode == "t2h":
            allowed = new_prob >= old_prob
        elif self.t2h_mode == "h2t":
            allowed = new_prob <= old_prob
        if self.t2h_cut_time >= 0:
            allowed &= t < self.t2h_cut_time
        selected = torch.where(allowed, sampled, identity)
        target = (x_t - _extract(self.sqrt_alpha_bar, t, x0.shape) * x0[selected]) / _extract(
            self.sqrt_one_minus_alpha_bar, t, x0.shape
        )
        return target, {
            "t2h_transfer_fraction": selected.ne(identity).float().mean(),
            "t2h_mean_similarity": weights.gather(1, sampled[:, None]).mean(),
        }

    def _hook_loss(self, hook: Callable[[Mapping[str, Any]], Any], context: Mapping[str, Any], *,
                   name: str, like: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        result = hook(context)
        diagnostics: Mapping[str, Any] = {}
        if isinstance(result, tuple):
            if len(result) != 2:
                raise TypeError(f"{name}_hook must return loss or (loss, diagnostics)")
            result, diagnostics = result
        if not isinstance(diagnostics, Mapping):
            raise TypeError(f"{name}_hook diagnostics must be a mapping")
        loss = _as_scalar(result, device=like.device, dtype=like.dtype)
        if loss.ndim != 0:
            loss = loss.mean()
        converted = {f"{name}_{key}": _as_scalar(value, device=like.device, dtype=like.dtype)
                     for key, value in diagnostics.items()}
        return loss, converted

    def forward(
        self,
        model: Callable[..., Any],
        x0: Tensor,
        y: Tensor,
        augm: Any = None,
        *,
        t: Tensor | None = None,
        noise: Tensor | None = None,
        step: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return a scalar loss and tensor diagnostics for one noisy minibatch."""
        if x0.ndim < 2 or y.ndim != 1 or x0.shape[0] != y.shape[0]:
            raise ValueError("x0 must be batched and y must be a matching rank-1 label tensor")
        if self.method == "cm" and self.cm_hook is None:
            raise NotImplementedError(
                "cm requires cm_hook: the shared objective cannot invent the CM model capacity/protocol"
            )
        if self.method == "ipsvt" and self.ipsvt_hook is None:
            raise NotImplementedError(
                "ipsvt requires ipsvt_hook: it needs class-uniform auxiliary data and condition embeddings"
            )
        if t is None:
            t = torch.randint(self.T, (x0.shape[0],), device=x0.device)
        if t.shape != y.shape or t.dtype not in (torch.int32, torch.int64):
            raise ValueError("t must be an integer tensor with shape y.shape")
        if (t < 0).any() or (t >= self.T).any():
            raise ValueError("t contains a timestep outside [0, T)")
        t = t.to(dtype=torch.long)
        noise = torch.randn_like(x0) if noise is None else noise
        if noise.shape != x0.shape:
            raise ValueError("noise must have x0.shape")
        x_t = _extract(self.sqrt_alpha_bar, t, x0.shape) * x0 + _extract(
            self.sqrt_one_minus_alpha_bar, t, x0.shape
        ) * noise
        # CM is intentionally a protocol boundary.  Unlike every other arm it
        # cannot be faithfully expressed against the ordinary epsilon U-Net:
        # the released implementation needs a model with ``use_cm`` capacity.
        # A supplied hook owns that model-specific forward and may use this
        # common corruption context; do not silently run a DDPM surrogate.
        if self.method == "cm":
            assert self.cm_hook is not None  # guarded above; narrows Optional for type checkers
            loss, diagnostics = self._hook_loss(
                self.cm_hook,
                {"model": model, "x0": x0, "x_t": x_t, "t": t, "noise": noise, "y": y,
                 "augm": augm, "step": step},
                name="cm",
                like=x0,
            )
            diagnostics["cm_aux"] = loss
            diagnostics["loss"] = loss
            return loss, diagnostics
        y_model = self._model_labels(y)
        wants_feature = self.method in {"coral", "ccua"}
        prediction, feature = self._call_model(
            model, x_t, t, y=y_model, augm=augm, return_mid=wants_feature, use_cm=self.method == "cm"
        )

        target, diagnostics = noise, {"dsm": _per_example_mse(prediction, noise).mean()}
        if self.method == "t2h":
            context = {"model": model, "x0": x0, "x_t": x_t, "t": t, "noise": noise, "y": y, "augm": augm}
            if self.transfer_target_hook is None:
                target, transfer_diag = self._t2h_target(x0, x_t, t, noise, y)
            else:
                hooked = self.transfer_target_hook(context)
                if isinstance(hooked, tuple):
                    target, transfer_diag = hooked
                else:
                    target, transfer_diag = hooked, {}
                if not isinstance(target, Tensor) or target.shape != x0.shape:
                    raise TypeError("transfer_target_hook must return a target tensor with x0.shape")
                if not isinstance(transfer_diag, Mapping):
                    raise TypeError("transfer_target_hook diagnostics must be a mapping")
                transfer_diag = {f"t2h_{key}": _as_scalar(value, device=x0.device, dtype=x0.dtype)
                                 for key, value in transfer_diag.items()}
            diagnostics.update(transfer_diag)

        dsm = _per_example_mse(prediction, target).mean()
        diagnostics["dsm"] = dsm
        loss = dsm

        if self.method == "cbdm":
            if y_model is None:
                balance = dsm.new_zeros(())
                diagnostics["cbdm_skipped_unconditional"] = dsm.new_ones(())
            else:
                if self.class_probs.numel() == 0:
                    raise ValueError("cbdm needs num_classes or empirical class_probs")
                balanced_y = torch.multinomial(self.class_probs.to(x0.device, dtype=torch.float32), len(y), replacement=True)
                balanced_prediction, _ = self._call_model(
                    model, x_t, t, y=balanced_y, augm=augm, return_mid=False, use_cm=False
                )
                time_weight = t.to(dtype=x0.dtype).view((len(t),) + (1,) * (x0.ndim - 1)) / self.T * self.cbdm_tau
                reg = time_weight * (prediction.float() - balanced_prediction.detach().float()).square()
                com = time_weight * (prediction.detach().float() - balanced_prediction.float()).square()
                balance = reg.mean() + 0.25 * com.mean()
            loss = loss + balance
            diagnostics["cbdm"] = balance

        elif self.method == "coral":
            if y_model is None or feature is None:
                coral = dsm.new_zeros(())
                diagnostics["coral_feature_available"] = dsm.new_zeros(())
            else:
                time_scale = torch.exp((1.0 - t.float() / self.T) / self.coral_time_scale).mean()
                # The released CORAL trainer applies one additional
                # ``supcon_temp`` multiplier when it combines the scalar
                # SupCon term into the non-AMP loss (main.py:530).  Keep that
                # quirk in the port: omitting it makes the common CORAL arm
                # 1 / temperature stronger than the native objective.
                supcon = _supervised_contrastive(feature, y, self.coral_temperature)
                coral = (self.coral_weight * time_scale
                          * self.coral_temperature * supcon)
                diagnostics["coral_feature_available"] = dsm.new_ones(())
            loss = loss + coral
            diagnostics["coral_supcon"] = coral

        elif self.method == "ccua":
            # The released CCUA trainer asks for a second unconditional forward
            # even if the ordinary CFG branch happened to be unconditional.
            uncond_prediction, uncond_feature = self._call_model(
                model, x_t, t, y=None, augm=augm, return_mid=True, use_cm=False
            )
            time_weight = t.float() / self.T
            if uncond_feature is None or self.ccua_ucl_weight == 0:
                ucl = dsm.new_zeros(())
            else:
                ucl = (time_weight * _unconditional_info_nce(uncond_feature, self.ccua_temperature)).mean()
            if y_model is None or self.ccua_alignment_weight == 0:
                alignment = dsm.new_zeros(())
            else:
                alignment = (
                    time_weight.view((len(t),) + (1,) * (x0.ndim - 1))
                    * (prediction.float() - uncond_prediction.float()).square()
                ).mean()
            loss = loss + self.ccua_ucl_weight * ucl + self.ccua_alignment_weight * alignment
            diagnostics.update({"ccua_ucl": ucl, "ccua_alignment": alignment})

        if self.method == "ipsvt":
            hook = self.ipsvt_hook
            assert hook is not None  # guarded above; narrows Optional for type checkers
            auxiliary, hook_diag = self._hook_loss(
                hook,
                {"model": model, "x0": x0, "x_t": x_t, "t": t, "noise": noise, "y": y,
                 "augm": augm, "step": step, "base_loss": dsm},
                name="ipsvt",
                like=dsm,
            )
            loss = loss + auxiliary
            diagnostics["ipsvt_aux"] = auxiliary
            diagnostics.update(hook_diag)

        diagnostics["loss"] = loss
        return loss, diagnostics


def compute_objective(method: str, model: Callable[..., Any], x0: Tensor, y: Tensor, **kwargs: Any) -> tuple[Tensor, dict[str, Tensor]]:
    """One-shot convenience wrapper; retain ``UnifiedObjective`` for buffers/state."""
    init_keys = {
        "T", "beta_1", "beta_T", "num_classes", "class_probs", "cfg_dropout", "cbdm_tau", "t2h_mode",
        "t2h_cut_time", "transfer_target_hook", "coral_weight", "coral_temperature", "coral_time_scale",
        "ccua_ucl_weight", "ccua_alignment_weight", "ccua_temperature", "ipsvt_hook", "cm_hook",
    }
    init = {key: kwargs.pop(key) for key in tuple(kwargs) if key in init_keys}
    return UnifiedObjective(method, **init).to(x0.device)(model, x0, y, **kwargs)


__all__ = ["METHODS", "ObjectiveResult", "UnifiedObjective", "compute_objective"]
