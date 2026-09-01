"""Runtime arithmetic for the CBDM ESC null-branch objective.

The training loop owns data loading and tensor reductions; this module owns the
class-reference and low-timestep scalar contracts so they can be audited without
starting a training job.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal


LEGACY_CLASS_MEAN = "legacy_class_mean"
EMPIRICAL_EXPECTATION = "empirical_expectation"
WeightNormalization = Literal["legacy_class_mean", "empirical_expectation"]

LEGACY_AMPLIFIED = "legacy_amplified"
LOW_AVERAGE = "low_average"
TRUNCATED_FULL = "truncated_full"
LowTObjectiveMode = Literal["legacy_amplified", "low_average", "truncated_full"]
WEIGHT_NORMALIZATIONS = (LEGACY_CLASS_MEAN, EMPIRICAL_EXPECTATION)
LOW_T_OBJECTIVE_MODES = (LEGACY_AMPLIFIED, LOW_AVERAGE, TRUNCATED_FULL)

_CIFAR10_CONFUSABILITY = (0.10, 0.10, 0.30, 0.55, 0.45, 0.875, 0.20, 0.40, 0.10, 0.000)
_HISTORICAL_NULL_BATCH_PROBABILITY = 0.1


def _finite_positive(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) and value > 0.0 for value in result):
        raise ValueError(f"{name} must contain only finite positive values")
    return result


def _prior(counts: Sequence[float]) -> tuple[float, ...]:
    checked_counts = _finite_positive(counts, name="counts")
    total = math.fsum(checked_counts)
    return tuple(count / total for count in checked_counts)


def _matching(values: Sequence[float], *, name: str, size: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} has length {len(result)}, expected {size}")
    if not all(math.isfinite(value) and value > 0.0 for value in result):
        raise ValueError(f"{name} must contain only finite positive values")
    return result


def confusability_vector(class_count: int) -> tuple[float, ...]:
    """Return CIFAR-10 confusability anchors or a neutral class-sized fallback."""
    if isinstance(class_count, bool) or int(class_count) != class_count or class_count <= 0:
        raise ValueError("class_count must be a positive integer")
    size = int(class_count)
    return _CIFAR10_CONFUSABILITY if size == len(_CIFAR10_CONFUSABILITY) else (0.0,) * size


def raw_bnt_weights(counts: Sequence[float], alpha: float) -> tuple[float, ...]:
    """Return BNT weights proportional to ``(n_max / n_c) ** alpha``."""
    checked_counts = _finite_positive(counts, name="counts")
    exponent = float(alpha)
    if not math.isfinite(exponent) or exponent < 0.0:
        raise ValueError("bnt_alpha must be finite and non-negative")
    maximum = max(checked_counts)
    return tuple((maximum / count) ** exponent for count in checked_counts)


def normalize_class_weights(
    raw_weights: Sequence[float],
    *,
    rho: Sequence[float],
    normalization: WeightNormalization,
) -> tuple[float, ...]:
    """Normalize class weights under the selected class-reference convention."""
    prior = _matching(rho, name="rho", size=len(raw_weights))
    if not math.isclose(math.fsum(prior), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("rho must sum to one")
    weights = _matching(raw_weights, name="raw_weights", size=len(prior))
    if normalization == LEGACY_CLASS_MEAN:
        denominator = math.fsum(weights) / len(weights)
    elif normalization == EMPIRICAL_EXPECTATION:
        denominator = math.fsum(probability * weight for probability, weight in zip(prior, weights))
    else:
        raise ValueError(f"unknown BNT normalization {normalization!r}")
    return tuple(weight / denominator for weight in weights)


def empirical_weight_mean(rho: Sequence[float], weights: Sequence[float]) -> float:
    prior = _matching(rho, name="rho", size=len(weights))
    class_weights = _matching(weights, name="weights", size=len(prior))
    return math.fsum(probability * weight for probability, weight in zip(prior, class_weights))


def effective_reference_prior(rho: Sequence[float], weights: Sequence[float]) -> tuple[float, ...]:
    """Return the normalized effective class law ``rho_c * w_c``."""
    prior = _matching(rho, name="rho", size=len(weights))
    class_weights = _matching(weights, name="weights", size=len(prior))
    unnormalized = tuple(probability * weight for probability, weight in zip(prior, class_weights))
    total = math.fsum(unnormalized)
    return tuple(value / total for value in unnormalized)


def low_t_multiplier(
    mode: LowTObjectiveMode,
    *,
    total_steps: int,
    low_steps: int,
    used_low_t: bool,
) -> float:
    """Return the low-t scalar, or one when this batch used the full schedule."""
    if mode not in LOW_T_OBJECTIVE_MODES:
        raise ValueError(f"unknown low-t objective mode {mode!r}")
    if not used_low_t:
        return 1.0
    if isinstance(total_steps, bool) or isinstance(low_steps, bool):
        raise ValueError("total_steps and low_steps must be integers")
    if int(total_steps) != total_steps or int(low_steps) != low_steps:
        raise ValueError("total_steps and low_steps must be integers")
    total, low = int(total_steps), int(low_steps)
    if total <= 0 or not 0 < low <= total:
        raise ValueError("low_steps must satisfy 0 < low_steps <= total_steps")
    if mode == LEGACY_AMPLIFIED:
        return total / low
    if mode == LOW_AVERAGE:
        return 1.0
    if mode == TRUNCATED_FULL:
        return low / total
    raise AssertionError("validated low-t objective mode was not handled")


def null_batch_probability(*, cfg_enabled: bool, cb_enabled: bool) -> float:
    """Return the null-branch probability used by the trainer.

    ``GaussianDiffusionTrainer.forward`` samples a null batch with probability
    ``1 / 10`` whenever CFG or class-balancing training is enabled; otherwise
    that branch is unreachable.  Keeping the constant here makes the JSON
    contract directly auditable against the training control flow.
    """
    return _HISTORICAL_NULL_BATCH_PROBABILITY if (cfg_enabled or cb_enabled) else 0.0


def low_t_mixture_expectation(*, low_branch_multiplier: float, mix_lambda: float) -> float:
    """Return ``(1 - lambda) + lambda * m`` for a null-batch low-t mixture."""
    probability = float(mix_lambda)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("low_t_mix_lambda must be finite and lie in [0, 1]")
    multiplier = float(low_branch_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("low_branch_multiplier must be finite and positive")
    return (1.0 - probability) + probability * multiplier


def runtime_null_class_weights(counts, *, bnt_alpha: float, canb_beta: float,
                               normalization: WeightNormalization):
    """Return runtime BNT/CANB weights using the incoming tensor's dtype/device.

    Keeping this tensor path mirrors the historical float32 legacy reduction,
    while the JSON contract below remains a portable double-precision audit.
    """
    import torch

    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("counts must be a non-empty one-dimensional tensor")
    if not torch.is_floating_point(counts):
        counts = counts.float()
    if not bool(torch.isfinite(counts).all()) or not bool((counts > 0).all()):
        raise ValueError("counts must contain only finite positive values")
    beta = float(canb_beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("canb_beta must be finite and non-negative")
    raw = (counts.max() / counts) ** float(bnt_alpha)
    if beta > 0.0:
        confuse = torch.tensor(confusability_vector(len(counts)), dtype=counts.dtype, device=counts.device)
        raw = raw * (1.0 + beta * confuse)
    if normalization == LEGACY_CLASS_MEAN:
        denominator = raw.mean()
    elif normalization == EMPIRICAL_EXPECTATION:
        rho = counts / counts.sum()
        denominator = (rho * raw).sum()
    else:
        raise ValueError(f"unknown BNT normalization {normalization!r}")
    return raw / denominator


def build_esc_objective_contract(
    counts: Sequence[float],
    *,
    bnt_alpha: float = 0.0,
    canb_beta: float = 0.0,
    normalization: WeightNormalization = LEGACY_CLASS_MEAN,
    low_t_objective_mode: LowTObjectiveMode = LEGACY_AMPLIFIED,
    total_steps: int = 1_000,
    low_steps: int = 0,
    low_t_enabled: bool | None = None,
    low_t_mix_lambda: float = 1.0,
    cfg_enabled: bool = False,
    cb_enabled: bool = False,
    null_reweight_enabled: bool | None = None,
    runtime_weights: Sequence[float] | None = None,
) -> dict[str, object]:
    """Build the JSON-serializable ESC contract written before training."""
    checked_counts = _finite_positive(counts, name="counts")
    rho = _prior(checked_counts)
    beta = float(canb_beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("canb_beta must be finite and non-negative")
    bnt_raw = raw_bnt_weights(checked_counts, bnt_alpha)
    bnt_weights = normalize_class_weights(bnt_raw, rho=rho, normalization=normalization)
    confuse = confusability_vector(len(checked_counts))
    canb_raw = tuple(weight * (1.0 + beta * confusion)
                     for weight, confusion in zip(bnt_raw, confuse))
    canb_weights = normalize_class_weights(canb_raw, rho=rho, normalization=normalization)
    active_weights = canb_weights if beta > 0.0 else bnt_weights
    if runtime_weights is not None:
        active_weights = _matching(runtime_weights, name="runtime_weights", size=len(rho))
    if null_reweight_enabled is None:
        # main.py applies BNT/CANB only when bnt_alpha is positive.
        null_reweight_enabled = float(bnt_alpha) > 0.0
    applied_weights = active_weights if null_reweight_enabled else (1.0,) * len(rho)
    if low_t_enabled is None:
        low_t_enabled = low_steps > 0
    low_branch_multiplier = low_t_multiplier(
        low_t_objective_mode,
        total_steps=total_steps,
        low_steps=low_steps,
        used_low_t=bool(low_steps > 0),
    )
    full_branch_multiplier = 1.0
    mixture_expectation = low_t_mixture_expectation(
        low_branch_multiplier=low_branch_multiplier,
        mix_lambda=low_t_mix_lambda)
    null_probability = null_batch_probability(
        cfg_enabled=bool(cfg_enabled), cb_enabled=bool(cb_enabled))
    empirical_mean = empirical_weight_mean(rho, applied_weights)
    expected_null_batch_scale = empirical_mean * mixture_expectation
    expected_per_training_update_scale = (
        (1.0 - null_probability) + null_probability * expected_null_batch_scale)
    return {
        "schema": "cbdm-esc-objective-contract-v2",
        "counts": [int(value) if value.is_integer() else value for value in checked_counts],
        "rho": list(rho),
        "bnt_alpha": float(bnt_alpha),
        "canb_beta": beta,
        "confusability": list(confuse),
        "normalization": normalization,
        "bnt_weights": list(bnt_weights),
        "canb_weights": list(canb_weights),
        "weights": list(applied_weights),
        "configured_weights": list(active_weights),
        "null_reweight_enabled": bool(null_reweight_enabled),
        "effective_prior": list(effective_reference_prior(rho, applied_weights)),
        "empirical_mean": empirical_mean,
        "mode": low_t_objective_mode,
        # ``multiplier`` is retained as the historical low-branch name.  The
        # expectation fields below are the canonical update-level contract.
        "multiplier": low_branch_multiplier,
        "null_batch_probability": null_probability,
        "expected_null_batch_scale": expected_null_batch_scale,
        "combined_expected_per_training_update_scale": expected_per_training_update_scale,
        "combined_expected_scale": expected_per_training_update_scale,
        "low_t": {
            "total_steps": int(total_steps),
            "low_steps": int(low_steps),
            "enabled": bool(low_t_enabled),
            "mix_lambda": float(low_t_mix_lambda),
            "low_branch_multiplier": low_branch_multiplier,
            "full_branch_multiplier": full_branch_multiplier,
            "mixture_expectation": mixture_expectation,
        },
    }


def write_esc_objective_contract(path: str | Path, contract: dict[str, object]) -> None:
    """Write the pre-training ESC contract as deterministic JSON."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
