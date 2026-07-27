"""Small, deterministic helpers for paper-facing paired-seed comparisons."""
from __future__ import annotations

from typing import Mapping

import numpy as np


METRIC_DIRECTIONS = {
    "FID": "lower",
    "KID": "lower",
    "IS": "higher",
    "F_8": "higher",
    "Recall": "higher",
    "F_1_8": "higher",
    "ImprovedPrecision": "higher",
}


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def paired_advantage(
    candidate: Mapping[int, float],
    comparator: Mapping[int, float],
    direction: str,
    *,
    repetitions: int = 10_000,
    confidence_level: float = 0.95,
) -> dict[str, float | int | bool | None]:
    """Return candidate advantage over comparator on paired seeds.

    Positive values always favor the candidate.  The bootstrap resamples the
    paired model-seed differences, never individual generated images.
    """
    common = sorted(set(candidate) & set(comparator))
    if len(common) < 2:
        return {"n_pairs": len(common), "mean": None, "std": None,
                "ci95_low": None, "ci95_high": None, "winner": False}
    cand = np.asarray([candidate[seed] for seed in common], dtype=np.float64)
    base = np.asarray([comparator[seed] for seed in common], dtype=np.float64)
    if direction == "lower":
        differences = base - cand
    elif direction == "higher":
        differences = cand - base
    else:
        raise ValueError(f"unknown metric direction: {direction}")
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(differences), size=(int(repetitions), len(differences)))
    means = differences[draws].mean(axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    mean, std = mean_std(differences.tolist())
    return {
        "n_pairs": len(common),
        "mean": mean,
        "std": std,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "winner": bool(low > 0.0),
    }


def ranks(values: Mapping[str, float], direction: str) -> dict[str, int]:
    reverse = direction == "higher"
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    return {method: index + 1 for index, (method, _) in enumerate(ordered)}
