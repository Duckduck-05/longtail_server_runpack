"""Metric primitives owned by the common T2H evaluation host."""
from __future__ import annotations

import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(
                f"FID covariance product has imaginary component {np.max(np.abs(covmean.imag))}"
            )
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def polynomial_mmd_kid(gen_features, real_features, num_subsets=100, max_subset_size=1000):
    dim = gen_features.shape[1]
    subset = min(min(gen_features.shape[0], real_features.shape[0]), max_subset_size)
    if subset < 2:
        raise ValueError("KID requires at least two real and generated feature vectors.")
    total = 0.0
    for _ in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], subset, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], subset, replace=False)]
        a = (x @ x.T / dim + 1) ** 3 + (y @ y.T / dim + 1) ** 3
        b = (x @ y.T / dim + 1) ** 3
        total += (a.sum() - np.diag(a).sum()) / (subset - 1) - b.sum() * 2 / subset
    return float(total / num_subsets / subset)
