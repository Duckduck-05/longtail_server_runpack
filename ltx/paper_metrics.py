"""Shared CORAL-2025 metric protocol: VGG16 improved PRD with k=3.

The public CBDM release uses an Inception/k=5 approximation.  This module
implements the evaluator stated in the paper and is deliberately shared by all
methods, including T2H.  It uses chunked exact squared distances, avoiding a
50k×50k host allocation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CORAL paper metrics require a CUDA GPU")
    return torch


def vgg16_fc2(images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Match the released IPR extractor: resize to 224 then VGG16 fc2."""
    torch = _torch()
    from torchvision import models
    try:
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    except AttributeError:  # torchvision < 0.13
        model = models.vgg16(pretrained=True)
    model = model.cuda().eval()
    result = np.empty((len(images), 4096), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size])).float().cuda(non_blocking=True)
            if batch.shape[-2:] != (224, 224):
                batch = torch.nn.functional.interpolate(batch, size=(224, 224))
            features = model.features(batch).flatten(1)
            result[start:start + len(batch)] = model.classifier[:4](features).cpu().numpy()
    return result


def _dist2(query, bank, bank_norm):
    torch = _torch()
    out = query.square().sum(dim=1, keepdim=True) + bank_norm.unsqueeze(0) - 2 * (query @ bank.T)
    return out.clamp_min_(0)


def knn_radii(features: np.ndarray, k: int = 3, query_batch: int = 128) -> np.ndarray:
    """Exact k-NN manifold radii; the self-distance is explicitly excluded."""
    torch = _torch(); bank = torch.from_numpy(np.ascontiguousarray(features)).cuda()
    bank_norm = bank.square().sum(dim=1); radii = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), query_batch):
        query = bank[start:start + query_batch]
        distances = _dist2(query, bank, bank_norm)
        rows = torch.arange(len(query), device=bank.device)
        distances[rows, start + rows] = float("inf")
        radii[start:start + len(query)] = torch.topk(distances, k=k, largest=False).values[:, -1].sqrt().cpu().numpy()
    return radii


def manifold_membership(query_features: np.ndarray, manifold_features: np.ndarray, manifold_radii: np.ndarray,
                        query_batch: int = 128) -> float:
    """Fraction of query features lying in the (exact) k-NN manifold."""
    torch = _torch(); bank = torch.from_numpy(np.ascontiguousarray(manifold_features)).cuda()
    norm = bank.square().sum(dim=1); radii2 = torch.from_numpy(np.ascontiguousarray(manifold_radii)).cuda().square()
    hits = 0
    for start in range(0, len(query_features), query_batch):
        query = torch.from_numpy(np.ascontiguousarray(query_features[start:start + query_batch])).cuda()
        hits += int((_dist2(query, bank, norm) <= radii2.unsqueeze(0)).any(dim=1).sum().item())
    return hits / len(query_features)


def improved_prd_vgg16_k3(generated_images: np.ndarray, reference_features: np.ndarray,
                           reference_radii: np.ndarray, batch_size: int = 64, query_batch: int = 128) -> tuple[float, float]:
    generated_features = vgg16_fc2(generated_images, batch_size=batch_size)
    generated_radii = knn_radii(generated_features, k=3, query_batch=query_batch)
    precision = manifold_membership(generated_features, reference_features, reference_radii, query_batch=query_batch)
    recall = manifold_membership(reference_features, generated_features, generated_radii, query_batch=query_batch)
    return precision, recall
