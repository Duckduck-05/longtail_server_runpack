#!/usr/bin/env python3
"""Train the preregistered 6k APP U/D/S/A mechanism gate.

The model is a class-only linear-in-z output factorization, not a generic
z-conditioned U-Net throughout. The loss is a class-balanced DSM pseudo-objective
plus β/n_c times one class-level Gaussian random-effect KL. The prior update is
alternating MAP; this program makes no exact-ELBO or epistemic-posterior claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from dataset import ImbalanceCIFAR10, ImbalanceCIFAR100
from model.anisotropic_partial_pooling import APPLinearFactorUNet
from model.model import UNet
from train_partial_pooling import StepBalancedBatchSampler, TensorCIFARLT


SCHEMA = "app-gate-v1"


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    config: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset_sha256: str,
    source_sha256: str,
    ema_model: nn.Module | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    """Atomically save all state needed for exact eager-mode resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ckpt_{step}.pt"
    _atomic_torch_save(path, {
        "schema": SCHEMA,
        "step": step,
        "config": config,
        "dataset_sha256": dataset_sha256,
        "source_sha256": source_sha256,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": _rng_state(),
    })
    latest = output_dir / "latest.pt"
    temporary_link = output_dir / ".latest.tmp"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(path.name)
    os.replace(temporary_link, latest)
    return path


def load_checkpoint(
    path: Path,
    *,
    config: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset_sha256: str,
    source_sha256: str,
    ema_model: nn.Module | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Load a trusted APP-gate checkpoint after strict provenance checks."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise RuntimeError("checkpoint schema mismatch")
    if checkpoint["config"] != config:
        raise RuntimeError("resume configuration mismatch")
    if checkpoint["dataset_sha256"] != dataset_sha256:
        raise RuntimeError("resume dataset fingerprint mismatch")
    if checkpoint["source_sha256"] != source_sha256:
        raise RuntimeError("resume source fingerprint mismatch")
    if (checkpoint["ema_model"] is None) != (ema_model is None):
        raise RuntimeError("resume EMA configuration mismatch")
    if (checkpoint["scheduler"] is None) != (scheduler is None):
        raise RuntimeError("resume scheduler configuration mismatch")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if ema_model is not None:
        ema_model.load_state_dict(checkpoint["ema_model"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    _restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["step"])


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for source in (
        Path(__file__),
        root / "model" / "anisotropic_partial_pooling.py",
        root / "model" / "model.py",
        root / "train_partial_pooling.py",
        root / "dataset.py",
    ):
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def checkpoint_is_due(
    step: int,
    *,
    total_steps: int,
    checkpoint_every: int,
    diagnostic_steps: tuple[int, ...] | list[int],
) -> bool:
    """Return whether a checkpoint is mandatory at this completed update."""
    return step == total_steps or step % checkpoint_every == 0 or step in diagnostic_steps


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standard", "deterministic", "isotropic", "anisotropic"), required=True)
    parser.add_argument("--data-type", choices=("cifar10lt", "cifar100lt"), default="cifar100lt")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--imbalance-factor", type=float, default=0.01)
    parser.add_argument("--total-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--beta-1", type=float, default=1e-4)
    parser.add_argument("--beta-T", type=float, default=0.02)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--channel-mult", type=int, nargs="+", default=(1, 2, 2, 2))
    parser.add_argument("--attention", type=int, nargs="+", default=(1,))
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0847)
    parser.add_argument("--prior-warmup-steps", type=int, default=5000)
    parser.add_argument("--map-update-step", type=int, default=5000)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--prior-scale2", type=float, default=1e-4)
    parser.add_argument("--covariance-floor", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--diagnostic-steps", type=int, nargs="+", default=(4999, 5000, 6000))
    parser.add_argument("--diagnostic-per-class", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "total_steps", "batch_size", "microbatch_size", "learning_rate", "grad_clip", "ema_decay",
        "diffusion_steps", "beta_1", "beta_T", "channels", "rank", "prior_strength", "prior_scale2",
        "covariance_floor", "checkpoint_every", "diagnostic_per_class",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.beta < 0 or args.warmup_steps < 0 or args.prior_warmup_steps < 0:
        raise ValueError("beta and warm-up steps must be non-negative")
    if not 0 < args.imbalance_factor <= 1 or not 0 < args.ema_decay < 1:
        raise ValueError("invalid imbalance factor or EMA decay")
    if args.microbatch_size > args.batch_size:
        raise ValueError("microbatch size cannot exceed batch size")
    if args.map_update_step < args.prior_warmup_steps:
        raise ValueError("MAP update must not precede the frozen-prior warm-up")


def _config(args: argparse.Namespace) -> dict[str, Any]:
    ignored = {"output_dir", "resume", "device", "num_workers", "total_steps"}
    config = {key: value for key, value in vars(args).items() if key not in ignored}
    config["channel_mult"] = list(config["channel_mult"])
    config["attention"] = list(config["attention"])
    config["diagnostic_steps"] = list(config["diagnostic_steps"])
    return config


def _build_dataset(args: argparse.Namespace) -> tuple[TensorCIFARLT, torch.Tensor, str]:
    dataset_cls = ImbalanceCIFAR100 if args.data_type == "cifar100lt" else ImbalanceCIFAR10
    base = dataset_cls(
        root=args.root, imb_type="exp", imb_factor=args.imbalance_factor, rand_number=0,
        train=True, transform=None, target_transform=None, download=True)
    dataset = TensorCIFARLT(base)
    num_class = 100 if args.data_type == "cifar100lt" else 10
    counts = torch.bincount(torch.from_numpy(dataset.targets), minlength=num_class).float()
    digest = hashlib.sha256()
    digest.update(args.data_type.encode("ascii"))
    digest.update(np.ascontiguousarray(dataset.targets).tobytes())
    digest.update(np.ascontiguousarray(dataset.data).tobytes())
    return dataset, counts, digest.hexdigest()


def _build_model(args: argparse.Namespace, num_class: int) -> nn.Module:
    architecture = dict(
        T=args.diffusion_steps, ch=args.channels, ch_mult=list(args.channel_mult), attn=list(args.attention),
        num_res_blocks=args.num_res_blocks, dropout=args.dropout, augm=False, num_class=num_class)
    if args.mode == "standard":
        return UNet(cond=True, **architecture)
    return APPLinearFactorUNet(
        rank=args.rank,
        posterior_structure="diagonal" if args.mode == "isotropic" else "full",
        population_structure="isotropic" if args.mode == "isotropic" else "full",
        covariance_floor=args.covariance_floor,
        **architecture,
    )


def _update_ema(source: nn.Module, target: nn.Module, decay: float) -> None:
    source_state = source.state_dict()
    for name, target_value in target.state_dict().items():
        source_value = source_state[name]
        if name == "population_covariance" or not torch.is_floating_point(target_value):
            target_value.copy_(source_value)
        else:
            target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)


def _model_prediction(
    model: nn.Module,
    mode: str,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_only: bool = False,
) -> torch.Tensor:
    if mode == "standard":
        if global_only:
            raise ValueError("standard mode has no global-only component")
        return model(x_t, timesteps, labels)
    return model(
        x_t, timesteps, labels,
        sample_posterior=False if mode == "deterministic" else None,
        global_only=global_only,
    )


def _fixed_diagnostic(
    args: argparse.Namespace,
    class_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    dataset_cls = CIFAR100 if args.data_type == "cifar100lt" else CIFAR10
    base = dataset_cls(root=args.root, train=False, transform=None, target_transform=None, download=True)
    labels_np = np.asarray(base.targets, dtype=np.int64)
    selected: list[int] = []
    for class_id in range(len(class_counts)):
        indices = np.flatnonzero(labels_np == class_id)
        selected.extend(indices[:args.diagnostic_per_class].tolist())
    images = torch.from_numpy(base.data[selected]).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    labels = torch.from_numpy(labels_np[selected]).long()
    generator = torch.Generator().manual_seed(args.seed + 71_337)
    timesteps = torch.randint(args.diffusion_steps, (len(labels),), generator=generator)
    noise = torch.randn(images.shape, generator=generator)
    order = torch.argsort(class_counts, descending=True)
    third = max(1, len(class_counts) // 3)
    groups = {"head": order[:third], "medium": order[third:-third], "tail": order[-third:]}
    return images, labels, timesteps, noise, groups


@torch.no_grad()
def heldout_dsm_diagnostic(
    model: nn.Module,
    *,
    mode: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    groups: dict[str, torch.Tensor],
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float | None]:
    """Measure held-out full/global-only DSM under fixed corruptions and labels."""
    was_training = model.training
    model.eval()
    per_example_full: list[torch.Tensor] = []
    per_example_global: list[torch.Tensor] = []
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        image = images[start:stop].to(device)
        label = labels[start:stop].to(device)
        timestep = timesteps[start:stop].to(device)
        target = noise[start:stop].to(device)
        x_t = sqrt_alpha_bar[timestep, None, None, None] * image + sqrt_one_minus_alpha_bar[timestep, None, None, None] * target
        full = _model_prediction(model, mode, x_t, timestep, label)
        per_example_full.append((full.float() - target.float()).square().mean(dim=(1, 2, 3)).cpu())
        if mode != "standard":
            global_prediction = _model_prediction(model, mode, x_t, timestep, label, global_only=True)
            per_example_global.append((global_prediction.float() - target.float()).square().mean(dim=(1, 2, 3)).cpu())
    if was_training:
        model.train()
    full_losses = torch.cat(per_example_full)
    global_losses = torch.cat(per_example_global) if per_example_global else None
    result: dict[str, float | None] = {}
    for name, group in groups.items():
        mask = torch.isin(labels, group.cpu())
        full_value = float(full_losses[mask].mean())
        result[f"heldout_full_dsm_{name}"] = full_value
        if global_losses is None:
            result[f"heldout_global_dsm_{name}"] = None
            result[f"heldout_local_utility_{name}"] = None
        else:
            global_value = float(global_losses[mask].mean())
            result[f"heldout_global_dsm_{name}"] = global_value
            result[f"heldout_local_utility_{name}"] = 1.0 - full_value / global_value
    return result


def _app_covariance_diagnostics(model: APPLinearFactorUNet) -> dict[str, float]:
    eigenvalues = torch.linalg.eigvalsh(model.population_covariance.float()).clamp_min(
        model.covariance_floor)
    normalized = eigenvalues / eigenvalues.sum()
    effective_rank = torch.exp(-(normalized * normalized.log()).sum())
    return {
        "population_effective_rank": float(effective_rank.cpu()),
        "population_condition_number": float((eigenvalues.max() / eigenvalues.min()).cpu()),
        "population_floor_fraction": float((eigenvalues <= model.covariance_floor * 1.01).float().mean().cpu()),
    }


def _truncate_metrics(path: Path, maximum_step: int) -> None:
    if not path.exists():
        return
    retained = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["step"] <= maximum_step:
            retained.append(json.dumps(record, sort_keys=True))
    path.write_text("\n".join(retained) + ("\n" if retained else ""))


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _config(args)
    _set_seed(args.seed)
    device = torch.device(args.device)
    dataset, class_counts, dataset_sha256 = _build_dataset(args)
    source_sha256 = _source_fingerprint()
    model = _build_model(args, len(class_counts)).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step + 1, args.warmup_steps) / args.warmup_steps if args.warmup_steps else 1.0)
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(
            args.resume, config=config, model=model, ema_model=ema_model, optimizer=optimizer,
            scheduler=scheduler, dataset_sha256=dataset_sha256, source_sha256=source_sha256)
    if start_step >= args.total_steps:
        raise ValueError("resume step must be less than total steps")
    metrics_path = args.output_dir / "metrics.jsonl"
    _truncate_metrics(metrics_path, start_step)
    sampler = StepBalancedBatchSampler(
        dataset.targets, num_class=len(class_counts), batch_size=args.batch_size, seed=args.seed,
        start_step=start_step, stop_step=args.total_steps)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 97_531))
    beta_schedule = torch.linspace(args.beta_1, args.beta_T, args.diffusion_steps, device=device)
    alpha_bar = torch.cumprod(1.0 - beta_schedule, dim=0)
    sqrt_alpha_bar = alpha_bar.sqrt()
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
    diagnostic = _fixed_diagnostic(args, class_counts)
    class_counts_device = class_counts.to(device)
    method_contract = {
        "schema": SCHEMA,
        "objective": "class-balanced DSM pseudo-objective + beta/n_c group KL",
        "inference": "posterior mean; no posterior sampling",
        "prior_update": "single alternating-MAP update after frozen warm-up",
        "mode": args.mode,
        "beta": args.beta,
        "lambda_equivalent": args.beta / float(class_counts.mean()),
        "dataset_sha256": dataset_sha256,
        "source_sha256": source_sha256,
        "config": config,
    }
    (args.output_dir / "METHOD_CONTRACT.json").write_text(json.dumps(method_contract, indent=2, sort_keys=True) + "\n")
    model.train()
    for offset, (images, labels) in enumerate(loader):
        step = start_step + offset
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        timesteps = torch.randint(args.diffusion_steps, (len(labels),), device=device)
        noise = torch.randn_like(images)
        x_t = sqrt_alpha_bar[timesteps, None, None, None] * images + sqrt_one_minus_alpha_bar[timesteps, None, None, None] * noise
        optimizer.zero_grad(set_to_none=True)
        kl_loss = torch.zeros((), device=device)
        if args.mode in {"isotropic", "anisotropic"}:
            app_model = model
            assert isinstance(app_model, APPLinearFactorUNet)
            kl_loss = args.beta / class_counts_device.mean() * app_model.frequency_weighted_kl(class_counts_device)
            kl_loss.backward()
        base_loss = torch.zeros((), device=device)
        for micro_start in range(0, len(labels), args.microbatch_size):
            micro_stop = min(micro_start + args.microbatch_size, len(labels))
            fraction = (micro_stop - micro_start) / len(labels)
            prediction = _model_prediction(
                model, args.mode, x_t[micro_start:micro_stop], timesteps[micro_start:micro_stop], labels[micro_start:micro_stop])
            micro_loss = F.mse_loss(prediction.float(), noise[micro_start:micro_stop].float())
            (fraction * micro_loss).backward()
            base_loss += fraction * micro_loss.detach()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed = step + 1
        map_updated = completed == args.map_update_step and args.mode in {"isotropic", "anisotropic"}
        if map_updated:
            app_model = model
            assert isinstance(app_model, APPLinearFactorUNet)
            app_model.map_update_population_(prior_strength=args.prior_strength, prior_scale2=args.prior_scale2)
        _update_ema(model, ema_model, args.ema_decay)
        record: dict[str, Any] = {
            "step": completed,
            "base_loss": float(base_loss.cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "loss": float((base_loss + kl_loss.detach()).cpu()),
            "grad_norm": float(torch.as_tensor(gradient_norm).cpu()),
            "learning_rate": scheduler.get_last_lr()[0],
            "map_updated": map_updated,
        }
        if completed in args.diagnostic_steps:
            record.update(heldout_dsm_diagnostic(
                ema_model, mode=args.mode, images=diagnostic[0], labels=diagnostic[1], timesteps=diagnostic[2],
                noise=diagnostic[3], groups=diagnostic[4], sqrt_alpha_bar=sqrt_alpha_bar,
                sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar, device=device))
            if args.mode != "standard":
                record.update(_app_covariance_diagnostics(ema_model))
        if completed % 100 == 0 or completed in args.diagnostic_steps or map_updated:
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if checkpoint_is_due(
            completed,
            total_steps=args.total_steps,
            checkpoint_every=args.checkpoint_every,
            diagnostic_steps=args.diagnostic_steps,
        ):
            save_checkpoint(
                args.output_dir, step=completed, config=config, model=model, ema_model=ema_model,
                optimizer=optimizer, scheduler=scheduler, dataset_sha256=dataset_sha256, source_sha256=source_sha256)
    (args.output_dir / "RUNTIME_SUMMARY.json").write_text(json.dumps({
        "schema": SCHEMA, "status": "completed", "completed_updates": args.total_steps,
        "latest_checkpoint": str(args.output_dir / "latest.pt"), "dataset_sha256": dataset_sha256,
        "source_sha256": source_sha256, "config": config,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
