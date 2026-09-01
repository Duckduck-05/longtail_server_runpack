#!/usr/bin/env python3
"""Train stochastic partial-pooling scores on CIFAR-LT.

This entry point is intentionally independent of the historical auxiliary-loss
trainer in ``main.py``.  It implements exactly one balanced diffusion loss and
one frequency-weighted KL regularizer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from dataset import ImbalanceCIFAR10, ImbalanceCIFAR100
from model.partial_pooling import StochasticPartialPoolingUNet


LOGGER = logging.getLogger("partial_pooling")


@dataclass(frozen=True)
class ExperimentConfig:
    data_type: str
    root: str
    imbalance_factor: float
    seed: int
    total_steps: int
    batch_size: int
    microbatch_size: int
    num_workers: int
    learning_rate: float
    warmup_steps: int
    grad_clip: float
    ema_decay: float
    diffusion_steps: int
    beta_1: float
    beta_T: float
    channels: int
    channel_mult: tuple[int, ...]
    attention: tuple[int, ...]
    num_res_blocks: int
    dropout: float
    rank: int
    time_bins: int
    lambda_kl: float
    eb_stage_steps: int
    checkpoint_every: int
    log_every: int
    compile: bool


class TensorCIFARLT(Dataset):
    """CIFAR-LT with stateless per-example horizontal flips.

    The batch sampler passes ``(index, flip)`` keys, making augmentation exactly
    reproducible after a checkpoint resume even with multiple loader workers.
    """

    def __init__(self, base: Dataset) -> None:
        self.data = base.data
        self.targets = np.asarray(base.targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, key: tuple[int, bool]) -> tuple[torch.Tensor, int]:
        index, flip = key
        image = torch.from_numpy(self.data[index]).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        if flip:
            image = image.flip(-1)
        return image, int(self.targets[index])


class StepBalancedBatchSampler(Sampler[list[tuple[int, bool]]]):
    """Stateless, exactly resumable class-balanced batch sampling."""

    def __init__(
        self,
        targets: np.ndarray,
        *,
        num_class: int,
        batch_size: int,
        seed: int,
        start_step: int,
        stop_step: int,
    ) -> None:
        self.indices = [
            torch.from_numpy(np.flatnonzero(targets == class_id)) for class_id in range(num_class)
        ]
        if any(len(values) == 0 for values in self.indices):
            raise ValueError("every class must contain at least one training example")
        self.num_class = num_class
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.stop_step = stop_step

    def __iter__(self) -> Iterator[list[tuple[int, bool]]]:
        for step in range(self.start_step, self.stop_step):
            generator = torch.Generator().manual_seed(self.seed + 1_000_003 * step)
            classes = torch.randint(
                self.num_class, (self.batch_size,), generator=generator)
            flips = torch.randint(2, (self.batch_size,), generator=generator)
            batch: list[tuple[int, bool]] = []
            for class_id, flip in zip(classes.tolist(), flips.tolist()):
                choices = self.indices[class_id]
                offset = int(torch.randint(len(choices), (), generator=generator))
                batch.append((int(choices[offset]), bool(flip)))
            yield batch

    def __len__(self) -> int:
        return self.stop_step - self.start_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-type", choices=("cifar10lt", "cifar100lt"), default="cifar100lt")
    parser.add_argument("--root", default="./data")
    parser.add_argument("--imbalance-factor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--microbatch-size", type=int, default=0,
        help="forward/backward chunk size; 0 uses the full batch while preserving one optimizer update per batch")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--diffusion-steps", type=int, default=1_000)
    parser.add_argument("--beta-1", type=float, default=1e-4)
    parser.add_argument("--beta-T", type=float, default=0.02)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--channel-mult", type=int, nargs="+", default=(1, 2, 2, 2))
    parser.add_argument("--attention", type=int, nargs="+", default=(1,))
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--eb-stage-steps", type=int, default=5_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "total_steps", "batch_size", "learning_rate", "grad_clip", "ema_decay",
        "diffusion_steps", "beta_1", "beta_T", "channels", "rank", "time_bins",
        "eb_stage_steps", "checkpoint_every", "log_every",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.lambda_kl < 0:
        raise ValueError("--lambda-kl must be non-negative")
    if not 0 < args.imbalance_factor <= 1:
        raise ValueError("--imbalance-factor must be in (0, 1]")
    if not 0 < args.ema_decay < 1:
        raise ValueError("--ema-decay must be in (0, 1)")
    if args.warmup_steps < 0 or args.num_workers < 0:
        raise ValueError("warmup steps and worker count must be non-negative")
    if args.microbatch_size < 0 or args.microbatch_size > args.batch_size:
        raise ValueError("--microbatch-size must be 0 or in [1, batch-size]")


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_type=args.data_type,
        root=str(Path(args.root).resolve()),
        imbalance_factor=args.imbalance_factor,
        seed=args.seed,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        diffusion_steps=args.diffusion_steps,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        channels=args.channels,
        channel_mult=tuple(args.channel_mult),
        attention=tuple(args.attention),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        rank=args.rank,
        time_bins=args.time_bins,
        lambda_kl=args.lambda_kl,
        eb_stage_steps=args.eb_stage_steps,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        compile=args.compile,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_dataset(config: ExperimentConfig) -> tuple[TensorCIFARLT, torch.Tensor, str]:
    dataset_cls = ImbalanceCIFAR100 if config.data_type == "cifar100lt" else ImbalanceCIFAR10
    base = dataset_cls(
        root=config.root,
        imb_type="exp",
        imb_factor=config.imbalance_factor,
        rand_number=0,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
    )
    dataset = TensorCIFARLT(base)
    num_class = 100 if config.data_type == "cifar100lt" else 10
    counts = torch.bincount(torch.from_numpy(dataset.targets), minlength=num_class).float()
    fingerprint = hashlib.sha256()
    fingerprint.update(config.data_type.encode("ascii"))
    fingerprint.update(np.ascontiguousarray(dataset.targets).tobytes())
    fingerprint.update(np.ascontiguousarray(dataset.data).tobytes())
    return dataset, counts, fingerprint.hexdigest()


def build_model(config: ExperimentConfig, num_class: int) -> StochasticPartialPoolingUNet:
    return StochasticPartialPoolingUNet(
        T=config.diffusion_steps,
        ch=config.channels,
        ch_mult=list(config.channel_mult),
        attn=list(config.attention),
        num_res_blocks=config.num_res_blocks,
        dropout=config.dropout,
        augm=False,
        num_class=num_class,
        rank=config.rank,
        time_bins=config.time_bins,
    )


@torch.no_grad()
def update_ema(source: torch.nn.Module, target: torch.nn.Module, decay: float) -> None:
    source_state = source.state_dict()
    for name, target_value in target.state_dict().items():
        source_value = source_state[name]
        if name == "prior_tau2" or not torch.is_floating_point(target_value):
            target_value.copy_(source_value)
        else:
            target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)


def rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).parent / "model" / "partial_pooling.py"):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    config: ExperimentConfig,
    model: StochasticPartialPoolingUNet,
    ema_model: StochasticPartialPoolingUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataset_sha256: str,
) -> Path:
    path = output_dir / f"ckpt_{step}.pt"
    temporary = output_dir / f".ckpt_{step}.tmp"
    torch.save({
        "schema": "stochastic-partial-pooling-v1",
        "step": step,
        "config": asdict(config),
        "dataset_sha256": dataset_sha256,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": rng_state(),
    }, temporary)
    os.replace(temporary, path)
    latest = output_dir / "latest.pt"
    temporary_link = output_dir / ".latest.tmp"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(path.name)
    os.replace(temporary_link, latest)
    return path


def load_checkpoint(
    path: Path,
    *,
    config: ExperimentConfig,
    model: StochasticPartialPoolingUNet,
    ema_model: StochasticPartialPoolingUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataset_sha256: str,
) -> int:
    # This is a trusted, trainer-authored checkpoint and includes Python/NumPy
    # RNG states, which are intentionally outside PyTorch's weights-only format.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "stochastic-partial-pooling-v1":
        raise RuntimeError("checkpoint schema mismatch")
    saved_config = checkpoint["config"]
    current_config = asdict(config)
    allowed_changes = {"total_steps", "num_workers", "checkpoint_every", "log_every", "compile"}
    mismatches = {
        key: (saved_config[key], current_config[key])
        for key in saved_config
        if key not in allowed_changes and saved_config[key] != current_config[key]
    }
    if mismatches:
        raise RuntimeError(f"resume configuration mismatch: {mismatches}")
    if checkpoint["dataset_sha256"] != dataset_sha256:
        raise RuntimeError("resume dataset fingerprint mismatch")
    model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["step"])


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "train.log"),
        ],
    )
    set_seed(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset, class_counts, dataset_sha256 = build_dataset(config)
    num_class = len(class_counts)
    model = build_model(config, num_class).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def warmup(step: int) -> float:
        if config.warmup_steps == 0:
            return 1.0
        return min(step + 1, config.warmup_steps) / config.warmup_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup)
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(
            args.resume,
            config=config,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset_sha256=dataset_sha256,
        )
    if start_step >= config.total_steps:
        raise ValueError("resume step must be smaller than total steps")

    sampler = StepBalancedBatchSampler(
        dataset.targets,
        num_class=num_class,
        batch_size=config.batch_size,
        seed=config.seed,
        start_step=start_step,
        stop_step=config.total_steps,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=device.type == "cuda",
        # Keep worker-base seeding off the model/noise RNG. Augmentations are
        # stateless, so this generator need not be checkpointed.
        generator=torch.Generator().manual_seed(config.seed + 97_531),
    )
    train_model: torch.nn.Module = model
    if config.compile:
        train_model = torch.compile(model)

    betas = torch.linspace(config.beta_1, config.beta_T, config.diffusion_steps, device=device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    method_contract = {
        "schema": "stochastic-partial-pooling-contract-v1",
        "objective": "balanced_diffusion_mse + lambda_kl * frequency_weighted_kl",
        "global_branch_receives_label": False,
        "class_law": "mean(class_counts) / class_count",
        "time_law": "balanced empirical-Bayes posterior second moment",
        "inference_random_effect": "posterior_mean",
        "config": asdict(config),
        "class_counts": class_counts.int().tolist(),
        "dataset_sha256": dataset_sha256,
        "source_sha256": source_fingerprint(),
        "torch_version": torch.__version__,
    }
    write_json(args.output_dir / "METHOD_CONTRACT.json", method_contract)
    metrics_path = args.output_dir / "metrics.jsonl"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info(
        "start step=%d/%d dataset=%s n=%d params=%.2fM lambda=%g",
        start_step, config.total_steps, config.data_type, len(dataset),
        parameter_count / 1e6, config.lambda_kl,
    )

    class_counts_device = class_counts.to(device)
    started_at = time.monotonic()
    last_checkpoint: Path | None = None
    model.train()
    for offset, (images, labels) in enumerate(loader):
        step = start_step + offset
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        timesteps = torch.randint(config.diffusion_steps, (images.shape[0],), device=device)
        noise = torch.randn_like(images)
        x_t = (
            sqrt_alpha_bar[timesteps, None, None, None] * images
            + sqrt_one_minus_alpha_bar[timesteps, None, None, None] * noise
        )

        optimizer.zero_grad(set_to_none=True)
        regularizer = model.partial_pooling_regularizer(class_counts_device)
        if config.lambda_kl > 0:
            (config.lambda_kl * regularizer).backward()
        microbatch_size = config.microbatch_size or images.shape[0]
        base_loss = torch.zeros((), device=device)
        for micro_start in range(0, images.shape[0], microbatch_size):
            micro_stop = min(micro_start + microbatch_size, images.shape[0])
            fraction = (micro_stop - micro_start) / images.shape[0]
            prediction = train_model(
                x_t[micro_start:micro_stop],
                timesteps[micro_start:micro_stop],
                y=labels[micro_start:micro_stop],
            )
            micro_loss = F.mse_loss(
                prediction.float(), noise[micro_start:micro_stop].float())
            (fraction * micro_loss).backward()
            base_loss = base_loss + fraction * micro_loss.detach()
        loss = base_loss + config.lambda_kl * regularizer.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        completed = step + 1
        eb_updated = completed % config.eb_stage_steps == 0
        if eb_updated:
            model.empirical_bayes_update()
        update_ema(model, ema_model, config.ema_decay)

        if completed == 1 or completed % config.log_every == 0 or eb_updated:
            elapsed = time.monotonic() - started_at
            record = {
                "step": completed,
                "loss": float(loss.detach().cpu()),
                "base_loss": float(base_loss.detach().cpu()),
                "regularizer": float(regularizer.detach().cpu()),
                "lambda_kl": config.lambda_kl,
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
                "updates_per_second": (completed - start_step) / elapsed,
                "eb_updated": eb_updated,
                **model.posterior_summary(class_counts_device),
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            LOGGER.info(
                "step=%d loss=%.5f base=%.5f kl=%.5f grad=%.3f ups=%.3f",
                completed, record["loss"], record["base_loss"],
                record["regularizer"], record["grad_norm"], record["updates_per_second"],
            )

        if completed % config.checkpoint_every == 0 or completed == config.total_steps:
            last_checkpoint = save_checkpoint(
                args.output_dir,
                step=completed,
                config=config,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataset_sha256=dataset_sha256,
            )
            LOGGER.info("checkpoint=%s", last_checkpoint)

    elapsed = time.monotonic() - started_at
    if last_checkpoint is None or not last_checkpoint.exists():
        raise RuntimeError("final checkpoint is missing")
    write_json(args.output_dir / "RUNTIME_SUMMARY.json", {
        "schema": "stochastic-partial-pooling-runtime-v1",
        "status": "completed",
        "completed_updates": config.total_steps,
        "elapsed_seconds": elapsed,
        "updates_per_second": (config.total_steps - start_step) / elapsed,
        "checkpoint": str(last_checkpoint),
        "dataset_sha256": dataset_sha256,
        "source_sha256": source_fingerprint(),
        "seed": config.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "config": asdict(config),
        "posterior": model.posterior_summary(class_counts_device),
    })
    LOGGER.info("completed updates=%d elapsed=%.1fs", config.total_steps, elapsed)


if __name__ == "__main__":
    main()
