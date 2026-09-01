#!/usr/bin/env python3
"""Train the Conditional Partial Pooling objective on a native CBDM checkpoint.

Every arm uses the same exact class-balanced batches, two native U-Net
forwards (conditional and null), timestep/noise schedule, and calibration mask.
The only difference is the coefficient on L_share and L_spread.  A mask is
estimated on a disjoint subset of the LT training set from counterfactual
conditional utility, then frozen for all optimizer steps.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from dataset import ImbalanceCIFAR10, ImbalanceCIFAR100
from model.model import ResBlock, UNet

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.conditional_partial_pooling import (  # noqa: E402
    AllocationMask,
    calibrate_allocation_mask,
    partial_pooling_losses,
)
from tools.conditional_residual_allocation_probe import one_cell  # noqa: E402

LOGGER = logging.getLogger("conditional_partial_pooling")
BLOCKS = tuple(f"downblocks.{i}" for i in range(0))  # replaced from model below
TIMESTEP_BOUNDARIES = (250, 600)
TIMESTEP_REPRESENTATIVES = (50, 300, 700)


class TensorCIFARLT(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data = np.asarray(data)
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, key: tuple[int, bool]) -> tuple[torch.Tensor, int]:
        index, flip = key
        image = torch.from_numpy(self.data[index]).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        if flip:
            image = image.flip(-1)
        return image, int(self.targets[index])


class ExactClassBatchSampler(Sampler[list[tuple[int, bool]]]):
    """Stateless, reproducible equal-count class batches."""

    def __init__(
        self,
        targets: np.ndarray,
        *,
        num_classes: int,
        states_per_class: int,
        seed: int,
        total_steps: int,
    ) -> None:
        self.indices = [
            np.flatnonzero(targets == class_id) for class_id in range(num_classes)
        ]
        if any(len(values) == 0 for values in self.indices):
            raise ValueError("every class must have at least one training example")
        self.num_classes = num_classes
        self.states_per_class = states_per_class
        self.seed = seed
        self.total_steps = total_steps

    def __iter__(self) -> Iterator[list[tuple[int, bool]]]:
        for step in range(self.total_steps):
            rng = np.random.default_rng(self.seed + 1_000_003 * step)
            batch: list[tuple[int, bool]] = []
            for class_id, indices in enumerate(self.indices):
                chosen = rng.choice(indices, size=self.states_per_class, replace=True)
                batch.extend((int(index), bool(rng.integers(2))) for index in chosen)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.total_steps


def capture_features(model: UNet, sink: dict[str, torch.Tensor]):
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, ResBlock):
            continue

        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...],
                 output: torch.Tensor, *, key: str = name) -> torch.Tensor:
            sink[key] = output
            return output

        hooks.append(module.register_forward_hook(hook))
    return hooks


def close_hooks(hooks: list[torch.utils.hooks.RemovableHandle]) -> None:
    for hook in hooks:
        hook.remove()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    dataset_cls = ImbalanceCIFAR10 if args.data_type == "cifar10lt" else ImbalanceCIFAR100
    base = dataset_cls(
        root=args.root, imb_type="exp", imb_factor=args.imbalance_factor,
        rand_number=0, train=True, transform=None, target_transform=None, download=True,
    )
    data = np.asarray(base.data)
    targets = np.asarray(base.targets, dtype=np.int64)
    num_classes = 10 if args.data_type == "cifar10lt" else 100
    calibration_ids: list[int] = []
    train_ids: list[int] = []
    for label in range(num_classes):
        ids = np.flatnonzero(targets == label)
        rng = np.random.default_rng(args.seed + 104729 * label)
        permuted = rng.permutation(ids)
        calibration_ids.extend(permuted[:args.calibration_per_class].tolist())
        train_ids.extend(permuted[args.calibration_per_class:].tolist())
    counts = torch.bincount(torch.from_numpy(targets), minlength=num_classes).float()
    return (
        data[train_ids], targets[train_ids],
        data[calibration_ids], targets[calibration_ids], counts,
    )


def load_model(path: Path, *, device: torch.device, num_classes: int) -> UNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["ema_model"] if isinstance(payload, Mapping) and "ema_model" in payload else payload
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain an ema_model state dict")
    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
                 num_res_blocks=2, dropout=.1, cond=True, augm=False,
                 num_class=num_classes).to(device)
    model.load_state_dict({str(k).replace("_orig_mod.", ""): v for k, v in state.items()}, strict=True)
    return model


@torch.no_grad()
def calibrate(
    model: UNet,
    calibration_data: np.ndarray,
    calibration_targets: np.ndarray,
    *,
    tail_labels: tuple[int, ...],
    states_per_class: int,
    seed: int,
    device: torch.device,
    output: Path,
    num_classes: int,
) -> AllocationMask:
    model.eval()
    schedule = torch.cumprod(1.0 - torch.linspace(1e-4, .02, 1000), dim=0)
    rows: list[dict[str, object]] = []
    for label in tail_labels:
        candidate = np.flatnonzero(calibration_targets == label)
        if len(candidate) < states_per_class:
            raise ValueError(f"calibration class {label} has only {len(candidate)} states")
        rng = np.random.default_rng(seed + 300_007 * label)
        selected = rng.choice(candidate, states_per_class, replace=False)
        x0 = torch.from_numpy(calibration_data[selected]).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0).to(device)
        for timestep in TIMESTEP_REPRESENTATIVES:
            noise_generator = torch.Generator(device="cpu").manual_seed(
                seed + 2_000_003 * label + 9_973 * timestep
            )
            noise = torch.randn(x0.shape, generator=noise_generator).to(device)
            rows.extend(one_cell(
                model, x0=x0, label=label, timestep=timestep, noise=noise,
                schedule=schedule,
            ))
    blocks = tuple(row["block"] for row in rows if row["block"].startswith("downblocks."))
    # Keep the native execution order and include middle/up blocks once.
    blocks = tuple(dict.fromkeys(row["block"] for row in rows))
    mask, report = calibrate_allocation_mask(
        rows, block_names=blocks, representatives=TIMESTEP_REPRESENTATIVES,
        boundaries=TIMESTEP_BOUNDARIES,
        utility_threshold=args_global.utility_threshold,
        min_supported_fraction=args_global.min_supported_fraction,
    )
    write_json(output, {
        "schema": "conditional-partial-pooling-allocation-v1",
        "mask": mask.to_dict(),
        "calibration_rows": rows,
        "report": report,
        "tail_labels": list(tail_labels),
        "states_per_class": states_per_class,
        "num_classes": num_classes,
    })
    return mask


def update_ema(model: UNet, ema: UNet, decay: float) -> None:
    with torch.no_grad():
        for name, value in ema.state_dict().items():
            source = model.state_dict()[name]
            if torch.is_floating_point(value):
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-type", choices=("cifar10lt", "cifar100lt"), default="cifar10lt")
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allocation-json", type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--tail-labels", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--imbalance-factor", type=float, default=.01)
    parser.add_argument("--calibration-per-class", type=int, default=8)
    parser.add_argument("--calibration-states-per-class", type=int, default=8)
    parser.add_argument("--utility-threshold", type=float, default=.005)
    parser.add_argument("--min-supported-fraction", type=float, default=2 / 3)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-states-per-class", type=int, default=4)
    parser.add_argument("--vectorized-batch", action="store_true",
                        help="run one conditional/null pair for the whole batch; needs more memory")
    parser.add_argument("--amp", action="store_true",
                        help="use BF16 autocast on CUDA (H100 recommended)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--lambda-share", type=float, default=.1)
    parser.add_argument("--lambda-spread", type=float, default=.002)
    parser.add_argument("--null-weight", type=float, default=.1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


args_global: argparse.Namespace


def main() -> int:
    global args_global
    args_global = parse_args()
    args = args_global
    if args.steps <= 0 or args.calibration_per_class < args.calibration_states_per_class:
        raise ValueError("steps must be positive and calibration pool must fit its probe")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "train.log")],
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    num_classes = 10 if args.data_type == "cifar10lt" else 100
    train_data, train_targets, calibration_data, calibration_targets, counts = load_dataset(args)
    model = load_model(args.checkpoint, device=device, num_classes=num_classes)
    blocks = tuple(name for name, module in model.named_modules() if isinstance(module, ResBlock))
    if args.allocation_json:
        payload = json.loads(args.allocation_json.read_text())
        allocation = AllocationMask.from_dict(payload["mask"])
    else:
        allocation = None
    if args.calibration_only:
        if args.allocation_json is not None:
            raise ValueError("--calibration-only does not accept --allocation-json")
        allocation = calibrate(
            model, calibration_data, calibration_targets,
            tail_labels=tuple(args.tail_labels),
            states_per_class=args.calibration_states_per_class,
            seed=args.seed, device=device, output=args.output_dir / "allocation.json",
            num_classes=num_classes,
        )
        LOGGER.info("calibration complete active_cells=%d/%d", int(allocation.active.sum()), allocation.active.numel())
        return 0
    if allocation is None:
        raise ValueError("training requires --allocation-json from a disjoint calibration run")
    if allocation.block_names != blocks:
        raise ValueError("allocation mask block order does not match native U-Net")

    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dataset = TensorCIFARLT(train_data, train_targets)
    sampler = ExactClassBatchSampler(
        train_targets, num_classes=num_classes,
        states_per_class=args.batch_states_per_class, seed=args.seed, total_steps=args.steps,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    betas = torch.linspace(1e-4, .02, 1000, device=device)
    abar = torch.cumprod(1.0 - betas, dim=0)
    counts_device = counts.to(device)
    class_weights = torch.zeros(num_classes, device=device)
    tail_counts = counts_device[args.tail_labels]
    class_weights[args.tail_labels] = tail_counts.mean() / tail_counts.clamp_min(1.0)
    metrics_path = args.output_dir / "metrics.jsonl"
    start = time.monotonic()
    model.train()
    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        timesteps = torch.randint(1000, (len(images),), device=device)
        noise = torch.randn_like(images)
        x_t = abar[timesteps, None, None, None].sqrt() * images + (
            1.0 - abar[timesteps, None, None, None]
        ).sqrt() * noise
        optimizer.zero_grad(set_to_none=True)
        total_base = torch.zeros((), device=device)
        total_share = torch.zeros((), device=device)
        total_spread = torch.zeros((), device=device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.amp and device.type == "cuda" else nullcontext()
        )
        with amp_context:
            if args.vectorized_batch:
                cond_features: dict[str, torch.Tensor] = {}
                null_features: dict[str, torch.Tensor] = {}
                cond_hooks = capture_features(model, cond_features)
                cond_pred = model(x_t, timesteps, y=labels, augm=None)
                close_hooks(cond_hooks)
                null_hooks = capture_features(model, null_features)
                null_pred = model(x_t, timesteps, y=None, augm=None)
                close_hooks(null_hooks)
                base = ((1.0 - args.null_weight) * F.mse_loss(cond_pred, noise)
                        + args.null_weight * F.mse_loss(null_pred, noise))
                share, spread, _ = partial_pooling_losses(
                    cond_features, null_features, labels=labels, timesteps=timesteps,
                    allocation=allocation, class_weights=class_weights,
                )
                (base + args.lambda_share * share + args.lambda_spread * spread).backward()
                total_base = base.detach()
                total_share = share.detach()
                total_spread = spread.detach()
            else:
                unique_labels = labels.unique(sorted=True)
                for label in unique_labels:
                    selected = labels == label
                    cond_features = {}
                    null_features = {}
                    cond_hooks = capture_features(model, cond_features)
                    cond_pred = model(x_t[selected], timesteps[selected], y=labels[selected], augm=None)
                    close_hooks(cond_hooks)
                    null_hooks = capture_features(model, null_features)
                    null_pred = model(x_t[selected], timesteps[selected], y=None, augm=None)
                    close_hooks(null_hooks)
                    fraction = selected.float().mean()
                    base = ((1.0 - args.null_weight) * F.mse_loss(cond_pred, noise[selected])
                            + args.null_weight * F.mse_loss(null_pred, noise[selected]))
                    share, spread, _ = partial_pooling_losses(
                        cond_features, null_features, labels=labels[selected],
                        timesteps=timesteps[selected], allocation=allocation,
                        class_weights=class_weights,
                    )
                    (fraction * (base + args.lambda_share * share + args.lambda_spread * spread)).backward()
                    total_base = total_base + fraction * base.detach()
                    total_share = total_share + fraction * share.detach()
                    total_spread = total_spread + fraction * spread.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scale = min(1.0, (step + 1) / max(args.warmup_steps, 1))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        optimizer.step()
        update_ema(model, ema, args.ema_decay)
        completed = step + 1
        if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
            record = {
                "step": completed,
                "base_loss": float(total_base.cpu()),
                "share_loss": float(total_share.cpu()),
                "spread_loss": float(total_spread.cpu()),
                "loss": float((total_base + args.lambda_share * total_share + args.lambda_spread * total_spread).cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).cpu()),
                "lr": args.lr * scale,
                "updates_per_second": completed / max(time.monotonic() - start, 1e-6),
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            LOGGER.info(
                "step=%d/%d base=%.5f share=%.5f spread=%.5f grad=%.3f ups=%.3f",
                completed, args.steps, record["base_loss"], record["share_loss"],
                record["spread_loss"], record["grad_norm"], record["updates_per_second"],
            )
        if completed % args.checkpoint_every == 0 or completed == args.steps:
            torch.save({
                "schema": "conditional-partial-pooling-v1",
                "step": completed,
                "model": model.state_dict(),
                "ema_model": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "allocation": allocation.to_dict(),
                "config": vars(args),
            }, args.output_dir / f"ckpt_{completed}.pt")
    write_json(args.output_dir / "summary.json", {
        "schema": "conditional-partial-pooling-v1",
        "status": "completed",
        "steps": args.steps,
        "arm": {"lambda_share": args.lambda_share, "lambda_spread": args.lambda_spread},
        "allocation": allocation.to_dict(),
        "tail_labels": args.tail_labels,
        "base_checkpoint": str(args.checkpoint),
        "elapsed_seconds": time.monotonic() - start,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
