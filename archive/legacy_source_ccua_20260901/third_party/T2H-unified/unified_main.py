"""Common T2H-host training and sampling entry point.

This file is the execution host for the migrated benchmark.  The backbone,
diffusion schedule, data construction, checkpoint format, and sampler come
from OC_LT/T2H.  Method-specific objectives are dispatched by
``unified_objectives``; they are ports of the objectives that used to live in
the other vendored trees.

The host intentionally does not call the old repositories at runtime.  This
is what makes a comparison a common-code experiment.  The old trees remain
available until the running legacy jobs release their paths and a source
manifest is written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataset import ImbalanceCIFAR10, ImbalanceCIFAR100
from diffusion import GaussianDiffusionSamplerOld
from ltx_manifest_dataset import FrozenManifestDataset, ManifestImageDataset
from model.model_cm import UNet_CM
from unified_objectives import UnifiedObjective

try:
    from tensorboardX import SummaryWriter
except Exception:  # pragma: no cover - optional in smoke environments
    SummaryWriter = None


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOST_REVISION = "t2h-unified-common-v2"
CHECKPOINT_SCHEMA = 2
DEFAULT_CHECKPOINT_PREFIX = "ckpt_unified_v2_"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InfiniteLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def __iter__(self):
        while True:
            yield from self.loader


def _transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([img_size, img_size]),
    ])


class ImbalanceImageFolder(ImageFolder):
    """ImageNet-LT ImageFolder with the same exponential split as CIFAR-LT."""

    def __init__(self, root: str, imb_factor: float, rand_number: int,
                 transform=None):
        super().__init__(root=root, transform=transform)
        self.num_classes = len(self.classes)
        rng = np.random.RandomState(rand_number)
        counts = self.get_img_num_per_cls(self.num_classes, imb_factor)
        targets = np.asarray(self.targets, dtype=np.int64)
        selected = []
        for cls, count in enumerate(counts):
            indices = np.flatnonzero(targets == cls)
            rng.shuffle(indices)
            selected.extend(indices[:count].tolist())
        self.samples = [self.samples[i] for i in selected]
        self.imgs = self.samples
        self.targets = [self.targets[i] for i in selected]

    def get_img_num_per_cls(self, cls_num: int, imb_factor: float):
        img_max = len(self.samples) / cls_num
        return [max(1, int(img_max * (imb_factor ** (i / (cls_num - 1.0)))))
                for i in range(cls_num)]


def build_dataset(args):
    transform = _transform(args.img_size)
    if args.frozen_manifest:
        return FrozenManifestDataset(args.frozen_manifest, transform=transform)
    data_type = args.data_type.lower()
    if data_type == "cifar10lt":
        return ImbalanceCIFAR10(
            root=args.root, imb_type="exp", imb_factor=args.imb_factor,
            rand_number=args.split_seed, train=True, transform=transform,
            download=args.download,
        )
    if data_type == "cifar100lt":
        return ImbalanceCIFAR100(
            root=args.root, imb_type="exp", imb_factor=args.imb_factor,
            rand_number=args.split_seed, train=True, transform=transform,
        )
    if data_type == "cifar10":
        return CIFAR10(root=args.root, train=True, transform=transform,
                       download=args.download)
    if data_type == "cifar100":
        return CIFAR100(root=args.root, train=True, transform=transform,
                        download=args.download)
    if data_type in {"imagenet_lt", "imagenet200lt"}:
        if args.train_manifest:
            return ManifestImageDataset(
                root=args.root, manifest=args.train_manifest, transform=transform)
        train_root = Path(args.root)
        if (train_root / "train").is_dir():
            train_root = train_root / "train"
        return ImbalanceImageFolder(
            str(train_root), args.imb_factor, args.split_seed, transform=transform)
    raise ValueError(f"Unsupported data_type={args.data_type!r}")


def dataset_images(dataset) -> torch.Tensor | None:
    """Return raw images in [-1,1] for the IP-SVT auxiliary sampler."""
    raw = getattr(dataset, "data", None)
    if raw is None:
        return None
    raw = np.asarray(raw)
    if raw.ndim != 4 or raw.shape[-1] not in (1, 3, 4):
        return None
    if raw.shape[-1] == 1:
        raw = np.repeat(raw, 3, axis=-1)
    if raw.shape[-1] == 4:
        raw = raw[..., :3]
    return torch.from_numpy(raw).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)


def class_probabilities(targets: Iterable[int], num_class: int) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(list(targets), dtype=torch.long),
                            minlength=num_class).double()
    if (counts <= 0).any():
        missing = torch.flatnonzero(counts <= 0).tolist()
        raise ValueError(f"dataset has no examples for classes {missing}")
    return (counts / counts.sum()).float()


def model_call(model, x, t, y=None, *, return_mid=False, use_cm=False):
    # The feature-bearing methods (CORAL/CCUA) must explicitly request the
    # middle representation.  Relying on the model's constructor default
    # silently turns those methods into a plain DDPM call.
    out = model(x, t, y=y, augm=None, return_mid=return_mid, use_cm=use_cm)
    if return_mid:
        if not isinstance(out, tuple):
            raise RuntimeError("unified T2H model must return (epsilon, middle) when requested")
        return out
    return out[0] if isinstance(out, tuple) else out


def _transfer_target(args, x0, x_t, t, noise, y, class_probs):
    """T2H/OC_LT's sample-similarity target, kept in the common host."""
    probs = class_probs.to(device=x0.device, dtype=torch.float32)
    betas = torch.linspace(args.beta_1, args.beta_T, args.T,
                           dtype=torch.float64, device=x0.device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    alpha = alpha_bar[t].float().view((len(t),) + (1,) * (x0.ndim - 1))
    sigma = (1.0 / alpha_bar[t] - 1.0).float().view((len(t),) + (1,) * (x0.ndim - 1))
    cxt = x0 + sigma.sqrt() * noise
    distances = (cxt.flatten(1)[:, None] - x0.flatten(1)[None, :]).square().sum(-1)
    logits = -distances / (2.0 * sigma.flatten()[:, None]).clamp_min(1e-12)
    sampled = torch.multinomial(torch.softmax(logits, dim=1), 1).squeeze(1)
    identity = torch.arange(len(y), device=x0.device)
    old_prob, new_prob = probs[y], probs[y[sampled]]
    if args.transfer_mode == "t2h":
        allowed = new_prob >= old_prob
    elif args.transfer_mode == "h2t":
        allowed = new_prob <= old_prob
    else:
        allowed = torch.ones_like(y, dtype=torch.bool)
    if args.t2h_cut_time >= 0:
        allowed &= t < args.t2h_cut_time
    selected = torch.where(allowed, sampled, identity)
    target = (x_t - alpha.sqrt() * x0[selected]) / (
        (1.0 - alpha_bar[t]).float().sqrt().view((len(t),) + (1,) * (x0.ndim - 1)))
    return target, {
        "t2h_transfer_fraction": selected.ne(identity).float().mean(),
        "t2h_mean_similarity": torch.softmax(logits, dim=1).gather(1, sampled[:, None]).mean(),
    }


def make_model(args, num_class: int) -> nn.Module:
    objective = args.objective.lower()
    lora_part = args.cm_lora_part if objective == "cm" else []
    # Native CORAL DDPM checkpoints carry the two 128-d projection heads even
    # when their DDPM loss never reads them.  Response-mode smoke continuations
    # retain these otherwise-unused parameters solely so a native import can be
    # strict-key compatible; epsilon forward remains the plain T2H path.
    needs_native_projection = objective == "ipsvt" and args.ipsvt_mode in {"response", "hybrid"}
    return UNet_CM(
        T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
        num_res_blocks=args.num_res_blocks, dropout=args.dropout,
        cond=args.conditional, augm=False, num_class=num_class,
        return_mid=False,
        r=args.cm_lora_r, lora_alpha=args.cm_lora_alpha,
        r_ratio=args.cm_lora_r_ratio, scaling=args.cm_lora_scaling,
        lora_mode=args.cm_lora_mode, lora_part=lora_part,
        coral_projection_dim=(
            args.coral_projection_dim if objective == "coral" or needs_native_projection else None
        ),
    )


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _legacy_import_identity(args) -> dict | None:
    """Return explicit lineage for the narrow native-checkpoint import path.

    This is intentionally not part of ``--resume_checkpoint``.  A native
    checkpoint has no unified-host provenance and is accepted only when the
    caller has opted into this one smoke-only import path.
    """
    if not args.import_checkpoint:
        return None
    if not args.allow_legacy_resume:
        raise ValueError("--import_checkpoint requires --allow_legacy_resume")
    source = Path(args.import_checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"native import checkpoint does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if args.import_checkpoint_sha256 and actual_sha256 != args.import_checkpoint_sha256:
        raise RuntimeError(
            "native import checkpoint SHA-256 mismatch: "
            f"expected {args.import_checkpoint_sha256}, got {actual_sha256}"
        )
    return {
        "kind": "native_weight_import_non_exact",
        "source_path": str(source),
        "source_sha256": actual_sha256,
        "source_step": int(args.import_checkpoint_step),
        "optimizer": "fresh",
        "scheduler": "fresh",
    }


def _strip_state_prefix(state: dict, *, label: str) -> dict:
    """Remove only known wrapper prefixes; reject a lossy key collision."""
    if not isinstance(state, dict):
        raise TypeError(f"native import {label} must be a state-dict mapping")
    normalized = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("_orig_mod.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name in normalized:
            raise RuntimeError(f"native import {label} has colliding key after prefix stripping: {name}")
        normalized[name] = value
    return normalized


def import_native_checkpoint(args, model: nn.Module, ema_model: nn.Module) -> dict:
    """Strictly import native DDPM weights/EMA, never its optimizer lineage."""
    identity = _legacy_import_identity(args)
    if identity is None:
        raise ValueError("import_native_checkpoint called without --import_checkpoint")
    try:
        checkpoint = torch.load(identity["source_path"], map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only=
        checkpoint = torch.load(identity["source_path"], map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("native import checkpoint must be a dictionary")
    try:
        actual_step = int(checkpoint["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("native import checkpoint has no integer step") from exc
    if actual_step != identity["source_step"]:
        raise RuntimeError(
            "native import checkpoint step mismatch: "
            f"flag={identity['source_step']} payload={actual_step}"
        )
    required = ("net_model", "ema_model")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise RuntimeError(f"native import checkpoint is missing {missing}")
    # strict=True is load-bearing.  Response mode builds the native 128-d
    # projection heads precisely so these state dicts are verified rather than
    # silently filtering their four unused tensors.
    for label, target in (("net_model", model), ("ema_model", ema_model)):
        source_state = _strip_state_prefix(checkpoint[label], label=label)
        try:
            target.load_state_dict(source_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"native import {label} is not strict-key compatible with the "
                "T2H response model; no tensors were coerced"
            ) from exc
    return identity


def _checkpoint_provenance(args, num_class: int, *, imported_from: dict | None = None) -> dict:
    """Return the identity that must match before a checkpoint is reused.

    The earlier campaigns mixed native and host checkpoints under the same
    filenames.  A tensor-compatible checkpoint is not necessarily a valid
    scientific continuation: objective, split, seed, or even the CORAL head
    can differ while the U-Net shapes still load.  Keep this manifest small,
    serialisable, and strict so that a wrong lineage fails before ``load_state``.
    """
    objective = args.objective.lower()
    objective_keys = (
        "cb_tau", "coral_weight", "coral_temperature",
        "coral_temperature_scaling", "ccua_al", "ccua_ucl",
        "cm_w_con", "cm_w_div", "cm_lora_r", "cm_lora_alpha",
        "cm_lora_r_ratio", "cm_lora_scaling", "cm_lora_mode",
        "ipsvt_mode", "ipsvt_lambda_aux", "ipsvt_lambda_svt", "ipsvt_K",
        "ipsvt_s", "ipsvt_delta", "ipsvt_every", "ipsvt_batch",
        "transfer_x0", "transfer_mode", "t2h_cut_time",
    )
    # Do not add these fields to legacy full/twin/clean provenance.  Those jobs
    # may resume checkpoints created before response/hybrid modes existed.
    if objective == "ipsvt" and args.ipsvt_mode == "response":
        objective_keys += ("ipsvt_response_variant", "ipsvt_response_eta", "ipsvt_lambda")
    if objective == "ipsvt" and args.ipsvt_mode == "hybrid":
        # Hybrid is an in-graph natural-batch hook; unlike full/twin/clean it
        # has neither an every-N-step gate nor a legacy auxiliary batch.
        objective_keys = tuple(
            key for key in objective_keys if key not in {"ipsvt_every", "ipsvt_batch"}
        )
        objective_keys += ("ipsvt_tau", "ipsvt_hybrid_chunk")
    objective_config = {}
    for key in objective_keys:
        value = getattr(args, key, None)
        if isinstance(value, tuple):
            value = list(value)
        objective_config[key] = value
    effective_lora = list(args.cm_lora_part or []) if objective == "cm" else []
    result = {
        "schema": CHECKPOINT_SCHEMA,
        "host": "T2H-unified",
        "host_revision": HOST_REVISION,
        "objective": objective,
        "data": {
            "data_type": str(args.data_type).lower(),
            "imb_factor": float(args.imb_factor),
            "split_seed": int(args.split_seed),
            "num_class": int(num_class),
            "img_size": int(args.img_size),
        },
        "model": {
            "T": int(args.T),
            "ch": int(args.ch),
            "ch_mult": list(args.ch_mult),
            "attn": list(args.attn),
            "num_res_blocks": int(args.num_res_blocks),
            "dropout": float(args.dropout),
            "conditional": bool(args.conditional),
            "coral_projection_dim": int(args.coral_projection_dim)
            if objective == "coral" or (objective == "ipsvt" and args.ipsvt_mode in {"response", "hybrid"}) else 0,
            "cm_lora_part": effective_lora,
        },
        "training": {
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "warmup": int(args.warmup),
            "grad_clip": float(args.grad_clip),
            "ema_decay": float(args.ema_decay),
            "cfg": bool(args.cfg),
            "amp": bool(args.amp),
        },
        "diffusion": {
            "beta_1": float(args.beta_1),
            "beta_T": float(args.beta_T),
            "var_type": str(args.var_type),
        },
        "objective_config": objective_config,
    }
    imported_from = imported_from if imported_from is not None else _legacy_import_identity(args)
    if imported_from is not None:
        result["initialization"] = imported_from
    return result


def _validate_checkpoint(ckpt: dict, path: Path, args, num_class: int) -> None:
    actual = ckpt.get("provenance") if isinstance(ckpt, dict) else None
    if actual is None:
        raise RuntimeError(
            f"refusing unverified legacy checkpoint {path}; it has no T2H host "
            f"provenance. Start a fresh {HOST_REVISION} run instead."
        )
    expected = _checkpoint_provenance(args, num_class)
    if actual != expected:
        mismatches = []
        for section in ("host_revision", "objective", "data", "model", "training", "diffusion", "objective_config"):
            if actual.get(section) != expected.get(section):
                mismatches.append(section)
        raise RuntimeError(
            f"checkpoint provenance mismatch for {path}: {', '.join(mismatches) or 'unknown fields'}; "
            "do not mix native/old checkpoints with the unified benchmark"
        )


def load_checkpoint(path: Path, *, args=None, num_class: int | None = None):
    ckpt = torch.load(path, map_location="cpu")
    if args is not None:
        if num_class is None:
            raise ValueError("num_class is required when validating a checkpoint")
        _validate_checkpoint(ckpt, path, args, num_class)
        missing = sorted({"net_model", "ema_model", "optim", "sched"} - set(ckpt))
        if missing:
            raise RuntimeError(
                f"verified checkpoint {path} is not full-state; missing {missing}"
            )
    return ckpt


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write checkpoints atomically so a killed job cannot leave a fake file."""
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_objective(args, model, dataset, num_class: int):
    response_mode = args.objective == "ipsvt" and args.ipsvt_mode == "response"
    hybrid_mode = args.objective == "ipsvt" and args.ipsvt_mode == "hybrid"
    # These natural-batch modes neither sample labels nor weight losses by
    # class. Avoid constructing even an unused frequency vector on their paths.
    probs = None if response_mode or hybrid_mode else class_probabilities(dataset.targets, num_class)
    objective_name = "ipsvt" if response_mode or hybrid_mode else ("ddpm" if args.objective == "ipsvt" else args.objective)

    def cm_hook(context):
        x0, x_t, t, noise, y = (context[k] for k in ("x0", "x_t", "t", "noise", "y"))
        y_model = None if args.cfg and torch.rand((), device=x0.device) < 0.1 else y
        h_on = model_call(context["model"], x_t, t, y_model, use_cm=True)
        h_off = model_call(context["model"], x_t, t, y_model, use_cm=False)
        target = noise
        transfer_diag = {}
        if args.transfer_x0:
            target, transfer_diag = _transfer_target(args, x0, x_t, t, noise, y, probs)
        loss_on = F.mse_loss(h_on, target, reduction="none").mean()
        empirical = probs.to(x0.device)
        inverse = (1.0 / empirical)
        inverse = inverse / inverse.sum()
        signed_weight = (empirical[y] * args.cm_w_con
                         - inverse[y] * args.cm_w_div)
        capacity = F.mse_loss(h_off.flatten(1), h_on.flatten(1), reduction="none").mean(1)
        capacity = capacity.mul(signed_weight).mean() * num_class
        diagnostics = {
            "base": loss_on.detach(), "capacity": capacity.detach(),
            **transfer_diag,
        }
        return loss_on + capacity, diagnostics

    kwargs = dict(
        T=args.T, beta_1=args.beta_1, beta_T=args.beta_T,
        num_classes=None if response_mode or hybrid_mode else num_class, class_probs=probs,
        cfg_dropout=0.1 if args.cfg else 0.0,
        cbdm_tau=args.cb_tau, t2h_mode=args.transfer_mode,
        t2h_cut_time=args.t2h_cut_time,
        coral_weight=args.coral_weight,
        coral_temperature=args.coral_temperature,
        coral_time_scale=args.coral_temperature_scaling,
        ccua_alignment_weight=args.ccua_al,
        ccua_ucl_weight=args.ccua_ucl,
    )
    if args.objective == "cm":
        kwargs["cm_hook"] = cm_hook
    if response_mode:
        if args.amp:
            raise ValueError("IP-SVT response mode requires --no-amp")
        from ipsvt_response import IPSVTResponseAuxiliary
        kwargs["ipsvt_hook"] = IPSVTResponseAuxiliary(
            T=args.T,
            beta_1=args.beta_1,
            beta_T=args.beta_T,
            eta_std=args.ipsvt_response_eta,
            lambda_weight=args.ipsvt_lambda,
            variant=args.ipsvt_response_variant,
        )
    if hybrid_mode:
        if args.amp:
            raise ValueError("IP-SVT hybrid mode requires --no-amp")
        from ipsvt_hybrid import IPSVTHybridAuxiliary
        from ipsvt_response import forward_with_condition
        kwargs["ipsvt_hook"] = IPSVTHybridAuxiliary(
            T=args.T,
            beta_1=args.beta_1,
            beta_T=args.beta_T,
            K=args.ipsvt_K,
            s=args.ipsvt_s,
            delta=args.ipsvt_delta,
            tau=args.ipsvt_tau,
            lambda_aux=args.ipsvt_lambda_aux,
            lambda_svt=args.ipsvt_lambda_svt,
            chunk_size=args.ipsvt_hybrid_chunk,
            conditioned_forward=forward_with_condition,
        )
    return UnifiedObjective(objective_name, **kwargs).to(DEVICE)


def maybe_make_ipsvt(args, dataset, num_class: int):
    if not args.ipsvt:
        return None
    if args.ipsvt_mode in {"response", "hybrid"}:
        # Both in-graph modes consume x0/y from the natural loader. They must
        # never instantiate IPSVTAuxiliary's class-uniform image pool.
        return None
    if args.amp:
        raise ValueError("IP-SVT auxiliary branch requires --no-amp in the common host")
    images = dataset_images(dataset)
    if images is None:
        raise ValueError("IP-SVT currently requires an in-memory image dataset")
    from ipsvt_aux import IPSVTAuxiliary
    return IPSVTAuxiliary(
        images=images, targets=dataset.targets, num_class=num_class, T=args.T,
        beta_1=args.beta_1, beta_T=args.beta_T, K=args.ipsvt_K,
        s=args.ipsvt_s, delta=args.ipsvt_delta, batch_size=args.ipsvt_batch,
        lambda_svt=args.ipsvt_lambda_svt, lambda_aux=args.ipsvt_lambda_aux,
        every=args.ipsvt_every, mode=args.ipsvt_mode, device=DEVICE,
        seed=args.seed,
    )


def _autocast(args):
    if not args.amp or DEVICE.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def train(args) -> None:
    set_seed(args.seed)
    run_dir = Path(args.logdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sample").mkdir(exist_ok=True)
    dataset = build_dataset(args)
    num_class = args.num_class or int(max(dataset.targets)) + 1
    if num_class != int(max(dataset.targets)) + 1:
        raise ValueError("num_class does not cover all dataset labels")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=DEVICE.type == "cuda",
    )
    data_iter = iter(InfiniteLoader(loader))
    model = make_model(args, num_class).to(DEVICE)
    ema_model = copy.deepcopy(model).to(DEVICE)
    objective = build_objective(args, model, dataset, num_class)
    ipsvt = maybe_make_ipsvt(args, dataset, num_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(step, args.warmup) / max(args.warmup, 1))
    start = 0
    imported_from = None
    if args.resume_checkpoint:
        ckpt = load_checkpoint(Path(args.resume_checkpoint), args=args, num_class=num_class)
        completed_step = int(ckpt.get("step", args.resume_step))
        if args.resume_step >= 0 and completed_step != args.resume_step:
            raise RuntimeError(
                f"resume_step={args.resume_step} does not match checkpoint payload step={completed_step}"
            )
        if completed_step < 0:
            raise RuntimeError(f"checkpoint {args.resume_checkpoint} has no completed step")
        model.load_state_dict(ckpt["net_model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        if "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
        if "sched" in ckpt:
            scheduler.load_state_dict(ckpt["sched"])
        start = completed_step + 1
    elif args.import_checkpoint:
        imported_from = import_native_checkpoint(args, model, ema_model)
        # The native loop names checkpoints by the completed zero-indexed
        # update.  Begin at the next update, so 200000 -> 220000 is exactly
        # 20,000 continuation updates rather than 20,001.
        start = int(imported_from["source_step"]) + 1
    elif args.ckpt_step > 0:
        path = run_dir / f"{args.checkpoint_prefix}{args.ckpt_step}.pt"
        ckpt = load_checkpoint(path, args=args, num_class=num_class)
        if int(ckpt.get("step", -1)) != args.ckpt_step:
            raise RuntimeError(
                f"checkpoint filename step={args.ckpt_step} does not match payload step={ckpt.get('step')}"
            )
        model.load_state_dict(ckpt["net_model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        if "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
        if "sched" in ckpt:
            scheduler.load_state_dict(ckpt["sched"])
        start = args.ckpt_step + 1

    provenance = _checkpoint_provenance(args, num_class, imported_from=imported_from)
    manifest_imported_from = imported_from or provenance.get("initialization")
    save_json(run_dir / "unified_host.json", {
        "host": "T2H-unified",
        "source": "OC_LT/T2H",
        "host_revision": HOST_REVISION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "objective": args.objective,
        "seed": args.seed,
        "data_type": args.data_type,
        "num_class": num_class,
        "total_steps_bound": args.total_steps,
        "device": str(DEVICE),
        "provenance": provenance,
        "imported_from": manifest_imported_from,
    })
    (run_dir / "flagfile.txt").write_text("\n".join(
        f"{k}={v}" for k, v in sorted(vars(args).items())) + "\n", encoding="utf-8")
    writer = SummaryWriter(str(run_dir)) if SummaryWriter else None
    fixed_x_T = torch.randn(min(args.sample_size, 100), 3, args.img_size,
                            args.img_size, device=DEVICE)
    if args.resume_checkpoint or args.ckpt_step > 0:
        if "fixed_x_T" in ckpt:
            fixed_x_T = ckpt["fixed_x_T"].to(DEVICE)

    model.train()
    for step in range(start, args.total_steps):
        optimizer.zero_grad(set_to_none=True)
        x0, y0 = next(data_iter)
        x0, y0 = x0.to(DEVICE, non_blocking=True), y0.to(DEVICE, non_blocking=True)
        with _autocast(args):
            loss, metrics = objective(model, x0, y0, step=step)
        if ipsvt is not None:
            aux = ipsvt(model, step)
            if aux is not None:
                twin, svt, aux_metrics = aux
                loss = loss + args.ipsvt_lambda_aux * (
                    twin + args.ipsvt_lambda_svt * svt)
                metrics.update(aux_metrics)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for source, target in zip(model.parameters(), ema_model.parameters()):
                target.mul_(args.ema_decay).add_(source, alpha=1.0 - args.ema_decay)
        if writer and (step % args.log_step == 0):
            writer.add_scalar("loss", float(loss.detach()), step)
            for key, value in metrics.items():
                writer.add_scalar(key, float(value), step)
        if step % args.log_step == 0:
            shown = {"loss": float(loss.detach()), **{k: float(v) for k, v in metrics.items()}}
            print(f"step={step} " + " ".join(f"{k}={v:.6f}" for k, v in shown.items()), flush=True)
        if args.sample_step > 0 and step > 0 and step % args.sample_step == 0:
            sample = sample_batch(ema_model, args, num_class, fixed_x_T)
            save_image((sample[: min(len(sample), 64)]), run_dir / "sample" / f"{step}.png",
                       nrow=8)
        if args.save_step > 0 and step % args.save_step == 0:
            _atomic_torch_save({
                "net_model": model.state_dict(), "ema_model": ema_model.state_dict(),
                "optim": optimizer.state_dict(), "sched": scheduler.state_dict(),
                "fixed_x_T": fixed_x_T.detach().cpu(), "step": step,
                "host": "T2H-unified", "objective": args.objective,
                "provenance": provenance,
            }, run_dir / f"{args.checkpoint_prefix}{step}.pt")
    if writer:
        writer.close()


@torch.no_grad()
def sample_batch(model, args, num_class: int, noise: torch.Tensor | None = None):
    model.eval()
    sampler = GaussianDiffusionSamplerOld(
        model, args.beta_1, args.beta_T, args.T, img_size=args.img_size,
        var_type=args.var_type, w=args.omega, cond=args.conditional,
    ).to(DEVICE)
    batch = noise if noise is not None else torch.randn(
        args.sample_batch_size, 3, args.img_size, args.img_size, device=DEVICE)
    labels = torch.arange(len(batch), device=DEVICE) % num_class
    method = "ddim" if args.sample_method in {"ddim", "cfg"} else "ddpm"
    return (sampler(batch, labels, method=method, skip=args.ddim_skip_step) + 1.0) / 2.0


@torch.no_grad()
def sample(args) -> None:
    set_seed(args.seed)
    run_dir = Path(args.logdir)
    dataset = build_dataset(args)
    num_class = args.num_class or int(max(dataset.targets)) + 1
    model = make_model(args, num_class).to(DEVICE)
    ckpt = load_checkpoint(
        run_dir / f"{args.checkpoint_prefix}{args.ckpt_step}.pt",
        args=args, num_class=num_class,
    )
    if int(ckpt.get("step", -1)) != args.ckpt_step:
        raise RuntimeError(
            f"checkpoint filename step={args.ckpt_step} does not match payload step={ckpt.get('step')}"
        )
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    sampler = GaussianDiffusionSamplerOld(
        model, args.beta_1, args.beta_T, args.T, img_size=args.img_size,
        var_type=args.var_type, w=args.omega, cond=args.conditional,
    ).to(DEVICE)
    images, labels = [], []
    for start in tqdm(range(0, args.num_images, args.sample_batch_size), desc="T2H-unified sampling"):
        n = min(args.sample_batch_size, args.num_images - start)
        x_T = torch.randn(n, 3, args.img_size, args.img_size, device=DEVICE)
        if args.sample_method == "uncond":
            y = None
        elif args.uniform_labels:
            y = torch.arange(start, start + n, device=DEVICE) % num_class
        else:
            y = torch.randint(num_class, (n,), device=DEVICE)
        method = "ddim" if args.sample_method in {"ddim", "cfg"} else "ddpm"
        out = sampler(x_T, y, method=method, skip=args.ddim_skip_step)
        images.append(((out + 1.0) / 2.0).clamp(0, 1).cpu())
        if y is not None:
            labels.append(y.cpu())
    image_array = torch.cat(images).numpy().astype(np.float32)
    label_array = torch.cat(labels).numpy().astype(np.int64) if labels else None
    if not args.sample_output:
        raise ValueError("--sample_output is required for the common host")
    output = Path(args.sample_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, image_array)
    if label_array is not None:
        np.save(str(output) + ".labels.npy", label_array)
    save_image(torch.from_numpy(image_array[: min(len(image_array), 64)]),
               output.parent / f"visual_{output.stem}.png", nrow=8)
    save_json(output.parent / f"{output.stem}.provenance.json", {
        "host": "T2H-unified", "objective": args.objective,
        "host_revision": HOST_REVISION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_step": args.ckpt_step, "num_images": args.num_images,
        "sample_method": args.sample_method, "sampler_method": method,
        "ddim_skip_step": args.ddim_skip_step, "omega": args.omega,
        "uniform_labels": args.uniform_labels, "seed": args.seed,
        "artifact_namespace": args.artifact_namespace,
        "T": args.T, "beta_1": args.beta_1, "beta_T": args.beta_T,
        "var_type": args.var_type, "img_size": args.img_size,
        "num_class": num_class,
    })


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--sample", action="store_true")
    p.add_argument("--objective", choices=["ddpm", "t2h", "cbdm", "coral", "ccua", "cm", "ipsvt"], default="ddpm")
    p.add_argument("--data_type", default="cifar100lt")
    p.add_argument("--root", default="./data")
    p.add_argument("--frozen_manifest", default="")
    p.add_argument("--train_manifest", default="")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--imb_factor", type=float, default=0.01)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--num_class", type=int, default=0)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--logdir", required=True)
    p.add_argument("--checkpoint_prefix", default=DEFAULT_CHECKPOINT_PREFIX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_step", type=int, default=0)
    p.add_argument("--resume_checkpoint", default="")
    p.add_argument("--resume_step", type=int, default=-1)
    # Deliberately separate from --resume_checkpoint.  This is a narrow,
    # auditable weight import for smoke continuations from a native DDPM
    # checkpoint, not a provenance bypass for normal T2H resume.
    p.add_argument("--import_checkpoint", default="")
    p.add_argument("--import_checkpoint_step", type=int, default=-1)
    p.add_argument("--import_checkpoint_sha256", default="")
    p.add_argument("--allow_legacy_resume", action="store_true")
    p.add_argument("--total_steps", type=int, default=300001)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sample_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_1", type=float, default=1e-4)
    p.add_argument("--beta_T", type=float, default=0.02)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--save_step", type=int, default=50000)
    p.add_argument("--sample_step", type=int, default=100000)
    p.add_argument("--log_step", type=int, default=100)
    p.add_argument("--sample_size", type=int, default=64)
    p.add_argument("--ch", type=int, default=128)
    p.add_argument("--ch_mult", type=int, action="append", default=None)
    p.add_argument("--attn", type=int, action="append", default=None)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--conditional", action="store_true")
    p.add_argument("--cfg", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--var_type", choices=["fixedlarge", "fixedsmall"], default="fixedlarge")
    p.add_argument("--sample_method", choices=["cfg", "ddim", "ddpm", "uncond"], default="ddim")
    p.add_argument("--ddim_skip_step", type=int, default=10)
    p.add_argument("--omega", type=float, default=1.5)
    p.add_argument("--num_images", type=int, default=50000)
    p.add_argument("--sample_output", default="")
    p.add_argument("--artifact_namespace", default="")
    p.add_argument("--uniform_labels", action="store_true")
    p.add_argument("--transfer_x0", action="store_true")
    p.add_argument("--transfer_mode", choices=["full", "t2h", "h2t"], default="t2h")
    p.add_argument("--t2h_cut_time", type=int, default=-1)
    p.add_argument("--cbdm", action="store_true")
    p.add_argument("--cb_tau", type=float, default=1.0)
    p.add_argument("--coral_weight", type=float, default=0.01)
    p.add_argument("--coral_temperature", type=float, default=0.09)
    p.add_argument("--coral_temperature_scaling", type=float, default=1.0)
    p.add_argument("--coral_projection_dim", type=int, default=128)
    p.add_argument("--ccua_al", type=float, default=0.0)
    p.add_argument("--ccua_ucl", type=float, default=0.0)
    p.add_argument("--cm_w_con", type=float, default=1.0)
    p.add_argument("--cm_w_div", type=float, default=0.2)
    p.add_argument("--cm_lora_r", type=int, default=0)
    p.add_argument("--cm_lora_alpha", type=float, default=1.0)
    p.add_argument("--cm_lora_r_ratio", type=float, default=0.1)
    p.add_argument("--cm_lora_scaling", type=float, default=0.5)
    p.add_argument("--cm_lora_mode", choices=["value", "ratio"], default="ratio")
    # ``append`` must start from ``None``.  An argparse default list is kept
    # and then appended to, so using ``default=["up"]`` would turn the
    # explicit adapter flag into ``["up", "up"]`` and record a misleading
    # model signature in the checkpoint provenance.
    p.add_argument("--cm_lora_part", action="append", default=None)
    p.add_argument("--ipsvt", action="store_true")
    p.add_argument("--ipsvt_mode", choices=["full", "twin", "clean", "response", "hybrid"], default="full")
    p.add_argument("--ipsvt_lambda_aux", type=float, default=1.0)
    p.add_argument("--ipsvt_lambda_svt", type=float, default=1.0)
    p.add_argument("--ipsvt_K", type=int, default=4)
    p.add_argument("--ipsvt_s", type=float, default=0.05)
    p.add_argument("--ipsvt_delta", type=float, default=0.1)
    p.add_argument("--ipsvt_every", type=int, default=4)
    p.add_argument("--ipsvt_batch", type=int, default=16)
    p.add_argument("--ipsvt_tau", type=float, default=1e-6)
    p.add_argument("--ipsvt_hybrid_chunk", type=int, default=16)
    # Response mode has one regularization weight for Twin+SVT.  Its K=1
    # response construction is fixed in code, not exposed as another knob.
    p.add_argument("--ipsvt_response_variant", choices=["twin", "full"], default="full")
    p.add_argument("--ipsvt_response_eta", type=float, default=0.05)
    p.add_argument("--ipsvt_lambda", type=float, default=1.0)
    args = p.parse_args()
    args.ch_mult = args.ch_mult or [1, 2, 2, 2]
    args.attn = args.attn or [1]
    args.cm_lora_part = args.cm_lora_part or ["up"]
    if args.objective == "t2h":
        args.transfer_x0 = True
        args.transfer_mode = "t2h"
    if args.objective == "cm":
        args.transfer_x0 = True
    if args.objective == "ipsvt":
        args.ipsvt = True
    if args.objective == "cbdm":
        args.cbdm = True
    if args.objective == "ccua" and args.ccua_al == 0 and args.ccua_ucl == 0:
        args.ccua_al = args.ccua_ucl = 1.0
    if args.conditional:
        args.cfg = bool(args.cfg)
    if args.import_checkpoint and not args.allow_legacy_resume:
        p.error("--import_checkpoint requires explicit --allow_legacy_resume")
    if args.allow_legacy_resume and not args.import_checkpoint:
        p.error("--allow_legacy_resume is valid only with --import_checkpoint")
    if args.import_checkpoint and args.import_checkpoint_step < 0:
        p.error("--import_checkpoint requires --import_checkpoint_step >= 0")
    if args.ipsvt_mode in {"response", "hybrid"} and args.objective != "ipsvt":
        p.error("--ipsvt_mode=response/hybrid requires --objective=ipsvt")
    if args.ipsvt_mode in {"response", "hybrid"} and not args.conditional:
        p.error("IP-SVT response/hybrid modes require --conditional")
    if args.import_checkpoint and not (args.objective == "ipsvt" and args.ipsvt_mode in {"response", "hybrid"}):
        p.error("--import_checkpoint is restricted to IP-SVT response/hybrid smoke continuations")
    return args


if __name__ == "__main__":
    parsed = parser()
    if parsed.train:
        train(parsed)
    else:
        sample(parsed)
