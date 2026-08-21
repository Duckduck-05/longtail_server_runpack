import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # LTX_CBDM_F_IMPORT
from torch.amp import autocast
import wandb
import yaml
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imbdiff_cm.dataset import ImbalanceCIFAR100, ImbalanceCIFAR10, LT_Dataset  # LTX_IMAGENET_LT_DATASET_IMPORT
from imbdiff_cm.diffusion import GaussianDiffusionSampler as OCSampler, extract  # LTX_CBDM_IMPORT
from imbdiff_cm.diffusion import GaussianDiffusionTrainer as OCTrainer
from imbdiff_cm.diffusion_cm import GaussianDiffusionSampler as CMSampler
from imbdiff_cm.diffusion_cm import GaussianDiffusionTrainer as CMTrainer
from imbdiff_cm.model.model import UNet
from imbdiff_cm.model.model_cm import UNet_CM

try:
    from tensorboardX import SummaryWriter
except Exception:
    SummaryWriter = None


# LTX_CBDM_TRAINER
class CBDMTrainer(torch.nn.Module):
    """Released CBDM class-balancing consistency loss on CM's shared U-Net."""
    def __init__(self, model, beta_1, beta_T, T, cfg, class_prob, tau):
        super().__init__()
        self.model, self.T, self.cfg, self.tau = model, T, cfg, tau
        self.register_buffer("class_prob", class_prob.float() / class_prob.float().sum())
        alphas_bar = torch.cumprod(1. - torch.linspace(beta_1, beta_T, T).double(), dim=0)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, y_0, augm=None, uncond_flag_out=False):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        y = None if self.cfg and torch.rand(1, device=x_0.device).item() < 0.1 else y_0
        h = self.model(x_t, t, y=y, augm=augm)
        ddpm = F.mse_loss(h, noise, reduction="none").mean()
        if y is None:
            return ddpm
        y_bal = torch.multinomial(self.class_prob.to(x_0.device), x_0.shape[0], replacement=True)
        h_bal = self.model(x_t, t, y=y_bal, augm=augm)
        weight = (t.float() / self.T * self.tau).view(-1, 1, 1, 1)
        reg = (weight * F.mse_loss(h, h_bal.detach(), reduction="none")).mean()
        com = (weight * F.mse_loss(h.detach(), h_bal, reduction="none")).mean()
        return ddpm + reg + 0.25 * com


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# LTX_CM_RESUME_RNG_HELPERS
def capture_rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_dataset(config):
    dataset_cfg = config["dataset"]
    # LTX_IMAGENET_LT_DATASET_FUNCTION
    if dataset_cfg.get("name") == "imagenet_lt":
        root = Path(dataset_cfg["root"])
        manifest = Path(dataset_cfg.get("manifest", ""))
        if not root.is_dir():
            raise FileNotFoundError(f"ImageNet root does not exist: {root}")
        if not manifest.is_file():
            raise FileNotFoundError(f"ImageNet-LT training manifest does not exist: {manifest}")
        transform = transforms.Compose([
            transforms.Resize(int(dataset_cfg["img_size"])),
            transforms.CenterCrop(int(dataset_cfg["img_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return LT_Dataset(root=str(root), txt=str(manifest), transform=transform)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([dataset_cfg["img_size"], dataset_cfg["img_size"]]),
    ])
    DatasetClass = ImbalanceCIFAR10 if int(dataset_cfg["num_classes"]) == 10 else ImbalanceCIFAR100
    return DatasetClass(
        root=dataset_cfg["root"], imb_type="exp", imb_factor=dataset_cfg["imb_factor"],
        rand_number=dataset_cfg.get("rand_number", 0), train=True, transform=transform,
        download=dataset_cfg.get("download", True),
    )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def warmup_lr(step, warmup):
    return min(step, warmup) / warmup


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key.replace("_orig_mod.", "")].data.copy_(
            target_dict[key.replace("_orig_mod.", "")].data * decay + source_dict[key].data * (1 - decay)
        )


def class_weight(labels):
    counts = torch.Tensor(np.unique(labels, return_counts=True)[1])
    return counts / counts.sum()


def build_model(config):
    model_cfg = config["model"]
    common = dict(
        T=config["diffusion"]["T"],
        ch=model_cfg["ch"],
        ch_mult=model_cfg["ch_mult"],
        attn=model_cfg["attn"],
        num_res_blocks=model_cfg["num_res_blocks"],
        dropout=model_cfg["dropout"],
        cond=model_cfg["conditional"],
        augm=False,
        num_class=int(config["dataset"]["num_classes"]),
    )
    if config["method"] == "cm":
        cm_cfg = config["cm"]
        return UNet_CM(
            **common,
            r=cm_cfg.get("lora_r", 0.0),
            lora_alpha=cm_cfg.get("lora_alpha", 1.0),
            r_ratio=cm_cfg["lora_r_ratio"],
            scaling=cm_cfg["lora_scaling"],
            lora_mode=cm_cfg["lora_mode"],
            lora_part=cm_cfg["lora_part"],
        )
    if config["method"] in {"oc", "cbdm"}:  # LTX_CBDM_MODEL
        return UNet(**common)
    raise ValueError(f"Unsupported method: {config['method']}")


def build_trainer_and_sampler(config, model, dataset, weights, device):
    diff = config["diffusion"]
    train = config["training"]
    transfer = config["transfer"]
    weight_matrix = torch.pow(weights.unsqueeze(1) @ weights.unsqueeze(0), transfer["tr_tau"])
    trainer_kwargs = dict(
        model=model,
        beta_1=diff["beta_1"],
        beta_T=diff["beta_T"],
        T=diff["T"],
        dataset=dataset,
        num_class=config["dataset"]["num_classes"],
        cfg=train["cfg"],
        weight=weights.unsqueeze(0),
        transfer_x0=transfer["transfer_x0"],
        transfer_tr_tau=transfer.get("transfer_tr_tau", False),
        transfer_mode=transfer["transfer_mode"],
        label_weight_tr=weight_matrix,
    )
    sampler_kwargs = dict(
        model=model,
        beta_1=diff["beta_1"],
        beta_T=diff["beta_T"],
        T=diff["T"],
        num_class=config["dataset"]["num_classes"],
        img_size=config["dataset"]["img_size"],
        var_type=diff["var_type"],
    )
    # LTX_CBDM_TRAINER_BRANCH
    if config["method"] == "cm":
        trainer = CMTrainer(**trainer_kwargs, w_con=config["cm"]["w_con"], w_div=config["cm"]["w_div"]).to(device)
        sampler = CMSampler(**sampler_kwargs).to(device)
    elif config["method"] == "cbdm":
        trainer = CBDMTrainer(model, diff["beta_1"], diff["beta_T"], diff["T"], train["cfg"], weights,
                              config.get("cbdm", {}).get("tau", 1.0)).to(device)
        sampler = OCSampler(**sampler_kwargs).to(device)
    else:
        trainer = OCTrainer(**trainer_kwargs).to(device)
        sampler = OCSampler(**sampler_kwargs).to(device)
    return trainer, sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ckpt_step", type=int, default=-1)  # LTX_CM_RESUME_ZERO_STEP
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.total_steps is not None:
        config["training"]["total_steps"] = args.total_steps
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    set_seed(config.get("seed", 0))

    # LTX_CM_EXTERNAL_WANDB_INIT: orchestration owns the W&B run.
    output_dir = Path(config["output_dir"])
    (output_dir / "sample").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.resolved.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    dataset = make_dataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    datalooper = infiniteloop(dataloader)
    weights = class_weight(dataset.targets)
    print(
        f"Dataset CIFAR100-LT contains {len(dataset.targets)} images "
        f"with {len(np.unique(dataset.targets))} classes."
    )

    ckpt_step = args.ckpt_step
    net_model = build_model(config).to(device)
    ema_model = copy.deepcopy(net_model).to(device)
    if ckpt_step >= 0:
        ckpt = torch.load(output_dir / f"ckpt_{ckpt_step}.pt", map_location="cpu")
        net_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["net_model"].items()})
        ema_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["ema_model"].items()})
        print(f"Resumed from ckpt_{ckpt_step}.pt")
    net_model = torch.compile(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=config["training"]["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: warmup_lr(step, config["training"]["warmup"])
    )
    if ckpt_step >= 0:
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])
    trainer, sampler = build_trainer_and_sampler(config, net_model, dataset, weights, device)
    ema_sampler = copy.deepcopy(sampler)
    ema_sampler.model = ema_model

    writer = SummaryWriter(str(output_dir)) if SummaryWriter is not None else NullWriter()
    if ckpt_step >= 0 and "fixed_x_T" in ckpt:
        fixed_x_T = ckpt["fixed_x_T"].to(device)  # LTX_CM_RESUME_FIXED_NOISE
    else:
        fixed_x_T = torch.randn(
            min(config["training"]["sample_size"], 100),
            3,
            config["dataset"]["img_size"],
            config["dataset"]["img_size"],
            device=device,
        )

    if ckpt_step >= 0 and "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])  # LTX_CM_RESUME_RNG_RESTORE

    model_size = sum(param.data.nelement() for param in net_model.parameters())
    print(f"Model params: {model_size / 1024 / 1024:.2f} M")

    total_steps = config["training"]["total_steps"]
    start_step = ckpt_step + 1 if ckpt_step >= 0 else 0  # LTX_CM_RESUME_NEXT_STEP
    with trange(start_step, total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x_0, y_0 = next(datalooper)
            x_0 = x_0.to(device)
            y_0 = y_0.to(device)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = trainer(x_0, y_0, augm=None, uncond_flag_out=False)
                # LTX_OC_LOSS_NORMALIZATION: OC returns two unreduced terms.
                if isinstance(loss, tuple):
                    loss = sum(term.mean() for term in loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), config["training"]["grad_clip"])
            optim.step()
            sched.step()
            ema(net_model, ema_model, config["training"]["ema_decay"])

            writer.add_scalar("loss", loss.item(), step)
            pbar.set_postfix(loss=f"{loss.item():.5f}")
            # LTX_CM_EXTERNAL_WANDB_LOG

            sample_step = config["training"]["sample_step"]
            if sample_step > 0 and step != 0 and step % sample_step == 0:
                ema_model.eval()
                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.bfloat16):
                    samples, _ = ema_sampler(fixed_x_T)
                    grid = (make_grid(samples) + 1) / 2
                    save_image(grid, output_dir / "sample" / f"{step}.png")
                    writer.add_image("sample", grid, step)
                ema_model.train()

            save_step = config["training"]["save_step"]
            if save_step > 0 and step % save_step == 0:
                ckpt = {
                    "net_model": net_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                    "fixed_x_T": fixed_x_T.detach().cpu(),
                    "rng_state": capture_rng_state(),  # LTX_CM_RESUME_RNG_CHECKPOINT
                }
                torch.save(ckpt, output_dir / f"ckpt_{step}.pt")
                # LTX: retain every checkpoint so the training budget stays
                # auditable. Set LTX_KEEP_CHECKPOINTS=0 for upstream behaviour.
                if os.environ.get("LTX_KEEP_CHECKPOINTS", "1") == "0":
                    prev_ckpt = output_dir / f"ckpt_{step - save_step}.pt"
                    if prev_ckpt.exists():
                        prev_ckpt.unlink()

    writer.close()


if __name__ == "__main__":
    main()
