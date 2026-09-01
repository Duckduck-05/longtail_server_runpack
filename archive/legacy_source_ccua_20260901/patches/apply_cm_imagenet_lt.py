#!/usr/bin/env python3
"""Idempotently add the upstream CM generic LT_Dataset training path.

The released CM tree already ships ``LT_Dataset`` but its train entrypoint only
selects CIFAR. This patch exposes that loader for ImageNet-LT manifests and
ports the released CBDM loss so every Table-5 baseline can run on that loader.
"""
from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    if old not in text:
        raise RuntimeError(f"patch anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_all(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"patch anchor not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_cm_imagenet_lt.py THIRD_PARTY/ImbDiff-CM")
    repo = Path(sys.argv[1]).resolve()
    train = repo / "tools" / "train.py"
    replace_once(
        train,
        "from imbdiff_cm.dataset import ImbalanceCIFAR100, ImbalanceCIFAR10\n",
        "from imbdiff_cm.dataset import ImbalanceCIFAR100, ImbalanceCIFAR10, LT_Dataset  # LTX_IMAGENET_LT_DATASET_IMPORT\n",
        "# LTX_IMAGENET_LT_DATASET_IMPORT",
    )
    replace_once(
        train,
        "from imbdiff_cm.diffusion import GaussianDiffusionSampler as OCSampler\n",
        "from imbdiff_cm.diffusion import GaussianDiffusionSampler as OCSampler, extract  # LTX_CBDM_IMPORT\n",
        "# LTX_CBDM_IMPORT",
    )
    replace_once(
        train,
        "import torch\n",
        "import torch\nimport torch.nn.functional as F  # LTX_CBDM_F_IMPORT\n",
        "# LTX_CBDM_F_IMPORT",
    )
    replace_once(
        train,
        "class NullWriter:\n",
        '''# LTX_CBDM_TRAINER
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
''',
        "# LTX_CBDM_TRAINER",
    )
    replace_once(
        train,
        '''def make_dataset(config):
    dataset_cfg = config["dataset"]
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([dataset_cfg["img_size"], dataset_cfg["img_size"]]),
        ]
    )
    DatasetClass = ImbalanceCIFAR10 if int(dataset_cfg["num_classes"]) == 10 else ImbalanceCIFAR100
    return DatasetClass(
        root=dataset_cfg["root"],
        imb_type="exp",
        imb_factor=dataset_cfg["imb_factor"],
        rand_number=dataset_cfg.get("rand_number", 0),
        train=True,
        transform=transform,
        download=dataset_cfg.get("download", True),
    )
''',
        '''def make_dataset(config):
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
''',
        "# LTX_IMAGENET_LT_DATASET_FUNCTION",
    )
    # The source-level W&B run is deliberately disabled: ltx.worker owns one
    # deterministic run per seed and logs all phases/metrics to that run.
    replace_once(
        train,
        '    wandb.init(project="longtail-baselines", name="cm", config=config, resume="allow")\n',
        '    # LTX_CM_EXTERNAL_WANDB_INIT: orchestration owns the W&B run.\n',
        "# LTX_CM_EXTERNAL_WANDB_INIT",
    )
    replace_once(
        train,
        '            wandb.log({"loss": loss.item()}, step=step)\n',
        '            # LTX_CM_EXTERNAL_WANDB_LOG\n',
        "# LTX_CM_EXTERNAL_WANDB_LOG",
    )
    replace_once(
        train,
        '    if config["method"] == "oc":\n',
        '    if config["method"] in {"oc", "cbdm"}:  # LTX_CBDM_MODEL\n',
        "# LTX_CBDM_MODEL",
    )
    replace_once(
        train,
        '''    # LTX_CBDM_TRAINER_BRANCH
    if config["method"] == "cm":
        trainer = CMTrainer(
            **trainer_kwargs,
            w_con=config["cm"]["w_con"],
            w_div=config["cm"]["w_div"],
        ).to(device)
        sampler = CMSampler(**sampler_kwargs).to(device)
    else:
        trainer = OCTrainer(**trainer_kwargs).to(device)
        sampler = OCSampler(**sampler_kwargs).to(device)
''',
        '''    if config["method"] == "cm":
        trainer = CMTrainer(**trainer_kwargs, w_con=config["cm"]["w_con"], w_div=config["cm"]["w_div"]).to(device)
        sampler = CMSampler(**sampler_kwargs).to(device)
    elif config["method"] == "cbdm":
        trainer = CBDMTrainer(model, diff["beta_1"], diff["beta_T"], diff["T"], train["cfg"], weights,
                              config.get("cbdm", {}).get("tau", 1.0)).to(device)
        sampler = OCSampler(**sampler_kwargs).to(device)
    else:
        trainer = OCTrainer(**trainer_kwargs).to(device)
        sampler = OCSampler(**sampler_kwargs).to(device)
''',
        "# LTX_CBDM_TRAINER_BRANCH",
    )
    replace_once(
        train,
        '                loss = trainer(x_0, y_0, augm=None, uncond_flag_out=False)\n',
        '''                loss = trainer(x_0, y_0, augm=None, uncond_flag_out=False)
                # LTX_OC_LOSS_NORMALIZATION: OC returns two unreduced terms.
                if isinstance(loss, tuple):
                    loss = sum(term.mean() for term in loss)
''',
        "# LTX_OC_LOSS_NORMALIZATION",
    )
    # A checkpoint is written after its numbered optimizer update.  Starting
    # the loop at that same number repeats one update after an interruption.
    # The upstream script already restores model/optimizer/scheduler through
    # --ckpt_step; this operational correction resumes at the next update.
    replace_once(
        train,
        '    with trange(ckpt_step, total_steps, dynamic_ncols=True) as pbar:\n',
        '    start_step = ckpt_step + 1 if ckpt_step > 0 else 0  # LTX_CM_RESUME_NEXT_STEP\n'
        '    with trange(start_step, total_steps, dynamic_ncols=True) as pbar:\n',
        "# LTX_CM_RESUME_NEXT_STEP",
    )
    replace_once(
        train,
        '''    fixed_x_T = torch.randn(
        min(config["training"]["sample_size"], 100),
        3,
        config["dataset"]["img_size"],
        config["dataset"]["img_size"],
        device=device,
    )
''',
        '''    if ckpt_step > 0 and "fixed_x_T" in ckpt:
        fixed_x_T = ckpt["fixed_x_T"].to(device)  # LTX_CM_RESUME_FIXED_NOISE
    else:
        fixed_x_T = torch.randn(
            min(config["training"]["sample_size"], 100),
            3,
            config["dataset"]["img_size"],
            config["dataset"]["img_size"],
            device=device,
        )
''',
        "# LTX_CM_RESUME_FIXED_NOISE",
    )
    replace_once(
        train,
        '\n\ndef make_dataset(config):\n',
        '''

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
''',
        "# LTX_CM_RESUME_RNG_HELPERS",
    )
    replace_once(
        train,
        '                    "fixed_x_T": fixed_x_T.detach().cpu(),\n',
        '                    "fixed_x_T": fixed_x_T.detach().cpu(),\n'
        '                    "rng_state": capture_rng_state(),  # LTX_CM_RESUME_RNG_CHECKPOINT\n',
        "# LTX_CM_RESUME_RNG_CHECKPOINT",
    )
    replace_once(
        train,
        '    model_size = sum(param.data.nelement() for param in net_model.parameters())\n',
        '    if ckpt_step > 0 and "rng_state" in ckpt:\n'
        '        restore_rng_state(ckpt["rng_state"])  # LTX_CM_RESUME_RNG_RESTORE\n\n'
        '    model_size = sum(param.data.nelement() for param in net_model.parameters())\n',
        "# LTX_CM_RESUME_RNG_RESTORE",
    )
    # Step zero is a valid saved checkpoint.  Reserve -1 as the no-resume
    # sentinel so an interruption immediately after the first save remains
    # resumable without replaying that optimizer update.
    replace_once(
        train,
        '    parser.add_argument("--ckpt_step", type=int, default=0)\n',
        '    parser.add_argument("--ckpt_step", type=int, default=-1)  # LTX_CM_RESUME_ZERO_STEP\n',
        "# LTX_CM_RESUME_ZERO_STEP",
    )
    replace_all(train, "if ckpt_step > 0", "if ckpt_step >= 0", "# LTX_CM_RESUME_ZERO_STEP_CONDITIONS")
    (repo / ".ltx_cm_imagenet_lt_patch_v1").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
