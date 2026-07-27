import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imbdiff_cm.diffusion import GaussianDiffusionSamplerOld as OCSampler
from imbdiff_cm.diffusion_cm import GaussianDiffusionSamplerOld as CMSampler
from imbdiff_cm.model.model import UNet
from imbdiff_cm.model.model_cm import UNet_CM


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    return UNet(**common)


def build_sampler(config, model, device):
    sampler_cls = CMSampler if config["method"] == "cm" else OCSampler
    return sampler_cls(
        model,
        config["diffusion"]["beta_1"],
        config["diffusion"]["beta_T"],
        config["diffusion"]["T"],
        img_size=config["dataset"]["img_size"],
        var_type=config["diffusion"]["var_type"],
        w=config["evaluation"]["omega"],
        cond=config["model"]["conditional"],
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = int(config["dataset"]["num_classes"])
    output_dir = Path(args.output_dir or config["evaluation"]["image_dir"])
    num_images = args.num_images or config["evaluation"]["num_images"]
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    num_per_class = num_images // num_classes
    if num_per_class == 0:
        raise ValueError(f"num_images must be at least num_classes ({num_classes}) for balanced sampling.")
    num_images = num_per_class * num_classes

    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    sampler = build_sampler(config, model, device)

    with torch.no_grad():
        for start in trange(0, num_images, batch_size, desc="sampling images"):
            current = min(batch_size, num_images - start)
            labels = [(start + j) // num_per_class for j in range(current)]
            y = torch.tensor(labels, device=device)
            x_t = torch.randn(current, 3, config["dataset"]["img_size"], config["dataset"]["img_size"], device=device)
            images = sampler(x_t, y, method=config["evaluation"]["sample_method"], skip=config["evaluation"]["ddim_skip_step"]).cpu()
            images = (images + 1) / 2
            with ThreadPoolExecutor() as executor:
                futures = []
                for idx, image in enumerate(images):
                    label = labels[idx]
                    path = output_dir / f"{start + idx:06d}_class_{label}.png"
                    futures.append(executor.submit(save_image, image, path))
                for future in futures:
                    future.result()


if __name__ == "__main__":
    main()
