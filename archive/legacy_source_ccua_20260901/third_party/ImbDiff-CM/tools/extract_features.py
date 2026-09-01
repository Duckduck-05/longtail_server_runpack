import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imbdiff_cm.score.inception import InceptionV3


def image_batches(path, batch_size, num_workers, num=-1):
    transform = transforms.ToTensor()
    paths = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                paths.append(os.path.join(root, name))
    paths = sorted(paths)
    if num != -1:
        paths = paths[:num]

    def load_image(img_path):
        return transform(Image.open(img_path).convert("RGB"))

    batch = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for tensor in executor.map(load_image, paths):
            batch.append(tensor)
            if len(batch) == batch_size:
                yield torch.stack(batch)
                batch = []
    if batch:
        yield torch.stack(batch)


def save_features(model, batches, total, output_prefix, device):
    features_2048 = np.empty((total, 2048))
    start = 0
    for batch in tqdm(batches, desc=f"extracting {output_prefix.name}"):
        end = start + batch.size(0)
        with torch.no_grad():
            feat_2048 = model(batch.to(device))[0].view(batch.size(0), -1)
        features_2048[start:end] = feat_2048.cpu().numpy()
        start = end
    np.save(f"{output_prefix}_2048.npy", features_2048)


def extract_generated(args, model, device):
    image_root = Path(args.image_dir)
    feature_dir = Path(args.feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [
        p for p in image_root.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    ]
    if not image_paths:
        raise ValueError(f"No generated images found under {image_root}.")
    batches = image_batches(image_root, args.batch_size, args.num_workers, args.num_images)
    total = len(image_paths) if args.num_images == -1 else min(len(image_paths), args.num_images)
    save_features(model, batches, total, feature_dir / args.name, device)


def extract_real(args, model, device):
    feature_dir = Path(args.feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    dataset = CIFAR100(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    batches = (batch for batch, _ in loader)
    save_features(model, batches, len(dataset), feature_dir / "real_cifar100_train", device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generated", "real"], required=True)
    parser.add_argument("--name", default=None, help="Feature prefix, e.g. OC-cifar100-100 or CM-cifar100-100.")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_images", type=int, default=-1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.mode == "generated" and (not args.name or not args.image_dir):
        raise ValueError("--name and --image_dir are required for generated feature extraction.")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = InceptionV3([3]).to(device)
    model.eval()

    if args.mode == "real":
        extract_real(args, model, device)
    else:
        extract_generated(args, model, device)


if __name__ == "__main__":
    main()
