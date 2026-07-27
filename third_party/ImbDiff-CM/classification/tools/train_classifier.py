import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import datasets, transforms
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imbdiff_classification.dataset import ImbalanceCIFAR100, WrappedDataset
from imbdiff_classification.losses import create_loss
from imbdiff_classification.models import ResNet_s, ResNet_s_CM


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_data(config):
    root = config["dataset"]["root"]
    imb_factor = config["dataset"]["imb_factor"]
    rand_number = config["dataset"].get("rand_number", 0)
    batch_size = config["training"]["batch_size"]
    workers = config["training"]["num_workers"]
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop((32, 32), padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    train_set = ImbalanceCIFAR100(
        root=root,
        imb_type="exp",
        imb_factor=imb_factor,
        rand_number=rand_number,
        train=True,
        transform=train_transform,
        download=config["dataset"].get("download", True),
    )
    test_set = datasets.CIFAR100(
        root=root,
        train=False,
        transform=test_transform,
        download=config["dataset"].get("download", True),
    )
    train_loader = torch.utils.data.DataLoader(
        WrappedDataset(train_set),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        WrappedDataset(test_set),
        batch_size=config["training"].get("test_batch_size", batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, test_loader, train_set.img_num_list


def build_model(config):
    name = config["model"]["type"]
    params = config["model"]["params"]
    if name == "ResNet_s":
        return ResNet_s(**params)
    if name == "ResNet_s_CM":
        return ResNet_s_CM(**params)
    raise ValueError(f"Unsupported classifier model: {name}")


def build_optimizer(model, config):
    opt = config["training"]["optimizer"]
    params = config["training"]["optim_params"]
    if opt == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=params["lr"],
            momentum=params["momentum"],
            weight_decay=params["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {opt}")


def build_scheduler(optimizer, config):
    params = config["training"]["scheduler_params"]
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=params["milestones"],
        gamma=params["gamma"],
    )


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
            preds.append(outputs["output"].detach().cpu())
            labels.append(targets.cpu())
    logits = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    pred_labels = torch.argmax(logits, dim=1)
    acc = (pred_labels == labels).float().mean().item()
    return {"acc": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers

    set_seed(config.get("seed", 0))
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.resolved.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_loader, test_loader, cls_num_list = build_data(config)
    model = build_model(config).to(device)
    criterion = create_loss(config["training"]["loss_type"], config["training"]["loss_params"], cls_num_list).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    best_acc = -1.0
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "lr", "acc"])
        writer.writeheader()
        for epoch in trange(config["training"]["num_epochs"], desc="training"):
            model.train()
            if hasattr(criterion, "set_epoch"):
                criterion.set_epoch(epoch)
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss = loss.item()
                if i % config["logging"]["print_iter"] == 0:
                    print(f"epoch={epoch} iter={i}/{len(train_loader)} loss={running_loss:.6f}")
            scheduler.step()
            metrics = evaluate(model, test_loader, device)
            row = {
                "epoch": epoch,
                "loss": running_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **metrics,
            }
            writer.writerow(row)
            f.flush()
            print(row)
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
            }
            torch.save(checkpoint, output_dir / "checkpoint.pth.tar")
            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                torch.save(checkpoint, output_dir / "model_best.pth.tar")


if __name__ == "__main__":
    main()
