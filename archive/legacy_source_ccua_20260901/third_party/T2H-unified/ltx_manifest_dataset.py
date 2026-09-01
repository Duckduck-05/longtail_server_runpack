from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FrozenManifestDataset(Dataset):
    """Immutable training dataset in exact, audited sample order.

    Required NPZ arrays:
      images: uint8 [N,H,W,C]
      train_labels: int64 [N]

    Optional arrays such as fine_labels remain inside the NPZ for the external
    evaluator. They are deliberately not attached to the training Dataset object.
    """

    def __init__(self, path: str, transform=None, target_transform=None):
        self.path = str(Path(path).expanduser().resolve())
        with np.load(self.path, allow_pickle=False) as payload:
            if "images" not in payload or "train_labels" not in payload:
                raise ValueError("Frozen manifest requires images and train_labels arrays")
            self.data = np.asarray(payload["images"])
            labels = np.asarray(payload["train_labels"], dtype=np.int64)
            self.sample_ids = (
                np.asarray(payload["sample_ids"]).astype(str)
                if "sample_ids" in payload
                else np.arange(len(self.data)).astype(str)
            )
        if self.data.dtype != np.uint8 or self.data.ndim != 4 or self.data.shape[-1] not in (1, 3, 4):
            raise ValueError(f"images must be uint8 [N,H,W,C], got {self.data.dtype} {self.data.shape}")
        if labels.ndim != 1 or len(self.data) != len(labels):
            raise ValueError("images/train_labels length mismatch")
        unique = np.unique(labels)
        if not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError(f"train labels must be contiguous 0..C-1, got {unique.tolist()}")
        if len(self.sample_ids) != len(labels):
            raise ValueError("sample_ids length mismatch")
        self.targets = labels.tolist()
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = int(len(unique))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        image = Image.fromarray(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class ManifestImageDataset(Dataset):
    """ImageFolder-compatible dataset backed by a pinned ``path label`` list.

    ImageNet-LT is already long-tailed in its published training manifest, so
    the common host must not apply a second synthetic imbalance transform.
    Keeping the manifest as the source of truth also avoids depending on the
    filesystem's class-directory ordering.
    """

    def __init__(self, root: str, manifest: str, transform=None,
                 target_transform=None):
        self.root = Path(root).expanduser().resolve()
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"ImageNet-LT image root missing: {self.root}")
        if not self.manifest.is_file():
            raise FileNotFoundError(f"ImageNet-LT train manifest missing: {self.manifest}")
        self.samples: list[tuple[Path, int]] = []
        for lineno, raw in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            fields = raw.split()
            if len(fields) != 2:
                raise ValueError(f"{self.manifest}:{lineno}: expected '<relative_image> <label>'")
            relative, label_raw = fields
            try:
                label = int(label_raw)
            except ValueError as exc:
                raise ValueError(f"{self.manifest}:{lineno}: invalid label {label_raw!r}") from exc
            path = Path(relative)
            path = path if path.is_absolute() else self.root / path
            if not path.is_file():
                raise FileNotFoundError(f"{self.manifest}:{lineno}: image missing: {path}")
            self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f"ImageNet-LT train manifest is empty: {self.manifest}")
        self.imgs = self.samples
        self.targets = [label for _, label in self.samples]
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = max(self.targets) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
