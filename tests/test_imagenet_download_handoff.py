import hashlib
import os
import subprocess
import tarfile
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_imagenet_handoff_downloads_verified_payload(tmp_path: Path):
    """A recipient can obtain the three payloads with only .env.local values."""
    payload = tmp_path / "payload"
    image = payload / "class0" / "example.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-an-image")
    archive = tmp_path / "imagenet-lt.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname=".")
    train = tmp_path / "ImageNet_LT_train.txt"
    reference = tmp_path / "ImageNet_LT_balanced_val.txt"
    train.write_text("class0/example.jpg 0\n", encoding="utf-8")
    reference.write_text("class0/example.jpg 0\n", encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "receiver-data"
    env = os.environ.copy()
    env.update(
        {
            "LTX_DATA_ROOT": str(data_root),
            "LTX_IMAGENET_ARCHIVE_URL": archive.as_uri(),
            "LTX_IMAGENET_ARCHIVE_SHA256": _sha256(archive),
            "LTX_IMAGENET_ARCHIVE_FORMAT": "tar",
            "LTX_IMAGENET_LT_TRAIN_MANIFEST_URL": train.as_uri(),
            "LTX_IMAGENET_LT_TRAIN_MANIFEST_SHA256": _sha256(train),
            "LTX_IMAGENET_LT_REFERENCE_MANIFEST_URL": reference.as_uri(),
            "LTX_IMAGENET_LT_REFERENCE_MANIFEST_SHA256": _sha256(reference),
        }
    )
    subprocess.run(
        ["bash", "-c", 'source "$1"', "bash", str(root / "scripts/prepare_imagenet_lt.sh")],
        check=True,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (data_root / "imagenet_lt/images/class0/example.jpg").read_bytes() == b"not-an-image"
    assert (data_root / "imagenet_lt/manifests/ImageNet_LT_train.txt").is_file()
    assert (data_root / "imagenet_lt/manifests/ImageNet_LT_balanced_val.txt").is_file()
