from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from ltx.adapters.t2h_unified import T2HUnifiedAdapter
from ltx.config import load_campaign


ROOT = Path(__file__).resolve().parents[1]


def test_secondary_imagenet_campaign_has_only_the_requested_two_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("LTX_IMAGENET_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("LTX_IMAGENET_LT_TRAIN_MANIFEST", str(tmp_path / "ImageNet_LT_train.txt"))
    monkeypatch.setenv("LTX_IMAGENET_LT_REFERENCE_MANIFEST", str(tmp_path / "ImageNet_LT_val.txt"))

    campaign = load_campaign(ROOT / "configs/secondary_imagenet_lt.yaml")
    assert len(campaign.tasks) == 2
    assert {(task.method, task.seed) for task in campaign.tasks} == {("ddpm", 0), ("ccua", 0)}
    assert {task.method: task.adapter for task in campaign.tasks} == {"ddpm": "t2h_unified", "ccua": "t2h_unified"}
    assert all(task.dataset["data_type"] == "imagenet_lt" for task in campaign.tasks)
    assert all(task.dataset["img_size"] == 64 and task.dataset["num_classes"] == 1000 for task in campaign.tasks)
    assert all(task.train["total_steps"] == 300000 and task.train["batch_size"] == 256 for task in campaign.tasks)
    assert all(task.eval["checkpoint_step"] == 300000 for task in campaign.tasks)


def test_secondary_imagenet_adapter_uses_manifest_and_only_dispatches_objective(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX_IMAGENET_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("LTX_IMAGENET_LT_TRAIN_MANIFEST", str(tmp_path / "ImageNet_LT_train.txt"))
    monkeypatch.setenv("LTX_IMAGENET_LT_REFERENCE_MANIFEST", str(tmp_path / "ImageNet_LT_val.txt"))
    campaign = load_campaign(ROOT / "configs/secondary_imagenet_lt.yaml")

    for method in ("ddpm", "ccua"):
        task = next(task for task in campaign.tasks if task.method == method)
        task = task.__class__(**{**task.to_dict(), "run_dir": str(tmp_path / method)})
        phases = T2HUnifiedAdapter(ROOT).phases(task)
        assert [phase.name for phase in phases] == ["train", "sample", "metrics"]
        train = " ".join(str(value) for value in phases[0].command)
        sample = " ".join(str(value) for value in phases[1].command)
        metrics = " ".join(str(value) for value in phases[2].command)
        assert "--data_type=imagenet_lt" in train
        assert f"--train_manifest={tmp_path / 'ImageNet_LT_train.txt'}" in train
        assert "--num_class=1000" in train and "--img_size=64" in train
        assert "--total_steps=300001" in train
        if method == "ddpm":
            assert "--objective=ddpm" in train
        else:
            assert "--objective=ccua" in train
            assert "--ccua_al=1.0" in train and "--ccua_ucl=1.0" in train
        assert "--uniform_labels" in sample
        assert "--num_class=1000" in sample and "--img_size=64" in sample
        assert "--sample_output=" in sample
        assert str(ROOT / "tools/evaluate_imagenet_lt.py") in metrics
        assert "--image-size 64" in metrics
        assert "--num-classes 1000" in metrics
        assert str(ROOT / "third_party" / "T2H-unified") in metrics
        assert phases[0].skip_if_exists == [
            Path(task.run_dir) / "ckpt_unified_v2_300000.pt",
            Path(task.run_dir) / "unified_host.json",
        ]


def test_ccua_imagenet_patch_is_idempotent(tmp_path):
    repo = tmp_path / "CCUA-DDPM"
    shutil.copytree(ROOT / "third_party/CCUA-DDPM", repo)
    patch = ROOT / "patches/apply_ccua_imagenet_lt.py"
    for _ in range(2):
        subprocess.run([sys.executable, str(patch), str(repo)], check=True)
    dataset = (repo / "dataset.py").read_text(encoding="utf-8")
    main = (repo / "main.py").read_text(encoding="utf-8")
    assert (repo / ".ltx_ccua_imagenet_lt_patch_v1").is_file()
    assert dataset.count("class ImageNetLTManifest") == 1
    assert main.count("flags.DEFINE_string('train_manifest'") == 1
    assert "data_type=imagenet_lt requires --train_manifest" in main
