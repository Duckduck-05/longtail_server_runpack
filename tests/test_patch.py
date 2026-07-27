from pathlib import Path
import subprocess, sys


def fake_coral_main():
    return """import os
import torch
import numpy as np
from torchvision.datasets import CIFAR10
FLAGS = type('F', (), {})()
flags = type('X', (), {'DEFINE_bool':lambda *a,**k:None,'DEFINE_integer':lambda *a,**k:None,'DEFINE_string':lambda *a,**k:None})()
flags.DEFINE_bool('amp', False, help='Use Automatic Mixed Precision for training')
device = torch.device('cuda')


def evaluate(sampler, model, sampled):
    images=[]; labels=[]
    save_image(torch.tensor(images[:256]), 'x', nrow=16)
    (IS, IS_std), FID, prd_score, ipr = get_inception_and_fid_score(
        images, labels, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, FLAGS=FLAGS)
    return (IS, IS_std), FID, prd_score, ipr

def train():
    if FLAGS.frozen_manifest:
        pass
    if FLAGS.data_type == 'cifar10':
        dataset = CIFAR10(
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size,
        shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)
    FLAGS.num_class = 100 if 'cifar100' in FLAGS.data_type else 10

def eval():
    FLAGS.num_class = 100 if 'cifar100' in FLAGS.data_type else 10
""".replace("    if FLAGS.frozen_manifest:\n        pass\n", "")


def test_coral_patch_fail_closed_and_idempotent(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir(); (repo/'main.py').write_text(fake_coral_main())
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root/'patches/apply_coral_weighted_sampler.py'), str(repo)], check=True)
    text=(repo/'main.py').read_text()
    assert "WeightedRandomSampler" in text and "sample_only" in text and "FrozenManifestDataset" in text
    before=text
    subprocess.run([sys.executable, str(root/'patches/apply_coral_weighted_sampler.py'), str(repo)], check=True)
    assert (repo/'main.py').read_text()==before

def test_cm_resume_patch(tmp_path):
    repo=tmp_path/'cm'; (repo/'tools').mkdir(parents=True)
    src='''import argparse\nimport torch\nfrom tqdm import trange\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--device", default=None)\n    args = parser.parse_args()\n    fixed_x_T = torch.randn(\n        min(config["training"]["sample_size"], 100),\n        3,\n        config["dataset"]["img_size"],\n        config["dataset"]["img_size"],\n        device=device,\n    )\n    total_steps = config["training"]["total_steps"]\n    with trange(0, total_steps, dynamic_ncols=True) as pbar:\n        pass\n'''
    (repo/'tools/train.py').write_text(src)
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'patches/apply_cm_resume_patch.py'),str(repo)],check=True)
    text=(repo/'tools/train.py').read_text()
    assert '--resume_ckpt' in text and 'trange(start_step, total_steps' in text


def test_oc_seed_resume_patch(tmp_path):
    repo=tmp_path/'oc'; repo.mkdir()
    (repo/'main.py').write_text("import os\nimport torch\nimport numpy as np\nFLAGS = flags.FLAGS\ndef train():\n    with trange(0, FLAGS.total_steps, dynamic_ncols=True) as pbar:\n        pass\n")
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'patches/apply_oc_seed_patch.py'),str(repo)],check=True)
    text=(repo/'main.py').read_text()
    assert "DEFINE_integer('seed'" in text and 'trange(FLAGS.ckpt_step, FLAGS.total_steps' in text
