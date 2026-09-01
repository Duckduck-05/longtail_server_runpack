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


def test_oc_compiled_ckpt_patch(tmp_path):
    repo=tmp_path/'oc'; repo.mkdir()
    src=("import torch\n"
         "def eval():\n"
         "    ckpt = torch.load('x')\n"
         "    model.load_state_dict(ckpt['net_model'])\n"
         "    model.load_state_dict(ckpt['ema_model'])\n")
    (repo/'ddpm_gen.py').write_text(src)
    root=Path(__file__).resolve().parents[1]
    script=str(root/'patches/apply_oc_compiled_ckpt.py')
    subprocess.run([sys.executable,script,str(repo)],check=True)
    text=(repo/'ddpm_gen.py').read_text()
    assert "_strip_compile_prefix(ckpt['net_model'])" in text
    assert "_strip_compile_prefix(ckpt['ema_model'])" in text
    subprocess.run([sys.executable,script,str(repo)],check=True)
    assert (repo/'ddpm_gen.py').read_text()==text
    ns={}
    exec(text.split('def eval():')[0], ns)
    stripped=ns['_strip_compile_prefix']({'_orig_mod.head.weight':1,'head.bias':2})
    assert stripped=={'head.weight':1,'head.bias':2}


def test_coral_preserve_ckpt_patch_is_idempotent_and_selective(tmp_path, monkeypatch):
    repo = tmp_path / 'coral'
    repo.mkdir()
    src = '''import os

class _Flags:
    logdir = None
    save_step = None

FLAGS = _Flags()
\ndef save_boundary(logdir, step, save_step):
    FLAGS.logdir = logdir
    FLAGS.save_step = save_step
    if True:
        if True:
            if True:
                prev_ckpt = os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step - FLAGS.save_step))
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
'''
    (repo / 'main.py').write_text(src)
    root = Path(__file__).resolve().parents[1]
    script = root / 'patches/apply_coral_preserve_ckpt.py'

    subprocess.run([sys.executable, str(script), str(repo)], check=True)
    patched = (repo / 'main.py').read_text()
    subprocess.run([sys.executable, str(script), str(repo)], check=True)
    assert (repo / 'main.py').read_text() == patched

    ns = {}
    exec(compile(patched, str(repo / 'main.py'), 'exec'), ns)
    old_ckpt = repo / 'ckpt_100.pt'
    old_ckpt.write_bytes(b'checkpoint')
    monkeypatch.delenv('PRESERVE_CKPT_STEPS', raising=False)
    ns['save_boundary'](str(repo), 200, 100)
    assert not old_ckpt.exists()

    old_ckpt.write_bytes(b'checkpoint')
    monkeypatch.setenv('PRESERVE_CKPT_STEPS', '100, 300')
    ns['save_boundary'](str(repo), 200, 100)
    assert old_ckpt.exists()
