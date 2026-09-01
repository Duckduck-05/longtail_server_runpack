from pathlib import Path
from types import SimpleNamespace

import pytest

from ltx.adapters.base import resolve_inception_batch_size
from ltx.adapters.coral import CoralAdapter
from ltx.checkpoints import get_resume_spec, get_resume_step, validate_checkpoint_keys
from ltx.cli import apply_resume_override
from ltx.config import Task, load_campaign


ROOT = Path(__file__).resolve().parents[1]


def coral_task(tmp_path, *, method_config=None, evaluate=None):
    return Task(
        id="coral-task", campaign="c", stage="s", adapter="coral", method="ddpm",
        seed=0, priority=1,
        dataset={"data_type": "cifar100lt", "imbalance_factor": 0.01, "root": str(tmp_path / "data")},
        train={"total_steps": 200, "batch_size": 64, "save_step": 100},
        eval={"num_images": 50, "paper_metrics": False, **(evaluate or {})},
        method_config=method_config or {},
        repository={}, runtime={"repos_root": str(tmp_path / "repos"), "python": "python"},
        retry={}, run_dir=str(tmp_path / "run"),
    )


def test_external_resume_requires_an_existing_explicit_file(tmp_path):
    train = {}
    with pytest.raises(FileNotFoundError, match="explicit resume checkpoint"):
        get_resume_spec(train, {"resume_checkpoint": str(tmp_path / "missing.pt")})


def test_resume_step_comes_from_filename_or_explicit_override(tmp_path):
    checkpoint = tmp_path / "renamed.pt"
    checkpoint.write_bytes(b"x")
    with pytest.raises(ValueError, match="cannot infer resume step"):
        get_resume_step({}, {}, checkpoint)
    assert get_resume_step({}, {"resume_step": 42}, checkpoint) == 42
    namespaced = tmp_path / "ckpt_unified_v2_200000.pt"
    namespaced.write_bytes(b"x")
    assert get_resume_step({}, {}, namespaced) == 200000


def test_checkpoint_key_validation_rejects_legacy_file_for_exact_resume():
    with pytest.raises(ValueError, match="resume_mode=ema_only"):
        validate_checkpoint_keys({"ema_model", "step"}, "full", "old.pt")
    validate_checkpoint_keys({"ema_model", "step"}, "ema_only", "old.pt")


def test_coral_local_resume_points_to_run_dir_and_target_checkpoint(tmp_path):
    task = coral_task(tmp_path, evaluate={"paper_metrics": True})
    run = Path(task.run_dir)
    run.mkdir(parents=True)
    (run / "ckpt_100.pt").write_bytes(b"checkpoint")

    phases = CoralAdapter(ROOT).phases(task)
    train = next(phase for phase in phases if phase.name == "train").command
    metrics = next(phase for phase in phases if phase.name == "paper_metrics_w1.0").command
    assert "--total_steps=201" in train
    assert "--total_steps=200" not in train
    assert f"--resume_checkpoint={run / 'ckpt_100.pt'}" in train
    assert "--ckpt_step=100" in train
    assert "--allow_non_exact_resume" not in train
    assert "--inception-batch-size" in metrics
    assert metrics[metrics.index("--inception-batch-size") + 1] == "16"


def test_coral_external_ema_checkpoint_is_explicit_non_exact_warm_start(tmp_path):
    checkpoint = tmp_path / "ckpt_100.pt"
    checkpoint.write_bytes(b"legacy")
    task = coral_task(tmp_path, method_config={
        "resume_checkpoint": str(checkpoint), "resume_mode": "ema_only",
    })
    train = next(phase for phase in CoralAdapter(ROOT).phases(task) if phase.name == "train").command
    assert f"--resume_checkpoint={checkpoint}" in train
    assert "--ckpt_step=100" in train
    assert "--allow_non_exact_resume" in train
    assert "--exact_resume" not in train


def test_cli_external_resume_override_is_per_seed_and_changes_payload():
    campaign = load_campaign(ROOT / "configs/unified_cifar_c100.yaml")
    args = SimpleNamespace(
        resume_checkpoint="/checkpoints/{method}/seed_{seed}/ckpt_200000.pt",
        resume_method="ddpm", resume_seed=None, resume_mode="ema_only", resume_step=None,
    )
    apply_resume_override(campaign, args)
    ddpm = sorted((task for task in campaign.tasks if task.method == "ddpm"), key=lambda task: task.seed)
    assert [task.method_config["resume_checkpoint"] for task in ddpm] == [
        "/checkpoints/ddpm/seed_0/ckpt_200000.pt",
        "/checkpoints/ddpm/seed_1/ckpt_200000.pt",
        "/checkpoints/ddpm/seed_2/ckpt_200000.pt",
    ]
    assert all(task.method_config["resume_mode"] == "ema_only" for task in ddpm)


def test_inception_batch_size_is_positive_and_independent_of_train_batch():
    assert resolve_inception_batch_size({"batch_size": 128}) == 16
    assert resolve_inception_batch_size({"inception_batch_size": 8}) == 8
    with pytest.raises(ValueError, match="inception_batch_size must be positive"):
        resolve_inception_batch_size({"inception_batch_size": 0})


def test_vendored_coral_trainer_has_explicit_external_resume_and_next_step_guard():
    source = (ROOT / "third_party/coral-lt-diffusion/main.py").read_text(encoding="utf-8")
    assert "DEFINE_string(\n    'resume_checkpoint'" in source
    assert "DEFINE_bool(\n    'allow_non_exact_resume'" in source
    assert "missing full training state" in source
    assert "resume_start_step = FLAGS.ckpt_step + 1" in source
