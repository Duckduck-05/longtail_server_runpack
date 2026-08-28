import json
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest

from ltx.config import load_campaign
from ltx.metrics import collect_metrics
from ltx.worker import init_wandb, wandb_config


def unified_campaign():
    return load_campaign("configs/unified_cifar.yaml")


def unified_tasks():
    return unified_campaign().tasks


def expected_counts():
    """Row/group counts derived from the contract, not hardcoded, so adding a
    method is a deliberate contract edit rather than an opaque test failure."""
    contract = unified_campaign().raw["fairness_contract"]
    cells, methods, seeds = (len(contract[k]) for k in ("cells", "methods", "seeds"))
    return cells * methods * seeds, cells * methods


def run_name(task):
    dataset = str(task.dataset.get("name", task.stage))
    return f"{dataset}-{task.method}-s{task.seed}"


def run_group(task):
    dataset = str(task.dataset.get("name", task.stage))
    return f"{dataset}-{task.method}"


def test_run_names_are_unique_and_free_of_stage_artifacts():
    """Naming by stage produced "c10_if100_cm-cm-s0", "c10_if100_t2h-t2h-s0",
    and a meaningless "core" for DDPM/CBDM/CORAL. The name must identify the
    table row: dataset, method, seed."""
    tasks = unified_tasks()
    total, _ = expected_counts()
    names = [run_name(t) for t in tasks]
    assert len(names) == len(set(names)) == total == 54
    for task, name in zip(tasks, names):
        assert "core" not in name
        assert name == f"{task.dataset['name']}-{task.method}-s{task.seed}"
        # no doubled method token, e.g. "...-cm-cm-s0"
        assert name.count(f"-{task.method}-") == 1


def test_runs_group_by_cell_and_method_so_seeds_aggregate():
    """The report averages the three seeds of one (dataset, method) cell, so
    that is what W&B must group. Grouping by stage put three different methods
    in one group, which cannot produce a seed mean/std band."""
    tasks = unified_tasks()
    _, expected_groups = expected_counts()
    groups = {}
    for task in tasks:
        groups.setdefault(run_group(task), set()).add(task.seed)
    assert len(groups) == expected_groups == 18
    for group, seeds in groups.items():
        assert seeds == {0, 1, 2}, f"{group} has seeds {seeds}"


def test_wandb_config_is_curated_not_a_full_task_dump():
    task = unified_tasks()[0]
    config = wandb_config(task)
    # identity + fairness contract, not paths/retry/runner settings
    for key in ("campaign", "dataset", "method", "seed", "train/total_steps",
                "train/batch_size", "train/lr", "eval/num_images", "eval/metric_protocol"):
        assert key in config, key
    assert config["eval/inception_batch_size"] == 16
    for absent in ("runtime", "retry", "priority", "tags", "repos_root", "wandb_project"):
        assert absent not in config, absent
    assert len(config) < 30


def test_online_wandb_init_fails_closed_instead_of_silently_losing_results(tmp_path, monkeypatch):
    class BrokenWandb:
        Settings = lambda self, **kwargs: SimpleNamespace(**kwargs)

        @staticmethod
        def init(**kwargs):
            raise OSError("network unavailable")

    task = unified_tasks()[0]
    monkeypatch.setitem(sys.modules, "wandb", BrokenWandb())
    monkeypatch.setenv("WANDB_MODE", "online")
    with pytest.raises(RuntimeError, match="W&B online mode"):
        init_wandb(task, Path(tmp_path))


def test_bookkeeping_values_never_reach_the_metric_stream(tmp_path):
    """Sample counts, seeds and protocol constants are provenance. Plotting
    them beside FID invites reading 50000 as a result."""
    (tmp_path / "metrics.cm.json").write_text(json.dumps({
        "protocol": "CM CIFAR-LT", "FID": 9.5,
        "KID": {"mean": 0.02, "std": 0.001, "all": [0.02]},
        "num_generated": 20000, "num_reference": 20000, "seed": 0,
    }))
    out = collect_metrics(tmp_path)
    assert out["generation/FID"] == 9.5
    assert out["generation/KID"] == 0.02
    for junk in ("generation/num_generated", "generation/num_reference", "generation/seed"):
        assert junk not in out
