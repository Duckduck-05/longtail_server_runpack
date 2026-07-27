from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from ltx.adapters.cm import CMAdapter
from ltx.adapters.coral import CoralAdapter
from ltx.adapters.oc import OCAdapter
from ltx.config import load_campaign
from ltx.paper_metrics import polynomial_mmd_kid


ROOT = Path(__file__).resolve().parents[1]


def test_common_kid_is_finite_and_reproducible_with_its_locked_subset_seed():
    rng = np.random.default_rng(7)
    generated = rng.normal(size=(31, 8)).astype(np.float32)
    reference = rng.normal(size=(37, 8)).astype(np.float32)
    first = polynomial_mmd_kid(generated, reference, num_subsets=4, max_subset_size=20, rng=np.random.default_rng(2026))
    second = polynomial_mmd_kid(generated, reference, num_subsets=4, max_subset_size=20, rng=np.random.default_rng(2026))
    assert np.isfinite(first)
    assert first == second


def test_unified_matrix_is_one_nonduplicated_fair_table():
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    assert len(campaign.tasks) == 45
    expected = {"ddpm", "cbdm", "t2h", "cm", "coral"}
    assert {task.method for task in campaign.tasks} == expected
    assert "oc" not in expected
    by_cell = {}
    for task in campaign.tasks:
        by_cell.setdefault(task.dataset["name"], []).append(task)
        assert task.train["total_steps"] == 200000
        assert task.train["batch_size"] == 64
        assert task.train["lr"] == 2e-4
        assert task.train["T"] == 1000
        assert task.dataset["split_seed"] == 0
        assert task.eval["num_images"] == 50000
        assert task.eval["uniform_labels"] is True
        assert task.eval["sampler_family"] == "ancestral_ddpm"
        assert task.eval["metric_protocol"] == "unified_cifar_v1"
    assert set(by_cell) == {"cifar10lt_if100", "cifar10lt_if1000", "cifar100lt_if100"}
    for tasks in by_cell.values():
        assert {task.method for task in tasks} == expected
        for method in expected:
            assert sorted(task.seed for task in tasks if task.method == method) == [0, 1, 2]


def test_unified_adapters_emit_common_metric_and_label_contract(tmp_path):
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    task_by_method = {task.method: task for task in campaign.tasks if task.seed == 0 and task.dataset["name"] == "cifar10lt_if100"}

    cm_task = replace(task_by_method["cm"], run_dir=str(tmp_path / "cm"))
    cm_phases = CMAdapter(ROOT).phases(cm_task)
    assert [phase.name for phase in cm_phases] == ["train", "sample", "unified_metrics"]
    resolved = yaml.safe_load((tmp_path / "cm" / "cm.resolved.yaml").read_text())
    assert resolved["dataset"]["imb_factor"] == 0.01
    assert resolved["evaluation"]["sample_method"] == "ddpm"
    assert resolved["evaluation"]["omega"] == 1.0
    assert "--samples_output" in cm_phases[1].command[-1]
    assert "evaluate_coral2025.py" in " ".join(cm_phases[-1].command)
    assert "--kid" in cm_phases[-1].command
    assert "--per-class-output" in cm_phases[-1].command

    coral_task = replace(task_by_method["ddpm"], run_dir=str(tmp_path / "coral"))
    coral_phases = CoralAdapter(ROOT).phases(coral_task)
    assert "--uniform_labels" in coral_phases[1].command
    assert "--kid" in coral_phases[-1].command
    assert "--per-class-output" in coral_phases[-1].command
    assert any(str(value).endswith("metrics.unified.json") for value in coral_phases[-1].command)

    t2h_task = replace(task_by_method["t2h"], run_dir=str(tmp_path / "t2h"))
    t2h_phases = OCAdapter(ROOT).phases(t2h_task)
    eval_command = " ".join(t2h_phases[1].command)
    assert "--uniform_labels" in eval_command
    assert "--sample_method=ddpm" in eval_command
    assert "--ddim_skip_step=1" in eval_command
    assert "--kid" in t2h_phases[-1].command
    assert "--per-class-output" in t2h_phases[-1].command


def test_uniform_label_ports_are_present_in_vendored_sources():
    coral_main = (ROOT / "third_party/coral-lt-diffusion/main.py").read_text(encoding="utf-8")
    coral_diffusion = (ROOT / "third_party/coral-lt-diffusion/diffusion.py").read_text(encoding="utf-8")
    t2h = (ROOT / "third_party/OC_LT/ddpm_gen.py").read_text(encoding="utf-8")
    assert "flags.DEFINE_bool('uniform_labels'" in coral_main
    assert "labels=forced_labels" in coral_main
    assert "method='cfg', labels=None" in coral_diffusion
    assert "flags.DEFINE_bool('uniform_labels'" in t2h
    assert "torch.arange(i, i + batch_size" in t2h


def test_unified_report_writes_one_complete_fifteen_row_table(tmp_path, monkeypatch):
    monkeypatch.setenv("LTX_RUNS_ROOT", str(tmp_path / "runs"))
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    for index, task in enumerate(campaign.tasks):
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True)
        (run_dir / "SUCCESS").write_text("ok\n")
        seed = task.seed
        metrics = {
            "FID": 10.0 + index / 100 + seed / 1000,
            "KID": 0.003 + index / 1_000_000 + seed / 10_000_000,
            "IS": 5.0 + index / 100 + seed / 1000,
            "F_8": 0.5 + index / 10000 + seed / 100000,
            "F_1_8": 0.4 + index / 10000 + seed / 100000,
            "ImprovedPrecision": 0.6 + index / 10000 + seed / 100000,
            "Recall": 0.7 + index / 10000 + seed / 100000,
        }
        (run_dir / "metrics.unified.json").write_text(json.dumps({"metrics": metrics}))
        (run_dir / "metrics.per_class.json").write_text(json.dumps({"groups": {
            group: {"FID": 12.0 + index / 100 + group_index / 10, "generated": 15000, "reference": 15000}
            for group_index, group in enumerate(("Many", "Medium", "Few"))
        }}))

    spec = importlib.util.spec_from_file_location("unified_report", ROOT / "tools/report_unified_cifar.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["report", "--config", str(ROOT / "configs/unified_cifar.yaml"), "--output", str(output)])
    assert module.main() == 0
    table = (output / "table.md").read_text(encoding="utf-8")
    assert table.count("| cifar") == 15
    assert "KID ↓" in table
    assert (output / "tail_breakdown.md").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(payload["aggregate"]) == 15
    assert all(row["complete"] for row in payload["aggregate"])
