from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ltx.adapters.t2h_unified import T2HUnifiedAdapter
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
    contract = campaign.raw["fairness_contract"]
    expected = set(contract["methods"])
    # Derived from the contract rather than hardcoded, so adding a method is a
    # deliberate one-line contract edit instead of a test that fails opaquely.
    assert expected == {"ddpm", "cbdm", "t2h", "cm", "coral", "ccua"}
    assert len(campaign.tasks) == len(contract["cells"]) * len(expected) * len(contract["seeds"]) == 54
    assert {task.method for task in campaign.tasks} == expected
    assert "oc" not in expected
    by_cell = {}
    for task in campaign.tasks:
        by_cell.setdefault(task.dataset["name"], []).append(task)
        # 300k x 64 = 19.2M images seen, matching CBDM/CM (300k x 64) and
        # CORAL (150k x 128) so no baseline runs below its own design point.
        assert task.train["total_steps"] == 300000
        assert task.train["batch_size"] == 64
        assert task.train["save_step"] == 50000
        assert task.eval["checkpoint_step"] == 300000
        # backbone pinned, not inherited from each repo's flag defaults
        assert task.train["ch"] == 128
        assert task.train["ch_mult"] == [1, 2, 2, 2]
        assert task.train["attn"] == [1]
        assert task.train["num_res_blocks"] == 2
        assert task.train["ema_decay"] == 0.9999
        assert task.train["lr"] == 2e-4
        assert task.train["T"] == 1000
        assert task.dataset["split_seed"] == 0
        assert task.eval["num_images"] == 50000
        assert task.eval["uniform_labels"] is True
        assert task.eval["sampler_family"] == "ddim_100"
        assert task.eval["metric_protocol"] == "unified_cifar_v1"
    assert set(by_cell) == {"cifar10lt_if100", "cifar10lt_if1000", "cifar100lt_if100"}
    for tasks in by_cell.values():
        assert {task.method for task in tasks} == expected
        for method in expected:
            assert sorted(task.seed for task in tasks if task.method == method) == [0, 1, 2]


def test_unified_adapters_emit_common_metric_and_label_contract(tmp_path):
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    task_by_method = {task.method: task for task in campaign.tasks if task.seed == 0 and task.dataset["name"] == "cifar10lt_if100"}
    for method, task in task_by_method.items():
        phases = T2HUnifiedAdapter(ROOT).phases(
            replace(task, run_dir=str(tmp_path / method)))
        assert [phase.name for phase in phases] == ["train", "sample", "metrics"]
        train = " ".join(str(value) for value in phases[0].command)
        sample = " ".join(str(value) for value in phases[1].command)
        metrics = " ".join(str(value) for value in phases[-1].command)
        assert "--objective=" + method in train
        assert "--conditional" in train and "--T=1000" in train
        assert "--uniform_labels" in sample and "--sample_method=ddim" in sample
        assert "--ddim_skip_step=10" in sample and "--omega=1.5" in sample
        assert "evaluate_coral2025.py" in metrics
        assert "--kid" in phases[-1].command
        assert "--per-class-output" in phases[-1].command


def test_uniform_label_contract_is_implemented_once_in_the_common_host():
    host = (ROOT / "third_party/T2H-unified/unified_main.py").read_text(encoding="utf-8")
    assert "--uniform_labels" in host
    assert "torch.arange(start, start + n" in host
    assert "method = \"ddim\"" in host


def test_common_host_exports_one_array_and_one_aligned_label_array():
    """A 50k CIFAR array is ~600 MB; the host writes one canonical pair."""
    host = (ROOT / "third_party/T2H-unified/unified_main.py").read_text(encoding="utf-8")
    assert "np.save(output, image_array)" in host
    assert "np.save(str(output) + \".labels.npy\", label_array)" in host


def test_ccua_row_is_the_alignment_contrastive_pair_alone():
    """CCUA is an objective dispatch inside the common T2H host."""
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    tasks = [t for t in campaign.tasks if t.method == "ccua"]
    assert len(tasks) == 9
    phases = T2HUnifiedAdapter(ROOT).phases(tasks[0])
    train = " ".join(str(value) for value in phases[0].command)
    assert "--objective=ccua" in train
    assert "--ccua_al=1.0" in train and "--ccua_ucl=1.0" in train
    evaluate = " ".join(str(value) for value in phases[1].command)
    for flag in ("--ch=128", "--num_res_blocks=2", "--ema_decay=0.9999"):
        assert flag in train
    assert "--uniform_labels" in evaluate
    assert "--sample_method=ddim" in evaluate and "--ddim_skip_step=10" in evaluate
    assert "--kid" in phases[-1].command and "--per-class-output" in phases[-1].command


def test_ccua_resume_asks_for_the_remaining_budget_not_a_second_full_run(tmp_path):
    """The common host keeps the endpoint fixed and resumes at checkpoint+1.

    A checkpoint filename by itself is not enough for automatic reuse: the
    host manifest must certify that it belongs to this exact task/objective.
    """
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    task = replace([t for t in campaign.tasks if t.method == "ccua"][0], run_dir=str(tmp_path / "run"))
    adapter = T2HUnifiedAdapter(ROOT)

    fresh = " ".join(str(value) for value in adapter.phases(task)[0].command)
    assert "--total_steps=300001" in fresh and "--resume" not in fresh

    expected = adapter._expected_host_provenance(task, 64)
    torch.save({
        "step": 50000,
        "net_model": {}, "ema_model": {}, "optim": {}, "sched": {},
        "provenance": expected,
    }, tmp_path / "run" / "ckpt_unified_v2_50000.pt")
    unverified = " ".join(str(value) for value in adapter.phases(task)[0].command)
    assert "--total_steps=300001" in unverified
    assert "--resume_checkpoint=" not in unverified

    (tmp_path / "run" / "unified_host.json").write_text(json.dumps({
        "host": "T2H-unified",
        "host_revision": adapter.host_revision,
        "checkpoint_schema": adapter.checkpoint_schema,
        "objective": expected["objective"],
        "seed": task.seed,
        "data_type": task.dataset["data_type"],
        "num_class": task.dataset["num_class"],
        "total_steps_bound": 300001,
        "provenance": expected,
    }), encoding="utf-8")
    resumed = " ".join(str(value) for value in adapter.phases(task)[0].command)
    assert "--total_steps=300001" in resumed
    assert f"--resume_checkpoint={tmp_path / 'run' / 'ckpt_unified_v2_50000.pt'}" in resumed


def test_explicit_unified_resume_wins_over_local_checkpoint_discovery(tmp_path):
    """The CLI-selected checkpoint is the intended provenance, not a local guess."""
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    selected = tmp_path / "external" / "ckpt_unified_v2_100000.pt"
    selected.parent.mkdir()
    selected.write_bytes(b"")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "ckpt_unified_v2_250000.pt").write_bytes(b"")
    task = replace(
        next(t for t in campaign.tasks if t.method == "ddpm"),
        run_dir=str(run_dir),
        method_config={"resume_checkpoint": str(selected), "resume_mode": "full"},
    )

    train = " ".join(str(value) for value in T2HUnifiedAdapter(ROOT).phases(task)[0].command)
    assert f"--resume_checkpoint={selected}" in train
    assert str(run_dir / "ckpt_unified_v2_250000.pt") not in train


def test_unified_rejects_ema_only_external_resume(tmp_path):
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    selected = tmp_path / "ckpt_unified_v2_100000.pt"
    selected.write_bytes(b"")
    task = replace(
        next(t for t in campaign.tasks if t.method == "ddpm"),
        run_dir=str(tmp_path / "run"),
        method_config={"resume_checkpoint": str(selected), "resume_mode": "ema_only"},
    )

    with pytest.raises(ValueError, match="full-state resume"):
        T2HUnifiedAdapter(ROOT).phases(task)


def seed_completed_runs(tmp_path, monkeypatch):
    """Populate a full campaign of completed runs under tmp_path."""
    monkeypatch.setenv("LTX_RUNS_ROOT", str(tmp_path / "runs"))
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    for index, task in enumerate(campaign.tasks):
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
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
        sample_provenance = {
            "host_revision": "t2h-unified-common-v2",
            "checkpoint_schema": 2,
            "artifact_namespace": "t2h_unified_v2",
        }
        metric_provenance = {
            "metric_host": "common_cifar_metrics_v2",
            "sample": sample_provenance,
        }
        (run_dir / "metrics.unified.v2.json").write_text(json.dumps({
            "metrics": metrics, "provenance": metric_provenance,
        }))
        (run_dir / "metrics.per_class.v2.json").write_text(json.dumps({
            "groups": {
            group: {"FID": 12.0 + index / 100 + group_index / 10, "generated": 15000, "reference": 15000}
            for group_index, group in enumerate(("Many", "Medium", "Few"))
            }, "provenance": metric_provenance,
        }))
    return campaign


def load_report_module():
    spec = importlib.util.spec_from_file_location("unified_report", ROOT / "tools/report_unified_cifar.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_report_writes_one_complete_row_per_cell_and_method(tmp_path, monkeypatch):
    campaign = seed_completed_runs(tmp_path, monkeypatch)
    contract = campaign.raw["fairness_contract"]
    cells, methods = len(contract["cells"]), len(contract["methods"])
    rows = cells * methods

    spec = importlib.util.spec_from_file_location("unified_report", ROOT / "tools/report_unified_cifar.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["report", "--config", str(ROOT / "configs/unified_cifar.yaml"), "--output", str(output)])
    assert module.main() == 0
    table = (output / "table.md").read_text(encoding="utf-8")
    # table.md now holds two tables; count rows per section so the main
    # one-row-per-cell-and-method assertion cannot be satisfied by the
    # advantage table's rows.
    main_table, _, advantage_table = table.partition("## Advantage over")
    assert main_table.count("| cifar") == rows
    # every non-baseline method x every cell, each compared against that cell's DDPM
    assert advantage_table.count("| cifar") == (methods - 1) * cells
    assert "ddpm" not in advantage_table.split("|---")[-1]
    assert "KID ↓" in main_table
    assert (output / "tail_breakdown.md").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(payload["aggregate"]) == rows
    assert all(row["complete"] for row in payload["aggregate"])
    # every non-baseline row carries a bootstrap CI on its paired-seed gain
    for row in payload["aggregate"]:
        if row["method"] == "ddpm":
            continue
        advantage = row["vs_ddpm_FID"]
        assert advantage["n_pairs"] == 3
        assert advantage["ci95_low"] <= advantage["mean"] <= advantage["ci95_high"]


def test_report_never_falls_back_to_legacy_metrics_for_a_configured_v2_row(tmp_path):
    module = load_report_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.unified.json").write_text(json.dumps({"FID": 1.0}))
    assert module.read_metrics(run_dir, "metrics.unified.v2.json") == {}


def test_unified_metric_phase_does_not_skip_on_headline_snapshot(tmp_path):
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    task = next(t for t in campaign.tasks if t.method == "ddpm" and t.seed == 0)
    adapter = T2HUnifiedAdapter(ROOT)
    expected = {
        "host_revision": adapter.host_revision,
        "checkpoint_schema": adapter.checkpoint_schema,
        "objective": "ddpm",
        "checkpoint_step": 300000,
        "num_images": 50000,
        "sample_method": "ddim",
        "sampler_method": "ddim",
        "ddim_skip_step": 10,
        "omega": 1.5,
        "uniform_labels": True,
        "seed": 0,
        "artifact_namespace": "t2h_unified_v2",
        "T": 1000,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "var_type": "fixedlarge",
        "img_size": 32,
        "num_class": 10,
    }
    provenance = {"metric_host": "common_cifar_metrics_v2", "sample": expected}
    headline = tmp_path / "headline.json"
    headline.write_text(json.dumps({
        "metrics": {"FID": 1.0, "KID": 0.1, "IS": 2.0, "F_8": 0.5, "F_1_8": 0.4},
        "provenance": provenance,
    }), encoding="utf-8")
    assert not adapter._metric_provenance_valid(
        headline, expected, "common_cifar_metrics_v2"
    )
    headline.write_text(json.dumps({
        "metrics": {
            "FID": 1.0, "KID": 0.1, "IS": 2.0, "F_8": 0.5, "F_1_8": 0.4,
            "ImprovedPrecision": 0.6, "Recall": 0.7,
        },
        "provenance": provenance,
    }), encoding="utf-8")
    assert adapter._metric_provenance_valid(
        headline, expected, "common_cifar_metrics_v2"
    )


def test_unified_metric_collection_ignores_legacy_files_in_a_v2_run(tmp_path):
    (tmp_path / "unified_host.json").write_text(json.dumps({
        "host_revision": "t2h-unified-common-v2", "checkpoint_schema": 2,
    }))
    (tmp_path / "metrics.unified.v2.json").write_text(json.dumps({
        "metrics": {"FID": 12.3},
    }))
    (tmp_path / "metrics.unified.json").write_text(json.dumps({
        "metrics": {"FID": 99.0},
    }))
    from ltx.metrics import collect_metrics
    assert collect_metrics(tmp_path)["generation/FID"] == 12.3


def test_report_bundle_includes_both_logs(tmp_path, monkeypatch):
    """Everything a remote reader needs must land in the report dir, since it
    is exactly the file set uploaded to W&B as the report artifact."""
    seed_completed_runs(tmp_path, monkeypatch)
    campaign_dir = tmp_path / "runs" / "unified_cifar_t2h_v1"
    (campaign_dir / "logs").mkdir(parents=True, exist_ok=True)
    run_log = campaign_dir / "logs" / "run_20260821T000000Z.log"
    run_log.write_text("[ltx] gpu slots={0: 4}\n[ltx] campaign finished completed=45 failed=0\n")
    (campaign_dir / "latest.log").symlink_to(run_log)

    module = load_report_module()
    output = campaign_dir / "report"
    monkeypatch.setattr("sys.argv", ["report", "--config", str(ROOT / "configs/unified_cifar.yaml"),
                                     "--output", str(output)])
    assert module.main() == 0

    for name in ("per_seed.csv", "tail_per_seed.csv", "table.md", "tail_breakdown.md",
                 "summary.json", "results.log", "campaign_run.log"):
        assert (output / name).is_file(), f"missing {name} from the hand-off bundle"
    # the campaign stdout is snapshotted, not linked, so log rotation cannot orphan it
    assert not (output / "campaign_run.log").is_symlink()
    assert "campaign finished" in (output / "campaign_run.log").read_text()


def test_omega_sweep_reuses_one_trained_checkpoint(tmp_path):
    """Guidance strength only affects sampling, so a sweep must add sampling
    and metric phases without ever retraining."""
    from ltx.adapters import make_adapter
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    for method in ("t2h", "cm", "ddpm"):
        task = [t for t in campaign.tasks if t.method == method and t.seed == 0][0]
        run_dir = tmp_path / f"{method}-sweep"
        swept = replace(task, run_dir=str(run_dir),
                        method_config={**task.method_config, "guidance_scales": [1.0, 1.5, 2.0]})
        phases = make_adapter(task.adapter, ROOT).phases(swept)
        names = [p.name for p in phases]
        assert names.count("train") == 1, f"{method} retrains per omega: {names}"
        for omega in (1.5, 2.0):
            assert any(f"w{omega}" in n for n in names), f"{method} missing omega {omega}: {names}"


def test_single_omega_keeps_the_original_phase_names(tmp_path):
    """A one-omega run must keep its historical artefact names so an
    in-progress campaign still resumes instead of re-sampling from scratch."""
    from ltx.adapters import make_adapter
    campaign = load_campaign(ROOT / "configs/unified_cifar.yaml")
    for method, expected in (("t2h", ["train", "sample", "metrics"]),
                             ("cm", ["train", "sample", "metrics"])):
        task = [t for t in campaign.tasks if t.method == method and t.seed == 0][0]
        phases = make_adapter(task.adapter, ROOT).phases(
            replace(task, run_dir=str(tmp_path / f"{method}-single")))
        assert [p.name for p in phases] == expected
