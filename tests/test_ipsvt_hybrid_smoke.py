"""Focused graph/config checks for the isolated natural-batch hybrid smoke."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from ltx.adapters.t2h_unified import T2HUnifiedAdapter
from ltx.config import load_campaign


ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT = ROOT / "third_party" / "T2H-unified"
HYBRID_PATH = HOST_ROOT / "ipsvt_hybrid.py"
OBJECTIVES_PATH = HOST_ROOT / "unified_objectives.py"
HOST_PATH = HOST_ROOT / "unified_main.py"
CONFIG = ROOT / "configs" / "ipsvt_hybrid_smoke_c100_if100.yaml"
RESPONSE_CONFIG = ROOT / "configs" / "ipsvt_response_smoke_c100_if100.yaml"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HYBRID = _load_module("ipsvt_hybrid_under_test", HYBRID_PATH)
OBJECTIVES = _load_module("unified_objectives_hybrid_under_test", OBJECTIVES_PATH)


class TinyHybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embedding = torch.nn.Embedding(3, 2)
        with torch.no_grad():
            self.label_embedding.weight.copy_(torch.tensor([
                [-0.4, 0.2], [0.1, 0.5], [0.6, -0.3],
            ]))
        self.scale = torch.nn.Parameter(torch.tensor(0.7))
        self.base_calls = 0
        self.aux_grad_enabled: list[bool] = []

    def forward(self, x, _t, *, y, augm, return_mid, use_cm):
        del augm, return_mid, use_cm
        self.base_calls += 1
        condition = torch.zeros((len(x), 2), dtype=x.dtype, device=x.device)
        if y is not None:
            condition = self.label_embedding(y)
        return self.scale * x * (1.0 + condition.mean(dim=1).view(-1, 1, 1, 1))


def _conditioned_forward(model, x, _t, condition):
    model.aux_grad_enabled.append(torch.is_grad_enabled())
    return model.scale * x * (1.0 + condition.mean(dim=1).view(-1, 1, 1, 1))


def _context(model):
    x0 = torch.tensor([[[[1.0]]], [[[2.0]]]])
    noise = torch.tensor([[[[0.25]]], [[[-0.75]]]])
    return {
        "model": model,
        "x0": x0,
        "x_t": x0 + noise,
        "t": torch.tensor([1, 2], dtype=torch.long),
        "noise": noise,
        "y": torch.tensor([0, 2], dtype=torch.long),
    }


def _hybrid(*, s: float, lambda_aux: float = 1.0, lambda_svt: float = 1.0):
    return HYBRID.IPSVTHybridAuxiliary(
        T=4, beta_1=1e-4, beta_T=0.02, K=2, s=s, delta=0.1, tau=1e-6,
        lambda_aux=lambda_aux, lambda_svt=lambda_svt, chunk_size=2,
        conditioned_forward=_conditioned_forward, use_checkpoint=False,
    )


def test_hybrid_hook_has_no_probe_ddpm_target_and_host_base_is_the_only_ddpm_loss():
    """At s=0 the Twin/SVT hook is exactly zero despite imperfect eps output."""
    torch.manual_seed(3)
    model = TinyHybridModel()
    hook = _hybrid(s=0.0)
    context = _context(model)
    hook_loss, hook_diag = hook(context)

    # Any f1 noise-target term would generally be non-zero here.  Clean and
    # perturbed conditions coincide, so the corrected hook contains only zero
    # Twin/SVT transfer losses.
    assert hook_loss.item() == pytest.approx(0.0, abs=1e-8)
    assert hook_diag["twin"].item() == pytest.approx(0.0, abs=1e-8)
    assert hook_diag["svt"].item() == pytest.approx(0.0, abs=1e-8)

    model = TinyHybridModel()
    objective = OBJECTIVES.UnifiedObjective("ipsvt", T=4, ipsvt_hook=_hybrid(s=0.0))
    x0, y = _context(model)["x0"], _context(model)["y"]
    noise = _context(model)["noise"]
    loss, diagnostics = objective(model, x0, y, t=torch.tensor([1, 2]), noise=noise)
    assert model.base_calls == 1
    assert loss.item() == pytest.approx(diagnostics["dsm"].item(), rel=1e-6)
    assert diagnostics["ipsvt_aux"].item() == pytest.approx(0.0, abs=1e-8)


def test_hybrid_stop_gradient_is_one_way_but_c_y_remains_trainable():
    torch.manual_seed(17)
    model = TinyHybridModel()
    loss, diagnostics = _hybrid(s=0.3, lambda_aux=1.5, lambda_svt=0.5)(_context(model))

    # K=2 means three stopped clean teachers, then three student forwards.
    assert model.aux_grad_enabled == [False, False, False, True, True, True]
    assert diagnostics["hook_logical_forwards"].item() == 6
    assert diagnostics["hook_gradient_forwards"].item() == 3
    assert diagnostics["total_logical_forwards"].item() == 7
    assert diagnostics["total_gradient_forwards"].item() == 4
    assert loss.item() == pytest.approx(
        1.5 * (diagnostics["twin"].item() + 0.5 * diagnostics["svt"].item()), rel=1e-6
    )
    loss.backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
    # c_tilde starts at c_y, not sg(c_y), so the natural label embedding gets
    # student-side gradients while all clean-teacher forwards stay detached.
    assert model.label_embedding.weight.grad is not None
    assert model.label_embedding.weight.grad.abs().sum().item() > 0


def test_hybrid_source_guards_stop_gradient_k_and_legacy_sampler_exclusion():
    source = HYBRID_PATH.read_text(encoding="utf-8")
    assert "c_y = embedding(y)" in source
    assert "c_y.detach()" not in source
    assert "embedding.weight.detach()" in source
    assert "condition_sigma = (self.s * r_c / (c_y.shape[-1] ** 0.5)).detach()" in source
    assert "clean_predictions[0].detach()" in source
    assert "ddpm_sum" not in source and "ddpm_probe" not in source and "f1_clean" not in source
    for forbidden in ("IPSVTAuxiliary", "class_probabilities", "ipsvt_every", "ipsvt_batch", "sample_batch"):
        assert forbidden not in source
    with pytest.raises(ValueError, match="K >= 2"):
        HYBRID.IPSVTHybridAuxiliary(
            T=4, beta_1=1e-4, beta_T=0.02, K=1, s=0.05, delta=0.1, tau=1e-6,
            conditioned_forward=_conditioned_forward,
        )


def test_hybrid_gram_uses_the_legacy_off_diagonal_k_times_k_minus_one_denominator():
    perturbed = torch.ones(1, 3, 3)
    clean = torch.zeros_like(perturbed)
    # Six off-diagonal entries / (3 * 2) = 1; diagonal differences must vanish.
    assert HYBRID.gram_discrepancy(perturbed, clean).item() == pytest.approx(1.0)
    off = HYBRID.off_diagonal(perturbed)
    assert torch.equal(torch.diagonal(off, dim1=1, dim2=2), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="K >= 2"):
        HYBRID.gram_discrepancy(torch.ones(1, 1, 1), torch.zeros(1, 1, 1))


def test_hybrid_smoke_config_generates_one_exact_220k_task_without_kid_or_legacy_flags(tmp_path):
    campaign = load_campaign(CONFIG)
    assert campaign.raw["smoke_continuation"] == {
        "source_checkpoint": "runs/unified_cifar_c100_v1/c100_if100_core/ddpm/seed_0/ckpt_200000.pt",
        "source_step": 200000,
        "target_step": 220000,
        "continuation_updates": 20000,
        "import_policy": "explicit --import_checkpoint + --allow_legacy_resume; strict model/EMA key compatibility; fresh optimizer/scheduler; manifest records source path/SHA-256/step",
        "diagnostic": True,
        "not_a_paper_result": True,
    }
    assert {(task.method, task.seed) for task in campaign.tasks} == {("ipsvt_hybrid", 0)}
    task = replace(campaign.tasks[0], run_dir=str(tmp_path / "hybrid"))
    adapter = T2HUnifiedAdapter(ROOT)
    expected = adapter._expected_host_provenance(task, 64)
    objective_config = expected["objective_config"]
    assert objective_config["ipsvt_mode"] == "hybrid"
    assert objective_config["ipsvt_K"] == 4
    assert objective_config["ipsvt_tau"] == pytest.approx(1e-6)
    assert "ipsvt_every" not in objective_config and "ipsvt_batch" not in objective_config

    phases = adapter.phases(task)
    train = next(phase for phase in phases if phase.name == "train")
    sample = next(phase for phase in phases if phase.name == "sample")
    metrics = next(phase for phase in phases if phase.name == "metrics")
    train_text = " ".join(map(str, train.command))
    sample_text = " ".join(map(str, sample.command))
    metrics_text = " ".join(map(str, metrics.command))
    assert "--objective=ipsvt" in train_text and "--ipsvt_mode=hybrid" in train_text
    assert "--ipsvt_K=4" in train_text and "--ipsvt_tau=1e-06" in train_text
    assert "--ipsvt_lambda_aux=1.0" in train_text and "--ipsvt_lambda_svt=1.0" in train_text
    assert "--ipsvt_every" not in train_text and "--ipsvt_batch" not in train_text
    assert "--coral_projection_dim=128" in train_text
    assert "--import_checkpoint_step=200000" in train_text and "--allow_legacy_resume" in train_text
    assert "--total_steps=220001" in train_text and "--save_step=5000" in train_text
    assert "--num_images=20000" in sample_text and "--ddim_skip_step=20" in sample_text
    assert "evaluate_coral2025.py" in metrics_text
    assert "--mode detailed" in metrics_text
    assert "--per-class-output" in metrics_text and "--longtail-groups cm_three_way" in metrics_text
    assert "--kid" not in metrics_text

    # Existing response smoke remains a distinct, two-arm response campaign.
    response = load_campaign(RESPONSE_CONFIG)
    assert {task.method for task in response.tasks} == {"ipsvt_response_twin", "ipsvt_response_full"}


def test_metric_completion_accepts_the_explicit_kid_disabled_diagnostic(tmp_path):
    expected = {"checkpoint_step": 220000, "num_images": 20000}
    metric = tmp_path / "metrics.json"
    metric.write_text(json.dumps({
        "provenance": {"metric_host": "common_cifar_metrics_v2", "sample": expected},
        "metrics": {
            "FID": 1.0, "IS": 2.0, "F_8": 0.3, "F_1_8": 0.4,
            "ImprovedPrecision": 0.5, "Recall": 0.6,
        },
    }), encoding="utf-8")
    assert T2HUnifiedAdapter._metric_provenance_valid(
        metric, expected, "common_cifar_metrics_v2", require_kid=False
    )
    assert not T2HUnifiedAdapter._metric_provenance_valid(
        metric, expected, "common_cifar_metrics_v2", require_kid=True
    )


def test_host_keeps_hybrid_native_projection_and_legacy_modes_available():
    source = HOST_PATH.read_text(encoding="utf-8")
    assert 'args.ipsvt_mode in {"response", "hybrid"}' in source
    assert 'choices=["full", "twin", "clean", "response", "hybrid"]' in source
    assert 'if args.ipsvt_mode in {"response", "hybrid"}:' in source
