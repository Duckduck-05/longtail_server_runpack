"""Focused checks for the isolated T2H IP-SVT response smoke path."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from ltx.adapters.t2h_unified import T2HUnifiedAdapter
from ltx.config import load_campaign


ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT = ROOT / "third_party" / "T2H-unified"
RESPONSE_PATH = HOST_ROOT / "ipsvt_response.py"
HOST_PATH = HOST_ROOT / "unified_main.py"
CONFIG = ROOT / "configs" / "ipsvt_response_smoke_c100_if100.yaml"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RESPONSE = _load_module("ipsvt_response_under_test", RESPONSE_PATH)


class TinyConditionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embedding = torch.nn.Embedding(3, 1)
        self.scale = torch.nn.Parameter(torch.tensor(0.7))
        self.calls = 0


def _conditioned_forward(model, x, _t, condition):
    model.calls += 1
    return model.scale * x * condition[:, :, None, None]


def _response_context(model):
    x0 = torch.tensor([[[[1.0]]], [[[2.0]]]])
    t = torch.tensor([1, 2], dtype=torch.long)
    noise = torch.tensor([[[[0.25]]], [[[-0.75]]]])
    return {
        "model": model,
        "x0": x0,
        "x_t": x0 + noise,
        "t": t,
        "noise": noise,
        "y": torch.tensor([0, 2], dtype=torch.long),
    }


@pytest.mark.parametrize("variant, expected_calls", [("twin", 2), ("full", 4)])
def test_response_loss_is_one_way_natural_batch_and_uses_one_lambda(variant, expected_calls):
    torch.manual_seed(7)
    model = TinyConditionModel()
    aux = RESPONSE.IPSVTResponseAuxiliary(
        T=4, beta_1=1e-4, beta_T=0.02,
        eta_std=0.2, lambda_weight=1.75, variant=variant,
        conditioned_forward=_conditioned_forward, use_checkpoint=False,
    )

    loss, diagnostics = aux(_response_context(model))

    assert model.calls == expected_calls
    assert float(loss.detach()) == pytest.approx(float(1.75 * diagnostics["raw"]), rel=1e-6)
    assert diagnostics["lambda"].item() == pytest.approx(1.75)
    if variant == "twin":
        assert diagnostics["svt"].item() == pytest.approx(0.0)
    else:
        assert diagnostics["svt"].item() >= 0.0
    loss.backward()
    # c_tilde starts from c_y.detach(), and clean f0/f1 are no_grad teachers:
    # the auxiliary graph can update epsilon parameters but not label_embedding.
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
    assert model.label_embedding.weight.grad is None


def test_response_source_has_no_legacy_geometry_or_class_uniform_path():
    source = RESPONSE_PATH.read_text(encoding="utf-8")
    assert "c_y.detach() + eta" in source
    assert "r_c * self.eta_std / (c_y.shape[-1] ** 0.5)" in source
    assert "class_index" not in source
    assert "sample_batch" not in source
    for forbidden in ("response_gram", "off_diagonal", "F.normalize", "tau_"):
        assert forbidden not in source
    assert "f1_perturbed.float() - f0_perturbed.float()" in source


def test_native_import_is_explicit_strict_and_records_fresh_optimizer_lineage(tmp_path):
    sys.path.insert(0, str(HOST_ROOT))
    try:
        host = _load_module("t2h_unified_main_import_test", HOST_PATH)
        source_model = torch.nn.Linear(2, 2)
        source_ema = torch.nn.Linear(2, 2)
        checkpoint_path = tmp_path / "native.pt"
        torch.save({
            "net_model": source_model.state_dict(),
            "ema_model": source_ema.state_dict(),
            "step": 200000,
        }, checkpoint_path)
        args = SimpleNamespace(
            import_checkpoint=str(checkpoint_path),
            import_checkpoint_step=200000,
            import_checkpoint_sha256="",
            allow_legacy_resume=True,
        )
        target_model = torch.nn.Linear(2, 2)
        target_ema = torch.nn.Linear(2, 2)
        imported = host.import_native_checkpoint(args, target_model, target_ema)
        assert imported["kind"] == "native_weight_import_non_exact"
        assert imported["source_path"] == str(checkpoint_path.resolve())
        assert len(imported["source_sha256"]) == 64
        assert imported["source_step"] == 200000
        assert imported["optimizer"] == imported["scheduler"] == "fresh"
        for left, right in zip(target_model.parameters(), source_model.parameters()):
            assert torch.equal(left, right)

        bad_path = tmp_path / "native_bad.pt"
        bad = source_model.state_dict() | {"unexpected.weight": torch.ones(1)}
        torch.save({"net_model": bad, "ema_model": source_ema.state_dict(), "step": 200000}, bad_path)
        args.import_checkpoint = str(bad_path)
        with pytest.raises(RuntimeError, match="strict-key compatible"):
            host.import_native_checkpoint(args, torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    finally:
        sys.path.pop(0)


def test_native_and_response_models_have_strictly_matching_projection_keys():
    native_path = ROOT / "third_party" / "coral-lt-diffusion" / "model" / "model.py"
    native_model_module = _load_module("coral_native_model_response_compat", native_path)
    sys.path.insert(0, str(HOST_ROOT))
    try:
        from model.model_cm import UNet_CM

        kwargs = dict(T=8, ch=32, ch_mult=[1, 1], attn=[0], num_res_blocks=1,
                      dropout=0.1, cond=True, augm=False, num_class=3)
        native_state = native_model_module.UNet(**kwargs).state_dict()
        response_state = UNet_CM(**kwargs, coral_projection_dim=128, lora_part=[]).state_dict()
        assert native_state.keys() == response_state.keys()
        assert {name for name in native_state if name.startswith(("mean_proj.", "logvar_proj."))} == {
            "mean_proj.weight", "mean_proj.bias", "logvar_proj.weight", "logvar_proj.bias"
        }
        assert all(native_state[name].shape == response_state[name].shape for name in native_state)
    finally:
        sys.path.pop(0)


def test_smoke_config_builds_two_220k_response_tasks_and_shared_ddim50_metrics(tmp_path):
    campaign = load_campaign(CONFIG)
    assert campaign.raw["smoke_continuation"]["continuation_updates"] == 20000
    assert {(task.method, task.seed) for task in campaign.tasks} == {
        ("ipsvt_response_twin", 0), ("ipsvt_response_full", 0)
    }
    adapter = T2HUnifiedAdapter(ROOT)
    for task in campaign.tasks:
        task = replace(task, run_dir=str(tmp_path / task.method))
        phases = adapter.phases(task)
        train = next(phase for phase in phases if phase.name == "train")
        sample = next(phase for phase in phases if phase.name == "sample")
        metrics = next(phase for phase in phases if phase.name == "metrics")
        train_text = " ".join(map(str, train.command))
        sample_text = " ".join(map(str, sample.command))
        metrics_text = " ".join(map(str, metrics.command))
        assert "--objective=ipsvt" in train_text
        assert "--ipsvt_mode=response" in train_text
        assert "--ipsvt_lambda=1.0" in train_text
        assert "--ipsvt_K" not in train_text and "--ipsvt_lambda_aux" not in train_text
        assert "--import_checkpoint_step=200000" in train_text
        assert "--allow_legacy_resume" in train_text
        assert "--total_steps=220001" in train_text
        assert "--ddim_skip_step=20" in sample_text and "--num_images=20000" in sample_text
        assert "--import_checkpoint_step=200000" in sample_text
        assert "evaluate_coral2025.py" in metrics_text
