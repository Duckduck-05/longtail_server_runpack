import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE = Path(__file__).parents[1] / "third_party" / "T2H-unified" / "unified_objectives.py"
SPEC = importlib.util.spec_from_file_location("unified_objectives", MODULE)
OBJECTIVES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = OBJECTIVES
SPEC.loader.exec_module(OBJECTIVES)
UnifiedObjective = OBJECTIVES.UnifiedObjective


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, x, t, *, y, augm, return_mid, use_cm):
        self.calls.append((y is None, return_mid, use_cm))
        prediction = x * self.scale
        if return_mid:
            return prediction, x
        return prediction


def batch():
    x0 = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]])
    return x0, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 2, 3]), torch.ones_like(x0)


@pytest.mark.parametrize("method", ["ddpm", "t2h", "cbdm", "coral", "ccua"])
def test_native_objectives_dispatch_and_return_scalar_diagnostics(method):
    x0, y, t, noise = batch()
    objective = UnifiedObjective(
        method, T=4, num_classes=2, class_probs=torch.tensor([0.8, 0.2]),
        coral_weight=0.5, ccua_ucl_weight=0.1, ccua_alignment_weight=0.1,
    )
    model = TinyModel()
    loss, diagnostics = objective(model, x0, y, t=t, noise=noise)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert diagnostics["loss"] is loss
    assert diagnostics["dsm"].ndim == 0
    loss.backward()
    assert model.scale.grad is not None
    if method == "t2h":
        assert "t2h_transfer_fraction" in diagnostics
    if method == "cbdm":
        assert "cbdm" in diagnostics and len(model.calls) == 2
    if method == "coral":
        assert diagnostics["coral_feature_available"].item() == 1
    if method == "ccua":
        assert "ccua_ucl" in diagnostics and len(model.calls) == 2


def test_t2h_transfer_hook_replaces_only_the_target():
    x0, y, t, noise = batch()
    seen = {}

    def hook(context):
        seen.update(context)
        return torch.zeros_like(context["noise"]), {"custom": 7.0}

    loss, diagnostics = UnifiedObjective("t2h", T=4, transfer_target_hook=hook)(
        TinyModel(), x0, y, t=t, noise=noise
    )
    assert loss.item() == 0.0
    assert diagnostics["t2h_custom"].item() == 7.0
    assert seen["x_t"].shape == x0.shape


def test_ipsvt_adds_hook_to_natural_ddpm_and_cm_requires_protocol_hook():
    x0, y, t, noise = batch()

    def ipsvt_hook(context):
        assert context["base_loss"].item() == pytest.approx(1.0)
        return torch.tensor(2.0), {"ran": 1}

    loss, diagnostics = UnifiedObjective("ipsvt", T=4, ipsvt_hook=ipsvt_hook)(
        TinyModel(), x0, y, t=t, noise=noise, step=12
    )
    assert loss.item() == pytest.approx(3.0)
    assert diagnostics["ipsvt_ran"].item() == 1

    with pytest.raises(NotImplementedError, match="cm_hook"):
        UnifiedObjective("cm", T=4)(TinyModel(), x0, y, t=t, noise=noise)

    called = []
    cm_loss, cm_diagnostics = UnifiedObjective(
        "cm", T=4, cm_hook=lambda context: (called.append(context) or torch.tensor(3.0))
    )(TinyModel(), x0, y, t=t, noise=noise)
    assert cm_loss.item() == 3.0
    assert cm_diagnostics["cm_aux"].item() == 3.0
    assert len(called) == 1


def test_common_coral_model_uses_the_method_projection_head_only_for_coral():
    """The common U-Net stays plain except for CORAL's required head."""
    model_path = Path(__file__).parents[1] / "third_party" / "T2H-unified" / "model" / "model_cm.py"
    spec = importlib.util.spec_from_file_location("t2h_common_model_under_test", model_path)
    model_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(model_module)

    kwargs = dict(T=8, ch=32, ch_mult=[1, 1], attn=[0], num_res_blocks=1,
                  dropout=0.0, cond=True, augm=False, num_class=3)
    plain = model_module.UNet_CM(**kwargs)
    coral = model_module.UNet_CM(**kwargs, coral_projection_dim=7)
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([1, 3])
    y = torch.tensor([0, 2])

    plain_prediction, plain_feature = plain(x, t, y=y, return_mid=True)
    coral_prediction, coral_feature = coral(x, t, y=y, return_mid=True)
    assert plain_prediction.shape == coral_prediction.shape == x.shape
    assert plain_feature.shape == (2, 32, 4, 4)
    assert coral_feature.shape == (2, 7)
    assert coral.mean_proj is not None and coral.logvar_proj is not None
    assert plain.mean_proj is None and plain.logvar_proj is None


def test_common_host_forwards_return_mid_to_feature_methods():
    """CORAL/CCUA must reach the feature-returning branch of the U-Net."""
    host_root = Path(__file__).parents[1] / "third_party" / "T2H-unified"
    sys.path.insert(0, str(host_root))
    try:
        spec = importlib.util.spec_from_file_location("t2h_unified_main_under_test", host_root / "unified_main.py")
        host = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(host)

        class Spy(torch.nn.Module):
            def forward(self, x, t, *, y, augm, return_mid, use_cm):
                assert return_mid is True
                return x, x.mean(dim=(2, 3))

        x = torch.randn(2, 3, 4, 4)
        prediction, feature = host.model_call(Spy(), x, torch.tensor([1, 2]), return_mid=True)
        assert prediction.shape == x.shape
        assert feature.shape == (2, 3)
    finally:
        sys.path.pop(0)
