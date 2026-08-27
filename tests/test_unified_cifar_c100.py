#!/usr/bin/env python3
"""The CIFAR-100-LT campaign's own contract.

``tests/test_unified_cifar.py`` loads ``configs/unified_cifar.yaml`` and nothing
else, so it stayed green through a sampler change to this campaign that it never
looked at. These tests cover ``configs/unified_cifar_c100.yaml`` directly.

The load-bearing property is that every method reaches the *same* sampler. The
repositories spell that two different ways -- cm/oc/ccua take a skip factor,
the coral-family trainer takes a step count -- so a config can satisfy one and
silently break the other, which is exactly what an all-methods comparison cannot
survive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx.config import load_campaign  # noqa: E402
from ltx.preflight import run_preflight  # noqa: E402

CONFIG = ROOT / "configs/unified_cifar_c100.yaml"
IPSVT_ARMS = {"ipsvt", "ipsvt_twin", "ipsvt_clean"}


def _campaign():
    return load_campaign(CONFIG)


def test_contract_carries_ipsvt_and_its_two_ablation_arms():
    campaign = _campaign()
    contract = campaign.raw["fairness_contract"]
    assert IPSVT_ARMS <= set(contract["methods"])
    assert len(campaign.tasks) == len(contract["methods"]) * len(contract["seeds"]) == 27
    for arm in IPSVT_ARMS:
        seeds = sorted(t.seed for t in campaign.tasks if t.method == arm)
        assert seeds == [0, 1, 2], f"{arm}: {seeds}"


def test_every_method_samples_with_the_same_number_of_ddim_steps():
    """One sampler for the whole table, whichever way a repo spells it."""
    campaign = _campaign()
    contract = campaign.raw["fairness_contract"]
    family = contract["sampler_family"]
    assert family == "ddim_100"
    want = int(family.split("_", 1)[1])

    seen = set()
    for task in campaign.tasks:
        T = int(task.train["T"])
        if task.adapter in {"cm", "oc", "ccua"}:
            assert task.eval["sample_method"] == "ddim", task.method
            skip = int(task.eval["ddim_skip_step"])
            steps = T // skip
        else:
            steps = int(task.eval["ddim_steps"])
        assert steps == want, f"{task.method}: {steps} steps, contract says {want}"
        seen.add(steps)
    assert seen == {want}


def test_guidance_scale_follows_the_published_setting():
    """omega 1.5 is what OC_LT's sampling command and ImbDiff-CM's config use."""
    campaign = _campaign()
    for task in campaign.tasks:
        assert task.eval["guidance_scale"] == 1.5, task.method


def test_ipsvt_arms_differ_from_the_ddpm_baseline_only_by_auxiliary_flags():
    """The comparison is only clean if the arms share the baseline's code path.

    They run the same trainer through the same adapter, so anything other than
    the auxiliary flags differing would mean the table is comparing two setups
    rather than one objective against another.
    """
    campaign = _campaign()
    by_method = {t.method: t for t in campaign.tasks if t.seed == 0}
    baseline = by_method["ddpm"]
    for arm in IPSVT_ARMS:
        task = by_method[arm]
        assert task.adapter == baseline.adapter
        assert task.train == baseline.train
        assert task.dataset == baseline.dataset
        assert task.eval == baseline.eval
        flags = [str(f) for f in task.method_config.get("flags", [])]
        assert "--ipsvt" in flags
        assert all(f.startswith("--ipsvt") for f in flags), flags
    assert not by_method["ddpm"].method_config.get("flags")


def test_ipsvt_arms_pin_the_frozen_lambda_and_their_own_mode():
    campaign = _campaign()
    modes = {}
    for task in campaign.tasks:
        if task.method not in IPSVT_ARMS:
            continue
        flags = {f.split("=")[0]: f.split("=")[-1] for f in
                 (str(x) for x in task.method_config["flags"]) if "=" in f}
        modes[task.method] = flags["--ipsvt_mode"]
        if task.method == "ipsvt":
            assert flags["--ipsvt_lambda_svt"] == "1.0"
        if task.method == "ipsvt_twin":
            assert flags["--ipsvt_lambda_svt"] == "0.0"
    assert modes == {"ipsvt": "full", "ipsvt_twin": "twin", "ipsvt_clean": "clean"}


def test_coral_adapter_passes_the_step_count_through_to_the_trainer():
    """A config value nothing forwards is a setting that does not exist."""
    from ltx.adapters.coral import CoralAdapter

    campaign = _campaign()
    task = next(t for t in campaign.tasks if t.method == "ipsvt" and t.seed == 0)
    phases = CoralAdapter(ROOT).phases(task)
    evals = [p for p in phases if p.name.startswith("eval_")]
    assert evals, [p.name for p in phases]
    for phase in evals:
        cmd = " ".join(str(c) for c in phase.command)
        assert "--ddim_steps=100" in cmd, cmd
        assert "--omega=1.5" in cmd, cmd
    train = " ".join(str(c) for c in next(p for p in phases if p.name == "train").command)
    assert "--ipsvt" in train and "--ipsvt_mode=full" in train
    # the sampler belongs to evaluation only; training must be untouched by it
    assert "--ddim_steps" not in train


def test_preflight_accepts_the_campaign():
    checks = run_preflight(_campaign())
    errors = [c for c in checks if c.level == "ERROR"]
    controls = [c for c in checks if c.name == "unified-controls"]
    assert controls and controls[0].level == "PASS", controls
    assert not [c for c in errors if c.name == "unified-controls"], errors


def test_preflight_rejects_a_method_left_on_the_old_sampler():
    """The check must actually bite -- a silent pass would be worse than none."""
    campaign = _campaign()
    victim = next(t for t in campaign.tasks if t.adapter == "cm")
    victim.eval["ddim_skip_step"] = 1
    checks = run_preflight(campaign)
    controls = [c for c in checks if c.name == "unified-controls"]
    assert controls and controls[0].level == "ERROR"
    assert "sampler steps" in controls[0].message
