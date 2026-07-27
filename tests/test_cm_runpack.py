import json
from pathlib import Path

import pytest

from ltx.comparison import paired_advantage
from ltx.config import load_campaign
from ltx.preflight import _validate_imagenet_manifest, run_preflight
from ltx.state import StateDB


def _write_manifest(root: Path, manifest: Path, *, per_class: int = 1, extra_first_class: bool = False) -> None:
    rows = []
    for label in range(1000):
        for index in range(per_class):
            image = root / f"{label}_{index}.jpg"
            image.touch()
            rows.append(f"{image.name} {label}")
    if extra_first_class:
        rows.append("0_0.jpg 0")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_cm_matrix_and_paper_contract_are_locked():
    root = Path(__file__).resolve().parents[1]
    campaign = load_campaign(root / "configs/cm_imagenet_lt.yaml")
    assert len(campaign.tasks) == 48
    assert {task.method for task in campaign.tasks} == {"ddpm", "cbdm", "oc", "cm"}
    assert sorted({task.seed for task in campaign.tasks}) == [0, 1, 2]
    imagenet = [task for task in campaign.tasks if task.dataset["name"] == "imagenet_lt"]
    assert len(imagenet) == 24
    assert {task.dataset["img_size"] for task in imagenet} == {32, 64}
    contract = json.loads((root / "contracts/cm_table5.json").read_text(encoding="utf-8"))
    assert set(contract["published"]) == {"imagenet_lt@32", "imagenet_lt@64"}
    assert set(contract["published"]["imagenet_lt@32"]) == {"ddpm", "cbdm", "oc", "cm"}


def test_imagenet_reference_must_be_exactly_balanced(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    balanced = tmp_path / "balanced.txt"
    _write_manifest(root, balanced, per_class=20)
    counts = _validate_imagenet_manifest(root, balanced, "reference", require_balanced=True)
    assert set(counts.values()) == {20}

    unbalanced = tmp_path / "unbalanced.txt"
    _write_manifest(root, unbalanced, per_class=20, extra_first_class=True)
    with pytest.raises(ValueError, match="exactly class-balanced"):
        _validate_imagenet_manifest(root, unbalanced, "reference", require_balanced=True)

    wrong_cardinality = tmp_path / "wrong_cardinality.txt"
    _write_manifest(root, wrong_cardinality)
    with pytest.raises(ValueError, match="exactly 20 images/class"):
        _validate_imagenet_manifest(root, wrong_cardinality, "reference", require_balanced=True)


def test_paired_advantage_respects_metric_direction():
    # Candidate wins both: lower FID and higher Recall.
    fid = paired_advantage({0: 8.0, 1: 8.2, 2: 8.1}, {0: 9.0, 1: 9.1, 2: 9.2}, "lower")
    recall = paired_advantage({0: 0.60, 1: 0.61, 2: 0.62}, {0: 0.55, 1: 0.56, 2: 0.57}, "higher")
    assert fid["mean"] > 0 and fid["ci95_low"] > 0
    assert recall["mean"] > 0 and recall["ci95_low"] > 0


def test_required_candidate_fails_before_expensive_launch(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("LTX_CANDIDATE_METHOD", "ours")
    monkeypatch.setenv("LTX_REQUIRE_CANDIDATE_FOR_PAPER_CLAIM", "true")
    checks = run_preflight(load_campaign(root / "configs/cm_imagenet_lt.yaml"))
    assert any(check.name == "candidate-method" and check.level == "ERROR" for check in checks)


def test_state_refuses_to_mix_campaign_fingerprints(tmp_path):
    db = StateDB(tmp_path / "state.sqlite")
    try:
        db.initialize([], campaign_fingerprint="first")
        with pytest.raises(RuntimeError, match="fingerprint"):
            db.initialize([], campaign_fingerprint="second")
    finally:
        db.close()
