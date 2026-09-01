from pathlib import Path
from ltx.config import load_campaign


def test_deadline_matrix(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("LTX_WEIGHT_RANDOM_GROUP", raising=False)
    c = load_campaign(root / "configs/deadline_full.yaml")
    assert len(c.tasks) == 57
    critical = [t for t in c.tasks if t.stage == "decisive_semantic_gate"]
    assert len(critical) == 15
    assert {t.method for t in critical} == {"lt", "oracle", "predictive", "pointfit", "permutation"}
    assert all(t.priority == 100 for t in critical)
    assert all(t.method_config.get("generated_weight") == "uniform_manifest" for t in critical if t.method == "lt")
