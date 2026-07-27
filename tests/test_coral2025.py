from pathlib import Path

from ltx.config import load_campaign
from ltx.metrics import parse_text_metrics


def test_coral2025_cifar_matrix_is_complete():
    root = Path(__file__).resolve().parents[1]
    campaign = load_campaign(root / "configs" / "coral2025_cifar.yaml")
    assert len(campaign.tasks) == 36
    by_dataset = {}
    for task in campaign.tasks:
        by_dataset.setdefault(task.dataset["name"], set()).add(task.method)
        assert task.eval["num_images"] == 50000
        assert task.eval.get("paper_metrics") or task.adapter == "coral"
        if task.adapter == "coral":
            assert task.train["total_steps"] == 150000
            assert task.eval["paper_metrics"] is True
        else:
            assert task.train["total_steps"] == 200000
    assert by_dataset == {
        "cifar10lt_if100": {"ddpm", "cbdm", "t2h", "coral"},
        "cifar10lt_if1000": {"ddpm", "cbdm", "t2h", "coral"},
        "cifar100lt_if100": {"ddpm", "cbdm", "t2h", "coral"},
    }
    assert sorted({task.seed for task in campaign.tasks}) == [0, 1, 2]


def test_coral_log_metrics_do_not_conflate_recall_columns(tmp_path):
    log = tmp_path / "res_ema_x.txt"
    log.write_text(
        "Model(EMA): IS:9.69000(0.01000), FID:5.32000\n"
        "Improved PRD:0.73000, RECALL:0.59000\n"
        "PRD PRECISION:0.97000, RECALL:0.97000\n",
        encoding="utf-8",
    )
    metrics = parse_text_metrics(log)
    assert metrics["Recall"] == 0.59
    assert metrics["F_8"] == 0.97
    assert metrics["F_1_8"] == 0.97
