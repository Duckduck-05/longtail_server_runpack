#!/usr/bin/env python3
"""Write the single fail-closed table for Unified CIFAR Benchmark v1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx.comparison import METRIC_DIRECTIONS, mean_std, ranks
from ltx.config import load_campaign
from ltx.utils import load_runtime_env


# ``Recall`` is the improved-PRD VGG16 k=3 manifold recall.  The two F terms
# are the Inception PRD endpoints.  Keeping those names makes the JSON / W&B
# values match the evaluator exactly while the Markdown header explains them.
METRICS = ("FID", "IS", "F_8", "F_1_8", "ImprovedPrecision", "Recall")
DISPLAY = {
    "FID": "FID ↓", "IS": "IS ↑", "F_8": "F₈ ↑", "F_1_8": "F₁⁄₈ ↑",
    "ImprovedPrecision": "IPR precision ↑", "Recall": "IPR recall ↑",
}


def finite(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def read_metrics(run_dir: Path) -> dict[str, float]:
    for filename in ("metrics.unified.json", "metrics.paper.json", "metrics.collected.json"):
        path = run_dir / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("metrics", payload)
        parsed = {metric: finite(values.get(metric, values.get(f"generation/{metric}"))) for metric in METRICS}
        return {metric: value for metric, value in parsed.items() if value is not None}
    return {}


def fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "MISSING"
    return f"{mean:.4f} ± {std:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unified_cifar.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    campaign = load_campaign(args.config)
    if campaign.raw.get("campaign", {}).get("protocol") != "unified_cifar_v1":
        raise ValueError("report_unified_cifar.py only accepts protocol=unified_cifar_v1")
    load_runtime_env(campaign.root)
    output = Path(args.output) if args.output else (
        Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "report"
    )
    output.mkdir(parents=True, exist_ok=True)
    expected_seeds = sorted(map(int, campaign.raw["campaign"]["paired_seeds"]))

    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in campaign.tasks:
        run_dir = Path(task.run_dir)
        metrics = read_metrics(run_dir)
        complete = (run_dir / "SUCCESS").is_file() and all(metric in metrics for metric in METRICS)
        row = {
            "dataset": task.dataset["name"], "method": task.method, "seed": int(task.seed),
            "adapter": task.adapter, "status": "complete" if complete else "MISSING", "run_dir": str(run_dir),
            **{metric: metrics.get(metric) for metric in METRICS},
        }
        rows.append(row)
        grouped[(row["dataset"], row["method"])].append(row)

    per_seed_columns = ["dataset", "method", "adapter", "seed", "status", *METRICS, "run_dir"]
    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_columns)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["method"], row["seed"])))

    summary: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for (dataset, method), group in sorted(grouped.items()):
        values = {
            metric: {row["seed"]: float(row[metric]) for row in group
                     if row["status"] == "complete" and row[metric] is not None}
            for metric in METRICS
        }
        complete = (sorted(row["seed"] for row in group) == expected_seeds and
                    all(row["status"] == "complete" for row in group))
        result: dict[str, Any] = {
            "dataset": dataset, "method": method, "adapter": sorted({row["adapter"] for row in group}),
            "complete": complete, "completed": sum(row["status"] == "complete" for row in group),
            "expected": len(expected_seeds), "seed_values": values,
        }
        for metric in METRICS:
            result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(list(values[metric].values()))
        summary.append(result)
        by_key[(dataset, method)] = result

    for dataset in sorted({row["dataset"] for row in summary}):
        complete_rows = [row for row in summary if row["dataset"] == dataset and row["complete"]]
        for metric in METRICS:
            scores = {row["method"]: row[f"{metric}_mean"] for row in complete_rows
                      if row[f"{metric}_mean"] is not None}
            for method, rank in ranks(scores, METRIC_DIRECTIONS[metric]).items():
                by_key[(dataset, method)][f"{metric}_rank"] = rank
        for row in complete_rows:
            row["mean_metric_rank"] = sum(row[f"{metric}_rank"] for metric in METRICS) / len(METRICS)

    serializable = []
    for item in summary:
        clone = dict(item)
        clone["seed_values"] = {metric: dict(values) for metric, values in item["seed_values"].items()}
        serializable.append(clone)
    payload = {
        "protocol": "unified_cifar_v1",
        "claim_status": "UNIFIED_BASELINE_TABLE_NOT_A_PAPER_REPRODUCTION",
        "metric_definitions": {
            "FID": "balanced CIFAR-train Inception reference; lower is better",
            "IS": "Inception Score; higher is better",
            "F_8": "Inception PRD F_8; higher is better",
            "F_1_8": "Inception PRD F_1/8; higher is better",
            "ImprovedPrecision": "VGG16-fc2 improved-PRD precision, k=3; higher is better",
            "Recall": "VGG16-fc2 improved-PRD recall, k=3; higher is better",
        },
        "fairness_contract": campaign.raw.get("fairness_contract", {}),
        "per_seed": rows,
        "aggregate": serializable,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Unified CIFAR-LT baseline table",
        "",
        "This is a single new controlled protocol, not a combination or reproduction of paper tables.",
        "Every cell requires seeds 0/1/2, 200k updates, 50k exact class-uniform generated labels, and the same evaluator.",
        "Each value is mean ± sample standard deviation across the three training seeds. Missing inputs remain `MISSING`.",
        "",
        "| Data | Method | Seeds | FID ↓ | IS ↑ | F₈ ↑ | F₁⁄₈ ↑ | IPR precision ↑ | IPR recall ↑ | FID rank | Mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: (item["dataset"], item["method"])):
        metrics = [fmt(row[f"{metric}_mean"], row[f"{metric}_std"]) for metric in METRICS]
        mean_rank = row.get("mean_metric_rank")
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['completed']}/{row['expected']} | "
            + " | ".join(metrics)
            + f" | {row.get('FID_rank', '—')} | {'—' if mean_rank is None else f'{mean_rank:.2f}'} |"
        )
    lines += [
        "",
        "`IPR` is improved-PRD on VGG16 fc2 with exact k-NN radius k=3. F₈/F₁⁄₈ are Inception PRD endpoints.",
    ]
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    incomplete = [item for item in summary if not item["complete"]]
    if args.wandb:
        try:
            import wandb
            run = wandb.init(
                project=campaign.server["runtime"].get("wandb_project", "longtail"),
                entity=campaign.server["runtime"].get("wandb_entity") or None,
                name=f"{campaign.raw['campaign']['name']}-report", group=campaign.raw["campaign"]["name"],
                job_type="unified-report", mode=campaign.server["runtime"].get("wandb_mode", "online"),
                config={"campaign": campaign.raw["campaign"], "fairness_contract": campaign.raw.get("fairness_contract", {})},
            )
            run.log({"comparison/per_seed": wandb.Table(
                columns=per_seed_columns, data=[[row.get(key) for key in per_seed_columns] for row in rows]
            )})
            summary_columns = ["dataset", "method", "complete", "completed", "expected", *[
                field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std", f"{metric}_rank")
            ], "mean_metric_rank"]
            run.log({"comparison/unified_main_table": wandb.Table(
                columns=summary_columns, data=[[row.get(key) for key in summary_columns] for row in summary]
            )})
            for row in summary:
                for metric in METRICS:
                    value = row.get(f"{metric}_mean")
                    if value is not None:
                        run.summary[f"table/{row['dataset']}/{row['method']}/{metric}"] = value
            artifact = wandb.Artifact(f"{campaign.raw['campaign']['name']}-report", type="evaluation-report")
            for path in (output / "per_seed.csv", output / "table.md", output / "summary.json"):
                artifact.add_file(str(path))
            run.log_artifact(artifact)
            run.summary["table/incomplete_cells"] = len(incomplete)
            run.summary["table/claim_status"] = payload["claim_status"]
            run.finish(exit_code=0 if not incomplete else 2)
        except Exception as exc:
            print(f"[report] W&B upload failed: {exc}")

    print(f"[report] wrote {output / 'table.md'}; complete cells={len(summary) - len(incomplete)}/{len(summary)}")
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
