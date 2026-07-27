#!/usr/bin/env python3
"""Fail-closed CORAL Table-1 report with paper and candidate comparisons."""
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

from ltx.comparison import METRIC_DIRECTIONS, mean_std, paired_advantage, ranks
from ltx.config import load_campaign
from ltx.utils import load_runtime_env

METRICS = ("FID", "IS", "F_8", "Recall", "F_1_8")


def num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def read_metrics(run_dir: Path) -> dict[str, float]:
    for filename in ("metrics.paper.json", "metrics.collected.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("metrics", payload)
        return {metric: parsed for metric in METRICS
                if (parsed := num(values.get(metric, values.get(f"generation/{metric}")))) is not None}
    return {}


def fmt(mean: float | None, std: float | None, paper: float | None, delta: float | None) -> str:
    if mean is None:
        return "MISSING"
    paper_text = "—" if paper is None else f"{paper:.4f}"
    delta_text = "—" if delta is None else f"{delta:+.4f}"
    return f"{mean:.4f} ± {std:.4f} / {paper_text} / {delta_text}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coral2025_cifar.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    campaign = load_campaign(args.config)
    load_runtime_env(campaign.root)
    contract = json.loads((campaign.root / "contracts/coral2025_table1.json").read_text(encoding="utf-8"))
    comparison = campaign.raw.get("comparison", {})
    candidate = str(comparison.get("candidate_method", "")).strip()
    require_candidate = bool_value(comparison.get("require_candidate_for_paper_claim", False))
    repetitions = int(comparison.get("bootstrap_repetitions", 10_000))
    confidence = float(comparison.get("confidence_level", 0.95))
    output = Path(args.output) if args.output else Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "report"
    output.mkdir(parents=True, exist_ok=True)
    seeds = sorted(set(map(int, campaign.raw["campaign"].get("paired_seeds", []))))

    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in campaign.tasks:
        run_dir = Path(task.run_dir)
        metrics = read_metrics(run_dir)
        complete = (run_dir / "SUCCESS").exists() and all(metric in metrics for metric in METRICS)
        row = {"dataset": task.dataset["name"], "method": task.method, "seed": task.seed,
               "status": "complete" if complete else "MISSING", "run_dir": str(run_dir),
               **{metric: metrics.get(metric) for metric in METRICS}}
        rows.append(row)
        grouped[(row["dataset"], row["method"])].append(row)
    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "seed", "status", *METRICS, "run_dir"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["method"], row["seed"])))

    aggregate: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for (dataset, method), group in sorted(grouped.items()):
        seed_values = {
            metric: {int(row["seed"]): float(row[metric]) for row in group if row["status"] == "complete" and row[metric] is not None}
            for metric in METRICS
        }
        complete = sorted(row["seed"] for row in group) == seeds and all(row["status"] == "complete" for row in group)
        target = contract["published"].get(dataset, {}).get(method, {})
        result: dict[str, Any] = {"dataset": dataset, "method": method, "complete": complete,
                                  "completed": sum(row["status"] == "complete" for row in group),
                                  "expected": len(seeds), "seed_values": seed_values}
        for metric in METRICS:
            mean, std = mean_std(list(seed_values[metric].values()))
            paper = target.get(metric)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"paper_{metric}"] = paper
            result[f"delta_run_minus_paper_{metric}"] = None if mean is None or paper is None else mean - float(paper)
        aggregate.append(result)
        by_key[(dataset, method)] = result

    for dataset in sorted({row["dataset"] for row in aggregate}):
        for metric in METRICS:
            scores = {row["method"]: row[f"{metric}_mean"] for row in aggregate
                      if row["dataset"] == dataset and row["complete"] and row[f"{metric}_mean"] is not None}
            for method, rank in ranks(scores, METRIC_DIRECTIONS[metric]).items():
                by_key[(dataset, method)][f"{metric}_rank"] = rank

    config_errors: list[str] = []
    configured_methods = {task.method for task in campaign.tasks}
    if require_candidate and not candidate:
        config_errors.append("LTX_REQUIRE_CANDIDATE_FOR_PAPER_CLAIM=true but LTX_CANDIDATE_METHOD is empty")
    if candidate and candidate not in configured_methods:
        config_errors.append(f"candidate method {candidate!r} is not present in the campaign")
    if candidate in configured_methods:
        for result in aggregate:
            candidate_row = by_key.get((result["dataset"], candidate))
            for metric in METRICS:
                if result["method"] == candidate or candidate_row is None:
                    result[f"candidate_advantage_{metric}"] = None
                else:
                    result[f"candidate_advantage_{metric}"] = paired_advantage(
                        candidate_row["seed_values"][metric], result["seed_values"][metric], METRIC_DIRECTIONS[metric],
                        repetitions=repetitions, confidence_level=confidence,
                    )
    if not candidate:
        claim_status = "BASELINES_ONLY_NO_SUPERIORITY_CLAIM"
    elif config_errors:
        claim_status = "INVALID_CANDIDATE_CONFIGURATION"
    else:
        comparisons = [row["candidate_advantage_FID"] for row in aggregate if row["method"] != candidate]
        claim_status = "CANDIDATE_WINS_ALL_FID_CI" if comparisons and all(item and item["winner"] for item in comparisons) else "CANDIDATE_SUPERIORITY_NOT_ESTABLISHED"

    serializable = []
    for row in aggregate:
        clone = dict(row)
        clone["seed_values"] = {metric: dict(values) for metric, values in row["seed_values"].items()}
        serializable.append(clone)
    (output / "summary.json").write_text(json.dumps({"paper_contract": contract["paper"], "claim_status": claim_status,
        "candidate_method": candidate or None, "config_errors": config_errors, "per_seed": rows, "aggregate": serializable}, indent=2) + "\n", encoding="utf-8")

    lines = ["# CORAL 2025 CIFAR reproduction", "", f"Claim status: `{claim_status}`.",
             "Each metric cell is `this run mean ± sample std / paper / (this run − paper)`.",
             "Candidate advantage is positive when the candidate is better; `WIN` requires paired-seed bootstrap 95% CI > 0.", ""]
    for dataset in sorted({row["dataset"] for row in aggregate}):
        lines += [f"## {dataset}", "", "| Method | Seeds | FID ↓ | IS ↑ | F₈ ↑ | Recall ↑ | F₁⁄₈ ↑ | FID rank | Candidate FID advantage |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in sorted((item for item in aggregate if item["dataset"] == dataset), key=lambda item: item["method"]):
            advantage = row.get("candidate_advantage_FID")
            advantage_text = "—" if advantage is None else (
                "MISSING" if advantage["mean"] is None else
                f"{advantage['mean']:.4f} [{advantage['ci95_low']:.4f}, {advantage['ci95_high']:.4f}] {'WIN' if advantage['winner'] else '—'}"
            )
            cells = [fmt(row[f"{metric}_mean"], row[f"{metric}_std"], row[f"paper_{metric}"], row[f"delta_run_minus_paper_{metric}"])
                     for metric in METRICS]
            lines.append(f"| {row['method']} | {row['completed']}/{row['expected']} | " + " | ".join(cells) +
                         f" | {row.get('FID_rank', '—')} | {advantage_text} |")
        lines.append("")
    if config_errors:
        lines += ["## Configuration errors", "", *[f"- {error}" for error in config_errors], ""]
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    incomplete = [row for row in aggregate if not row["complete"]]
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=campaign.server["runtime"].get("wandb_project", "longtail"),
                             entity=campaign.server["runtime"].get("wandb_entity") or None,
                             name=f"{campaign.raw['campaign']['name']}-summary", group=campaign.raw["campaign"]["name"],
                             job_type="summary", mode=campaign.server["runtime"].get("wandb_mode", "online"),
                             config={"campaign": campaign.raw["campaign"], "comparison": comparison, "protocol": contract["evaluation"]})
            per_seed_columns = ["dataset", "method", "seed", "status", *METRICS, "run_dir"]
            run.log({"comparison/per_seed": wandb.Table(columns=per_seed_columns, data=[[row.get(col) for col in per_seed_columns] for row in rows])})
            table_rows = [{key: (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
                           for key, value in row.items() if key != "seed_values"} for row in serializable]
            table_columns = sorted({key for row in table_rows for key in row})
            run.log({"comparison/mean_vs_paper": wandb.Table(columns=table_columns, data=[[row.get(col) for col in table_columns] for row in table_rows])})
            artifact = wandb.Artifact(f"{campaign.raw['campaign']['name']}-report", type="evaluation-report")
            for path in (output / "per_seed.csv", output / "table.md", output / "summary.json"):
                artifact.add_file(str(path))
            run.log_artifact(artifact)
            run.summary["comparison/claim_status"] = claim_status
            run.summary["comparison/incomplete_cells"] = len(incomplete)
            run.finish(exit_code=0 if not incomplete and not config_errors else 2)
        except Exception as exc:
            print(f"[report] W&B summary upload failed: {exc}")
    print(f"[report] wrote {output / 'table.md'}; complete cells={len(aggregate) - len(incomplete)}/{len(aggregate)}")
    return 0 if not incomplete and not config_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
