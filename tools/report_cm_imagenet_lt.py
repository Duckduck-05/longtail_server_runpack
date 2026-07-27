#!/usr/bin/env python3
"""Fail-closed released-code CM sensitivity report with paper deltas."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx.comparison import METRIC_DIRECTIONS, mean_std, paired_advantage, ranks
from ltx.config import load_campaign
from ltx.utils import load_runtime_env

METRICS = ("FID", "KID")


def metric_value(payload: dict, name: str) -> float | None:
    value = payload.get(name)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    return None


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def fmt(mean: float | None, std: float | None) -> str:
    return "MISSING" if mean is None else f"{mean:.4f} ± {std:.4f}"


def fmt_num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cm_imagenet_lt.yaml")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    campaign = load_campaign(args.config)
    load_runtime_env(campaign.root)
    contract = json.loads((campaign.root / "contracts" / "cm_table5.json").read_text(encoding="utf-8"))
    comparison = campaign.raw.get("comparison", {})
    candidate = str(comparison.get("candidate_method", "")).strip()
    require_candidate = bool_value(comparison.get("require_candidate_for_paper_claim", False))
    repetitions = int(comparison.get("bootstrap_repetitions", 10_000))
    confidence = float(comparison.get("confidence_level", 0.95))

    out = Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "report"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for task in campaign.tasks:
        metrics_path = Path(task.run_dir) / "metrics.cm.json"
        success = (Path(task.run_dir) / "SUCCESS").is_file()
        payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        values = {metric: metric_value(payload, metric) for metric in METRICS}
        complete = success and all(values[metric] is not None for metric in METRICS)
        row = {
            "dataset": task.dataset.get("name"), "resolution": task.dataset.get("img_size"),
            "method": task.method, "seed": task.seed,
            "status": "complete" if complete else "MISSING",
            "run_dir": task.run_dir, **values,
        }
        rows.append(row)
        grouped[(str(task.dataset.get("name")), int(task.dataset.get("img_size", 0)), task.method)].append(row)
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "resolution", "method", "seed", "status", *METRICS, "run_dir"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["resolution"], row["method"], row["seed"])))

    expected_seeds = sorted(set(map(int, campaign.raw["campaign"].get("paired_seeds", []))))
    summary: list[dict] = []
    summary_by_key: dict[tuple[str, int, str], dict] = {}
    for key, items in sorted(grouped.items()):
        dataset, resolution, method = key
        values_by_metric = {
            metric: {int(row["seed"]): float(row[metric]) for row in items if row["status"] == "complete" and row[metric] is not None}
            for metric in METRICS
        }
        complete = sorted(row["seed"] for row in items) == expected_seeds and all(row["status"] == "complete" for row in items)
        target = contract["published"].get(f"{dataset}@{resolution}", {}).get(method, {})
        result = {
            "dataset": dataset, "resolution": resolution, "method": method,
            "complete": complete, "completed": sum(row["status"] == "complete" for row in items),
            "expected": len(expected_seeds), "seed_values": values_by_metric,
        }
        for metric in METRICS:
            mean, std = mean_std(list(values_by_metric[metric].values()))
            paper = target.get(metric)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"paper_{metric}"] = paper
            result[f"delta_run_minus_paper_{metric}"] = None if mean is None or paper is None else mean - float(paper)
        summary.append(result)
        summary_by_key[key] = result

    for dataset, resolution in sorted({(row["dataset"], row["resolution"]) for row in summary}):
        entries = [row for row in summary if row["dataset"] == dataset and row["resolution"] == resolution and row["complete"]]
        for metric in METRICS:
            scores = {row["method"]: row[f"{metric}_mean"] for row in entries if row[f"{metric}_mean"] is not None}
            for method, rank in ranks(scores, METRIC_DIRECTIONS[metric]).items():
                summary_by_key[(dataset, resolution, method)][f"{metric}_rank"] = rank

    config_errors: list[str] = []
    configured_methods = {task.method for task in campaign.tasks}
    if require_candidate and not candidate:
        config_errors.append("LTX_REQUIRE_CANDIDATE_FOR_PAPER_CLAIM=true but LTX_CANDIDATE_METHOD is empty")
    if candidate and candidate not in configured_methods:
        config_errors.append(f"candidate method {candidate!r} is not present in the campaign")
    if candidate in configured_methods:
        for result in summary:
            candidate_row = summary_by_key.get((result["dataset"], result["resolution"], candidate))
            for metric in METRICS:
                if result["method"] == candidate or candidate_row is None:
                    result[f"candidate_advantage_{metric}"] = None
                    continue
                result[f"candidate_advantage_{metric}"] = paired_advantage(
                    candidate_row["seed_values"][metric], result["seed_values"][metric], METRIC_DIRECTIONS[metric],
                    repetitions=repetitions, confidence_level=confidence,
                )

    complete_rows = [row for row in summary if row["complete"]]
    if not candidate:
        claim_status = "BASELINES_ONLY_NO_SUPERIORITY_CLAIM"
    elif config_errors:
        claim_status = "INVALID_CANDIDATE_CONFIGURATION"
    else:
        comparisons = [row["candidate_advantage_FID"] for row in summary if row["method"] != candidate]
        claim_status = "CANDIDATE_WINS_ALL_FID_CI" if comparisons and all(item and item["winner"] for item in comparisons) else "CANDIDATE_SUPERIORITY_NOT_ESTABLISHED"
    serializable = []
    for row in summary:
        clone = dict(row)
        clone["seed_values"] = {metric: dict(values) for metric, values in row["seed_values"].items()}
        serializable.append(clone)
    payload = {
        "paper_contract": contract["paper"], "claim_status": claim_status, "candidate_method": candidate or None,
        "config_errors": config_errors, "per_seed": rows, "aggregate": serializable,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# CM released-code sensitivity comparison", "", f"Claim status: `{claim_status}`.",
             "This source-port campaign is not a literal CM paper-table reproduction; paper deltas are diagnostic only (`this run − paper`).",
             "Lower is better for FID/KID.",
             "A positive candidate advantage means the configured candidate is better; `WIN` requires paired-seed bootstrap 95% CI > 0.", ""]
    for dataset, resolution in sorted({(row["dataset"], row["resolution"]) for row in summary}):
        lines += [f"## {dataset} @ {resolution}×{resolution}", "",
                  "| Method | Seeds | FID ↓ | Paper FID | Δ FID | Rank | KID ↓ | Paper KID | Δ KID | Rank | Candidate FID advantage |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in sorted((item for item in summary if item["dataset"] == dataset and item["resolution"] == resolution), key=lambda item: item["method"]):
            advantage = row.get("candidate_advantage_FID")
            advantage_text = "—" if advantage is None else (
                "MISSING" if advantage["mean"] is None else
                f"{advantage['mean']:.4f} [{advantage['ci95_low']:.4f}, {advantage['ci95_high']:.4f}] {'WIN' if advantage['winner'] else '—'}"
            )
            lines.append(
                f"| {row['method']} | {row['completed']}/{row['expected']} | {fmt(row['FID_mean'], row['FID_std'])} | "
                f"{fmt_num(row['paper_FID'])} | {fmt_num(row['delta_run_minus_paper_FID'])} | {row.get('FID_rank', '—')} | "
                f"{fmt(row['KID_mean'], row['KID_std'])} | {fmt_num(row['paper_KID'])} | {fmt_num(row['delta_run_minus_paper_KID'])} | "
                f"{row.get('KID_rank', '—')} | {advantage_text} |"
            )
        lines.append("")
    if config_errors:
        lines += ["## Configuration errors", "", *[f"- {error}" for error in config_errors], ""]
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    incomplete = [row for row in rows if row["status"] != "complete"]
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=campaign.server["runtime"].get("wandb_project", "longtail"),
                             entity=campaign.server["runtime"].get("wandb_entity") or None,
                             name=f"{campaign.raw['campaign']['name']}-comparison", job_type="comparison",
                             config={"campaign": campaign.raw["campaign"], "comparison": comparison, "protocol": contract["evaluation"]})
            per_seed_columns = ["dataset", "resolution", "method", "seed", "status", *METRICS, "run_dir"]
            run.log({"comparison/per_seed": wandb.Table(columns=per_seed_columns, data=[[row.get(col) for col in per_seed_columns] for row in rows])})
            table_rows = [{key: (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
                           for key, value in row.items() if key != "seed_values"} for row in serializable]
            table_columns = sorted({key for row in table_rows for key in row})
            run.log({"comparison/summary": wandb.Table(columns=table_columns, data=[[row.get(col) for col in table_columns] for row in table_rows])})
            artifact = wandb.Artifact(f"{campaign.raw['campaign']['name']}-report", type="evaluation-report")
            for path in (out / "per_seed.csv", out / "table.md", out / "summary.json"):
                artifact.add_file(str(path))
            run.log_artifact(artifact)
            run.summary["comparison/claim_status"] = claim_status
            run.summary["comparison/incomplete_runs"] = len(incomplete)
            run.finish(exit_code=0 if not incomplete and not config_errors else 2)
        except Exception as exc:
            print(f"W&B comparison upload failed: {exc}")
    if incomplete or config_errors:
        raise SystemExit(f"comparison incomplete/invalid: missing={len(incomplete)} config_errors={len(config_errors)}; see {out / 'table.md'}")


if __name__ == "__main__":
    main()
