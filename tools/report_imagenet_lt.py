#!/usr/bin/env python3
"""Produce a small fail-closed report for the deferred ImageNet-LT cell."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx.config import load_campaign
from ltx.utils import load_runtime_env


METRICS = ("FID", "KID")


def metric_value(payload: dict, name: str) -> float | None:
    value = payload.get(name)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)) and math.isfinite(float(value["mean"])):
        return float(value["mean"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/secondary_imagenet_lt.yaml")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    campaign = load_campaign(args.config)
    load_runtime_env(campaign.root)
    out = Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "report"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for task in campaign.tasks:
        run_dir = Path(task.run_dir)
        metrics_name = str(task.eval.get("metrics_file", "metrics.imagenet.json"))
        metrics_path = run_dir / metrics_name
        payload: dict = {}
        if metrics_path.is_file():
            try:
                loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = {}
        values = {metric: metric_value(payload, metric) for metric in METRICS}
        complete = (run_dir / "SUCCESS").is_file() and all(value is not None for value in values.values())
        rows.append({
            "dataset": task.dataset.get("name"),
            "resolution": task.dataset.get("img_size"),
            "batch": task.train.get("batch_size"),
            "method": task.method,
            "seed": task.seed,
            "status": "complete" if complete else "MISSING",
            **values,
            "run_dir": task.run_dir,
        })

    fields = ["dataset", "resolution", "batch", "method", "seed", "status", *METRICS, "run_dir"]
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["method"], row["seed"])))

    aggregate: list[dict] = []
    for method in sorted({str(row["method"]) for row in rows}):
        items = [row for row in rows if row["method"] == method]
        complete_items = [row for row in items if row["status"] == "complete"]
        result = {
            "dataset": items[0]["dataset"],
            "resolution": items[0]["resolution"],
            "method": method,
            "complete": len(complete_items) == len(items) and len(items) == 1,
            "completed": len(complete_items),
            "expected": len(items),
            "seed_values": {metric: {str(row["seed"]): row[metric] for row in complete_items} for metric in METRICS},
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in complete_items if row[metric] is not None]
            result[f"{metric}_mean"] = values[0] if len(values) == 1 else (sum(values) / len(values) if values else None)
            result[f"{metric}_std"] = 0.0 if len(values) <= 1 else None
        aggregate.append(result)

    incomplete = [row for row in rows if row["status"] != "complete"]
    payload = {
        "campaign": campaign.raw["campaign"]["name"],
        "protocol": campaign.raw["campaign"].get("paper_protocol"),
        "per_seed": rows,
        "aggregate": aggregate,
        "complete": not incomplete,
        "incomplete": len(incomplete),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Deferred ImageNet-LT 64×64",
        "",
        "This is a secondary/access setting, not part of the CIFAR main table.",
        "",
        "| Method | Seed(s) | Resolution / batch target | FID ↓ | KID ↓ | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['method']} | {row['completed']}/{row['expected']} | {row['resolution']}×{row['resolution']} / {row['batch']} | "
            f"{row['FID_mean'] if row['FID_mean'] is not None else 'MISSING'} | "
            f"{row['KID_mean'] if row['KID_mean'] is not None else 'MISSING'} | "
            f"{'complete' if row['complete'] else 'MISSING'} |"
        )
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if args.wandb and str(campaign.server["runtime"].get("wandb_mode", os.getenv("WANDB_MODE", "online"))).lower() != "disabled":
        try:
            import wandb

            mode = str(campaign.server["runtime"].get("wandb_mode", os.getenv("WANDB_MODE", "online")))
            run = wandb.init(
                project=campaign.server["runtime"].get("wandb_project", "longtail"),
                entity=campaign.server["runtime"].get("wandb_entity") or None,
                name=f"{campaign.raw['campaign']['name']}-comparison",
                job_type="comparison",
                config={"campaign": campaign.raw["campaign"], "protocol": campaign.raw["campaign"].get("paper_protocol")},
                mode=mode,
            )
            run.log({
                "comparison/per_seed": wandb.Table(
                    columns=fields,
                    data=[[row.get(field) for field in fields] for row in rows],
                ),
                "comparison/summary": wandb.Table(
                    columns=sorted({key for row in aggregate for key in row}),
                    data=[[row.get(key) for key in sorted({key for item in aggregate for key in item})] for row in aggregate],
                ),
            })
            artifact = wandb.Artifact(f"{campaign.raw['campaign']['name']}-report", type="evaluation-report")
            for path in (out / "per_seed.csv", out / "table.md", out / "summary.json"):
                artifact.add_file(str(path))
            run.log_artifact(artifact)
            run.summary["comparison/incomplete_runs"] = len(incomplete)
            run.finish(exit_code=0 if not incomplete else 2)
        except Exception as exc:
            print(f"W&B comparison upload failed: {exc}")

    if incomplete:
        print(f"comparison incomplete: {len(incomplete)} task(s); see {out / 'table.md'}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
