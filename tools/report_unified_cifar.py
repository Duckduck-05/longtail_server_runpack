#!/usr/bin/env python3
"""Write the single fail-closed table for Unified CIFAR Benchmark v1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx.comparison import METRIC_DIRECTIONS, mean_std, ranks
from ltx.config import load_campaign, LoadedCampaign
from ltx.state import StateDB
from ltx.utils import load_runtime_env


# ``Recall`` is the improved-PRD VGG16 k=3 manifold recall.  The two F terms
# are the Inception PRD endpoints.  Keeping those names makes the JSON / W&B
# values match the evaluator exactly while the Markdown header explains them.
METRICS = ("FID", "KID", "IS", "F_8", "F_1_8", "ImprovedPrecision", "Recall")
DISPLAY = {
    "FID": "FID ↓", "KID": "KID ↓", "IS": "IS ↑", "F_8": "F₈ ↑", "F_1_8": "F₁⁄₈ ↑",
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


def read_tail_breakdown(run_dir: Path, filename: str) -> dict[str, dict[str, Any]]:
    """Read per-seed CM-style group FIDs produced by the common evaluator."""
    path = run_dir / filename
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        groups = payload.get("groups", {})
        return {str(name): value for name, value in groups.items()
                if isinstance(value, dict) and finite(value.get("FID")) is not None}
    except (OSError, ValueError, TypeError):
        return {}


def fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "MISSING"
    return f"{mean:.4f} ± {std:.4f}"


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def render_task_status_table(campaign: LoadedCampaign) -> str:
    """Final per-task scheduler status, straight from state.sqlite."""
    state_path = Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "state.sqlite"
    if not state_path.is_file():
        return "(no state.sqlite yet; the campaign has not been launched)"
    db = StateDB(state_path)
    try:
        rows = db.rows()
    finally:
        db.close()
    lines = [f"{'STATUS':10} {'ATT':3} {'STAGE':31} {'METHOD':10} {'SEED':4} MESSAGE"]
    for row in sorted(rows, key=lambda r: r["id"]):
        task = json.loads(row["payload"])
        lines.append(
            f"{row['status'][:10]:10} {row['attempt']:3d} {task['stage'][:31]:31} "
            f"{task['method'][:10]:10} {task['seed']:4d} {(row.get('message') or '')[:100]}"
        )
    return "\n".join(lines)


def try_make_project_public(campaign: LoadedCampaign) -> tuple[bool, str]:
    """Best-effort: flip the W&B project to public-read so the report link
    needs no login. Never raises; on failure it returns the one-time manual
    UI path instead."""
    project = campaign.server["runtime"].get("wandb_project", "longtail")
    entity = campaign.server["runtime"].get("wandb_entity") or ""
    try:
        import wandb
        api = wandb.Api()
        entity = entity or api.default_entity
        api.client.execute(api.CREATE_PROJECT, {"entityName": entity, "name": project, "access": "USER_READ"})
        return True, f"https://wandb.ai/{entity}/{project}"
    except Exception as exc:
        entity_display = entity or "<entity>"
        return False, (
            f"could not set the W&B project public automatically ({exc}); "
            f"set it once by hand at https://wandb.ai/{entity_display}/{project}/settings"
        )


def build_wandb_report(campaign: LoadedCampaign, table_lines: list[str], tail_lines: list[str]) -> str:
    """Publish a W&B Report summarizing this campaign. Returns its URL, or ""
    if the optional wandb-workspaces package or the API call is unavailable —
    the campaign result never depends on this succeeding."""
    try:
        import wandb_workspaces.reports.v2 as wr
    except Exception as exc:
        print(f"[report] wandb-workspaces not available; skipping W&B Report ({exc})")
        return ""
    project = campaign.server["runtime"].get("wandb_project", "longtail")
    entity = campaign.server["runtime"].get("wandb_entity") or ""
    campaign_name = campaign.raw["campaign"]["name"]
    try:
        report = wr.Report(
            entity=entity,
            project=project,
            title=f"{campaign_name} — unified CIFAR-LT baseline table",
            description=campaign.raw.get("campaign", {}).get("description", ""),
            blocks=[
                wr.H1(text="Unified CIFAR-LT baseline table"),
                wr.MarkdownBlock(text="\n".join(table_lines)),
                wr.H1(text="Long-tail FID breakdown"),
                wr.MarkdownBlock(text="\n".join(tail_lines)),
                wr.PanelGrid(
                    runsets=[wr.Runset(entity=entity, project=project,
                                        filters=f'Config("campaign") == "{campaign_name}"')],
                    panels=[
                        wr.RunComparer(),
                        wr.BarPlot(title="FID (lower is better)", metrics=["generation/FID"]),
                        wr.BarPlot(title="KID (lower is better)", metrics=["generation/KID"]),
                    ],
                ),
            ],
        )
        report.save()
        return report.url
    except Exception as exc:
        print(f"[report] W&B Report creation failed: {exc}")
        return ""


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
    tail_rows: list[dict[str, Any]] = []
    for task in campaign.tasks:
        run_dir = Path(task.run_dir)
        metrics = read_metrics(run_dir)
        tail_filename = str(task.eval.get("per_class_metrics_file", "")).strip()
        tail = read_tail_breakdown(run_dir, tail_filename) if tail_filename else {}
        tail_complete = not tail_filename or set(tail) == {"Many", "Medium", "Few"}
        complete = (run_dir / "SUCCESS").is_file() and all(metric in metrics for metric in METRICS) and tail_complete
        row = {
            "dataset": task.dataset["name"], "method": task.method, "seed": int(task.seed),
            "adapter": task.adapter, "status": "complete" if complete else "MISSING", "run_dir": str(run_dir),
            **{metric: metrics.get(metric) for metric in METRICS},
        }
        rows.append(row)
        grouped[(row["dataset"], row["method"])].append(row)
        for group_name, group_values in tail.items():
            tail_rows.append({
                "dataset": task.dataset["name"], "method": task.method, "seed": int(task.seed),
                "group": group_name, "FID": finite(group_values.get("FID")),
                "generated": int(group_values.get("generated", 0)), "reference": int(group_values.get("reference", 0)),
                "run_dir": str(run_dir),
            })

    per_seed_columns = ["dataset", "method", "adapter", "seed", "status", *METRICS, "run_dir"]
    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_columns)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], row["method"], row["seed"])))

    tail_columns = ["dataset", "method", "seed", "group", "FID", "generated", "reference", "run_dir"]
    with (output / "tail_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tail_columns)
        writer.writeheader()
        writer.writerows(sorted(tail_rows, key=lambda row: (row["dataset"], row["method"], row["group"], row["seed"])))

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

    tail_summary: list[dict[str, Any]] = []
    tail_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tail_rows:
        tail_grouped[(row["dataset"], row["method"], row["group"])].append(row)
    for (dataset, method, group_name), group_rows in sorted(tail_grouped.items()):
        by_seed = {int(row["seed"]): float(row["FID"]) for row in group_rows if row["FID"] is not None}
        mean, std = mean_std(list(by_seed.values()))
        counts = {(int(row["generated"]), int(row["reference"])) for row in group_rows}
        tail_summary.append({
            "dataset": dataset, "method": method, "group": group_name,
            "completed": len(by_seed), "expected": len(expected_seeds),
            "complete": sorted(by_seed) == expected_seeds,
            "FID_mean": mean, "FID_std": std, "seed_values": by_seed,
            "sample_counts": sorted({f"generated={generated}, reference={reference}" for generated, reference in counts}),
        })

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
            "KID": "deterministic CM-style cubic-kernel MMD on the same pinned Inception features; lower is better",
            "IS": "Inception Score; higher is better",
            "F_8": "Inception PRD F_8; higher is better",
            "F_1_8": "Inception PRD F_1/8; higher is better",
            "ImprovedPrecision": "VGG16-fc2 improved-PRD precision, k=3; higher is better",
            "Recall": "VGG16-fc2 improved-PRD recall, k=3; higher is better",
        },
        "fairness_contract": campaign.raw.get("fairness_contract", {}),
        "per_seed": rows,
        "aggregate": serializable,
        "tail_breakdown": tail_summary,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Unified CIFAR-LT baseline table",
        "",
        "This is a single new controlled protocol, not a combination or reproduction of paper tables.",
        "Every cell requires seeds 0/1/2, 200k updates, 50k exact class-uniform generated labels, and the same evaluator.",
        "Each value is mean ± sample standard deviation across the three training seeds. Missing inputs remain `MISSING`.",
        "",
        "| Data | Method | Seeds | FID ↓ | KID ↓ | IS ↑ | F₈ ↑ | F₁⁄₈ ↑ | IPR precision ↑ | IPR recall ↑ | FID rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: (item["dataset"], item["method"])):
        metrics = [fmt(row[f"{metric}_mean"], row[f"{metric}_std"]) for metric in METRICS]
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['completed']}/{row['expected']} | "
            + " | ".join(metrics)
            + f" | {row.get('FID_rank', '—')} |"
        )
    lines += [
        "",
        "`IPR` is improved-PRD on VGG16 fc2 with exact k-NN radius k=3. F₈/F₁⁄₈ are Inception PRD endpoints. KID is a deterministic CM-style cubic-kernel estimate (100 subsets, at most 1,000 features each).",
    ]
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tail_lines = [
        "# Long-tail FID breakdown",
        "",
        "These are auxiliary FIDs on the **same 50k class-uniform samples** used for the main table. They are not CM Table-3 reproductions, because CM evaluates separately sampled 20k images per split.",
        "",
        "| Data | Method | Split | Seeds | FID ↓ | Generated / reference images |",
        "|---|---|---|---:|---:|---|",
    ]
    group_order = {"Many": 0, "Medium": 1, "Few": 2}
    for row in sorted(tail_summary, key=lambda item: (item["dataset"], item["method"], group_order.get(item["group"], 99))):
        tail_lines.append(
            f"| {row['dataset']} | {row['method']} | {row['group']} | {row['completed']}/{row['expected']} | "
            f"{fmt(row['FID_mean'], row['FID_std'])} | {'; '.join(row['sample_counts'])} |"
        )
    (output / "tail_breakdown.md").write_text("\n".join(tail_lines) + "\n", encoding="utf-8")

    incomplete = [item for item in summary if not item["complete"]]
    urls: dict[str, str] = {}
    visibility_note = ""
    if args.wandb:
        made_public, visibility_note = try_make_project_public(campaign)
        if made_public:
            urls["project"] = visibility_note
            print(f"[report] W&B project is public: {visibility_note}")
        else:
            print(f"[report] {visibility_note}")
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
            run.log({"comparison/tail_fid_per_seed": wandb.Table(
                columns=tail_columns, data=[[row.get(key) for key in tail_columns] for row in tail_rows]
            )})
            summary_columns = ["dataset", "method", "complete", "completed", "expected", *[
                field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std", f"{metric}_rank")
            ]]
            run.log({"comparison/unified_main_table": wandb.Table(
                columns=summary_columns, data=[[row.get(key) for key in summary_columns] for row in summary]
            )})
            for row in summary:
                for metric in METRICS:
                    value = row.get(f"{metric}_mean")
                    if value is not None:
                        run.summary[f"table/{row['dataset']}/{row['method']}/{metric}"] = value
            artifact = wandb.Artifact(f"{campaign.raw['campaign']['name']}-report", type="evaluation-report")
            for path in (output / "per_seed.csv", output / "tail_per_seed.csv", output / "table.md", output / "tail_breakdown.md", output / "summary.json"):
                artifact.add_file(str(path))
            run.log_artifact(artifact)
            run.summary["table/incomplete_cells"] = len(incomplete)
            run.summary["table/claim_status"] = payload["claim_status"]
            urls["project"] = urls.get("project") or run.get_project_url() or ""
            urls["run"] = run.get_url() or ""
            report_url = build_wandb_report(campaign, lines, tail_lines)
            if report_url:
                urls["report"] = report_url
                run.summary["table/report_url"] = report_url
                print(f"[report] W&B Report: {report_url}")
            for key, url in urls.items():
                if url:
                    run.summary[f"table/url_{key}"] = url
            run.finish(exit_code=0 if not incomplete else 2)
        except Exception as exc:
            print(f"[report] W&B upload failed: {exc}")

    payload["wandb_urls"] = urls
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    url_lines = [f"- {key}: {url}" for key, url in urls.items() if url] or ["(W&B upload not requested or unavailable)"]
    results_log = [
        f"# {campaign.raw['campaign']['name']} — results log",
        f"generated_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "",
        "## Campaign fingerprint",
        read_text_or_empty(Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "campaign_fingerprint.txt").strip() or "(not launched yet)",
        "",
        "## Vendor / environment provenance",
        read_text_or_empty(Path(campaign.server["runtime"]["repos_root"]) / "VENDOR_AND_ENV.txt").strip() or "(missing third_party/VENDOR_AND_ENV.txt)",
        "",
        "## Fairness contract",
        json.dumps(campaign.raw.get("fairness_contract", {}), indent=2, sort_keys=True),
        "",
        "## Per-task scheduler status",
        render_task_status_table(campaign),
        "",
        "## W&B links",
        *url_lines,
        "",
        *lines,
        "",
        *tail_lines,
        "",
        f"## Verdict: {payload['claim_status']}",
        f"complete cells: {len(summary) - len(incomplete)}/{len(summary)}",
    ]
    (output / "results.log").write_text("\n".join(results_log) + "\n", encoding="utf-8")

    print(f"[report] wrote {output / 'table.md'}; complete cells={len(summary) - len(incomplete)}/{len(summary)}")
    print(f"[report] wrote {output / 'results.log'}")
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
