#!/usr/bin/env python3
"""Write the single fail-closed table for Unified CIFAR Benchmark v1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx.comparison import METRIC_DIRECTIONS, mean_std, paired_advantage, ranks
from ltx.config import load_campaign, LoadedCampaign
from ltx.state import StateDB
from ltx.utils import load_runtime_env


# ``Recall`` is the improved-PRD VGG16 k=3 manifold recall.  The two F terms
# are the Inception PRD endpoints.  Keeping those names makes the JSON / W&B
# values match the evaluator exactly while the Markdown header explains them.
METRICS = ("FID", "KID", "IS", "F_8", "F_1_8", "ImprovedPrecision", "Recall")

# The reference every method in this comparison is measured against, matching
# how CBDM, T2H, CM and CORAL each frame their own gain.
BASELINE_METHOD = "ddpm"

# Shown as columns in the advantage table: exactly CM's main-table metric set
# (ICLR 2026, Tab. 2). The remaining metrics still get a full bootstrap CI in
# summary.json — a 7-metric grid on screen is noise, not evidence.
HEADLINE_METRICS = ("FID", "KID", "IS", "Recall")
COMMON_HOST_REVISION = "t2h-unified-common-v2"
COMMON_CHECKPOINT_SCHEMA = 2
COMMON_ARTIFACT_NAMESPACE = "t2h_unified_v2"
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


def _valid_common_metric_payload(payload: Any) -> bool:
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if not isinstance(provenance, dict):
        return False
    if provenance.get("metric_host") != "common_cifar_metrics_v2":
        return False
    sample = provenance.get("sample")
    valid_identity = (
        isinstance(sample, dict)
        and sample.get("host_revision") == COMMON_HOST_REVISION
        and sample.get("checkpoint_schema") == COMMON_CHECKPOINT_SCHEMA
        and sample.get("artifact_namespace") == COMMON_ARTIFACT_NAMESPACE
    )
    if not valid_identity:
        return False
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if isinstance(metrics, dict):
        # Do not let the evaluator's early headline snapshot become a
        # paper-facing final row.  The detailed common protocol includes the
        # VGG16 improved-PRD pair as well as the fast Inception metrics.
        return {
            "FID", "KID", "IS", "F_8", "F_1_8",
            "ImprovedPrecision", "Recall",
        }.issubset(metrics)
    groups = payload.get("groups") if isinstance(payload, dict) else None
    return isinstance(groups, dict) and {"Many", "Medium", "Few"}.issubset(groups)


def read_metrics(run_dir: Path, configured_filename: str = "") -> dict[str, float]:
    # A configured filename is a protocol decision.  Once present, falling
    # back to a legacy file would recreate the exact mixed-evaluator bug this
    # report is meant to prevent.
    filenames = ([configured_filename] if configured_filename else
                 ["metrics.unified.json", "metrics.paper.json", "metrics.collected.json"])
    seen = set()
    for filename in filenames:
        if not filename or filename in seen:
            continue
        seen.add(filename)
        path = run_dir / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if configured_filename and configured_filename.endswith(".v2.json") and not _valid_common_metric_payload(payload):
            return {}
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
        if filename.endswith(".v2.json") and not _valid_common_metric_payload(payload):
            return {}
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


def write_result_files(campaign: LoadedCampaign, output: Path, payload: dict, urls: dict,
                       lines: list[str], tail_lines: list[str], summary: list, incomplete: list) -> None:
    """Write summary.json and the consolidated results.log.

    Called before the artifact upload so W&B receives these files with the
    W&B URLs already embedded, rather than the pre-URL version.
    """
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
    snapshot_campaign_log(campaign, output)


def snapshot_campaign_log(campaign: LoadedCampaign, output: Path) -> Path | None:
    """Copy the live campaign stdout log next to the report.

    `scripts/run_unified_cifar.sh` tees the whole run into
    runs/<campaign>/logs/run_<ts>.log and this report is still appending to it,
    so upload a snapshot rather than the file being written: W&B hashes an
    artifact file, and a file that changes mid-upload can fail the whole
    artifact. A copy also survives the next run rotating the symlink.
    """
    latest = Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "latest.log"
    if not latest.exists():
        return None
    destination = output / "campaign_run.log"
    try:
        shutil.copyfile(latest, destination)
    except OSError as exc:
        print(f"[report] could not snapshot the campaign log: {exc}")
        return None
    return destination


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
        metrics = read_metrics(run_dir, str(task.eval.get("metrics_file", "")).strip())
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

    # Every long-tail paper in this comparison frames its gain against plain
    # DDPM, and three seeds is too few for mean±std alone to answer "is this
    # difference real?". Bootstrap the paired per-seed differences against the
    # DDPM row of the same cell, so each method carries a CI95 on its own
    # advantage rather than only an aggregate it cannot be tested against.
    for result in summary:
        baseline_row = by_key.get((result["dataset"], BASELINE_METHOD))
        for metric in METRICS:
            if result["method"] == BASELINE_METHOD or baseline_row is None:
                result[f"vs_{BASELINE_METHOD}_{metric}"] = None
                continue
            result[f"vs_{BASELINE_METHOD}_{metric}"] = paired_advantage(
                result["seed_values"][metric], baseline_row["seed_values"][metric],
                METRIC_DIRECTIONS[metric],
            )

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
        "protocol": campaign.raw["campaign"].get("protocol", "unified_cifar_v1"),
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
        "Every cell requires seeds 0/1/2, 300k updates, 50k exact class-uniform generated labels, and the same evaluator.",
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
        "",
        "Note on `Recall`: this is the Kynkäänniemi et al. improved recall on VGG16-fc2 with k=3, as used by CORAL. CBDM's published `Recall` column is a different estimator (Inception-V3 features, K=5), so its paper numbers are not directly comparable to this column.",
        "",
        f"## Advantage over {BASELINE_METHOD.upper()} (paired seeds, bootstrap CI95)",
        "",
        f"Each method minus the {BASELINE_METHOD.upper()} row of the same cell, paired on seeds 0/1/2. "
        "Positive always favours the method; `*` marks a CI95 that excludes zero. "
        "Columns match CM's main table. Full CI bounds for every metric are in `summary.json`.",
        "",
        "| Data | Method | " + " | ".join(f"Δ {DISPLAY[m]}" for m in HEADLINE_METRICS) + " |",
        "|---|---|" + "---:|" * len(HEADLINE_METRICS),
    ]
    for row in sorted(summary, key=lambda item: (item["dataset"], item["method"])):
        if row["method"] == BASELINE_METHOD:
            continue
        cells = []
        for metric in HEADLINE_METRICS:
            adv = row.get(f"vs_{BASELINE_METHOD}_{metric}")
            if not adv or adv.get("mean") is None:
                cells.append("MISSING")
            else:
                cells.append(f"{adv['mean']:+.4f}{'*' if adv['winner'] else ''}")
        lines.append(f"| {row['dataset']} | {row['method']} | " + " | ".join(cells) + " |")
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
    wrote_result_files = False
    wandb_upload_failed = False
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
            advantage_columns = ["dataset", "method", "metric", "delta", "ci95_low", "ci95_high", "significant"]
            advantage_rows = []
            for row in summary:
                if row["method"] == BASELINE_METHOD:
                    continue
                for metric in METRICS:
                    adv = row.get(f"vs_{BASELINE_METHOD}_{metric}")
                    if not adv or adv.get("mean") is None:
                        continue
                    advantage_rows.append([row["dataset"], row["method"], metric, adv["mean"],
                                           adv["ci95_low"], adv["ci95_high"], adv["winner"]])
            run.log({f"comparison/advantage_over_{BASELINE_METHOD}": wandb.Table(
                columns=advantage_columns, data=advantage_rows
            )})
            for row in summary:
                for metric in METRICS:
                    value = row.get(f"{metric}_mean")
                    if value is not None:
                        run.summary[f"table/{row['dataset']}/{row['method']}/{metric}"] = value
            run.summary["table/incomplete_cells"] = len(incomplete)
            run.summary["table/claim_status"] = payload["claim_status"]
            # Resolve every URL first, then write the files, then upload them.
            # Writing after the upload shipped a summary.json with no W&B links
            # in it and never shipped results.log at all.
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
            write_result_files(campaign, output, payload, urls, lines, tail_lines, summary, incomplete)
            wrote_result_files = True

            artifact = wandb.Artifact(f"{campaign.raw['campaign']['name']}-report", type="evaluation-report")
            # campaign_run.log is the whole-campaign stdout (bootstrap, GPU
            # packing decisions, task launches and failures) — the only place a
            # reader who cannot see the machine can diagnose a partial run.
            uploads = [output / "per_seed.csv", output / "tail_per_seed.csv", output / "table.md",
                       output / "tail_breakdown.md", output / "summary.json", output / "results.log",
                       output / "campaign_run.log"]
            for path in uploads:
                if path.is_file():
                    artifact.add_file(str(path))
            run.log_artifact(artifact)
            run.finish(exit_code=0 if not incomplete else 2)
        except Exception as exc:
            wandb_upload_failed = True
            print(f"[report] W&B upload failed: {exc}")

    # Without --wandb, or if the upload raised before reaching them, the files
    # are still the local deliverable and must exist either way.
    if not wrote_result_files:
        write_result_files(campaign, output, payload, urls, lines, tail_lines, summary, incomplete)

    print(f"[report] wrote {output / 'table.md'}; complete cells={len(summary) - len(incomplete)}/{len(summary)}")
    print(f"[report] wrote {output / 'results.log'}")
    report_mode = str(campaign.server["runtime"].get("wandb_mode", "online")).lower()
    if args.wandb and report_mode == "online" and wandb_upload_failed:
        # Keep the local report, but do not let the shell wrapper claim a
        # successful online hand-off when no W&B run was actually published.
        return 3
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
