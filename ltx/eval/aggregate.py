from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from jsonschema import validate as jsonschema_validate

from ..config import LoadedCampaign
from ..utils import atomic_write_json, stable_id

SEMANTIC_KEYS = ("js", "rare_mode_mass", "coarse_consistency", "memorization")
GEN_KEYS = ("FID", "KID", "Recall", "Precision", "IS")


def _finite_float(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _load_semantic(run_dir: Path, schema: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    path = run_dir / "semantic_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    jsonschema_validate(payload, schema)
    point: Dict[str, float] = {k: float(payload[k]) for k in SEMANTIC_KEYS}
    point.update({k: float(v) for k, v in payload.get("generation", {}).items() if k in GEN_KEYS and _finite_float(v) is not None})
    draw_path = Path(payload["bootstrap_draws_file"])
    if not draw_path.is_absolute(): draw_path = run_dir / draw_path
    if not draw_path.exists(): raise FileNotFoundError(f"bootstrap_draws_file missing: {draw_path}")
    with np.load(draw_path, allow_pickle=False) as z:
        draws = {k: np.asarray(z[k], dtype=np.float64) for k in z.files if k in set(SEMANTIC_KEYS + GEN_KEYS)}
    for k in SEMANTIC_KEYS:
        if k not in draws or draws[k].ndim != 1 or len(draws[k]) < 100:
            raise ValueError(f"{draw_path}: required aligned 1D bootstrap array {k} with >=100 draws")
        if not np.all(np.isfinite(draws[k])):
            raise ValueError(f"{draw_path}: non-finite {k} draws")
    lengths = {len(v) for v in draws.values()}
    if len(lengths) != 1: raise ValueError(f"{draw_path}: bootstrap arrays are not aligned: lengths={lengths}")
    return point, draws


def _ci(values: np.ndarray, level: float) -> List[float]:
    alpha = 1.0 - level
    return [float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))]


def _hierarchical_bootstrap(data: Dict[str, Dict[int, Dict[str, np.ndarray]]], arms: Iterable[str],
                            reps: int, level: float, seed: int = 20260727) -> Dict[str, Any]:
    arms = list(arms)
    common_seeds = sorted(set.intersection(*(set(data[a]) for a in arms)))
    if len(common_seeds) < 3:
        raise ValueError(f"need all paired >=3 model seeds; common={common_seeds}")
    rng = np.random.default_rng(seed)
    endpoints = {k: np.empty(reps, dtype=np.float64) for k in (
        "R_gen", "perm_minus_predictive_js", "predictive_minus_lt_rare",
        "predictive_minus_lt_consistency", "predictive_minus_lt_memorization",
        "predictive_minus_pointfit_js", "predictive_minus_lt_FID", "predictive_minus_lt_KID",
        "predictive_minus_lt_Recall"
    )}
    for r in range(reps):
        chosen = rng.choice(common_seeds, size=len(common_seeds), replace=True)
        means: Dict[str, Dict[str, float]] = {a: {} for a in arms}
        for a in arms:
            keys = set.intersection(*(set(data[a][int(s)]) for s in chosen))
            for key in keys:
                vals = []
                for s in chosen:
                    arr = data[a][int(s)][key]
                    vals.append(arr[rng.integers(0, len(arr))])
                means[a][key] = float(np.mean(vals))
        lt, oracle, pred, perm, point = [means[a] for a in arms]
        denom = lt["js"] - oracle["js"]
        endpoints["R_gen"][r] = (lt["js"] - pred["js"]) / denom if abs(denom) > 1e-12 else np.nan
        endpoints["perm_minus_predictive_js"][r] = perm["js"] - pred["js"]
        endpoints["predictive_minus_lt_rare"][r] = pred["rare_mode_mass"] - lt["rare_mode_mass"]
        endpoints["predictive_minus_lt_consistency"][r] = pred["coarse_consistency"] - lt["coarse_consistency"]
        endpoints["predictive_minus_lt_memorization"][r] = pred["memorization"] - lt["memorization"]
        endpoints["predictive_minus_pointfit_js"][r] = point["js"] - pred["js"]
        for key in ("FID", "KID", "Recall"):
            out = f"predictive_minus_lt_{key}"
            endpoints[out][r] = pred.get(key, np.nan) - lt.get(key, np.nan)
    result = {}
    for key, arr in endpoints.items():
        arr = arr[np.isfinite(arr)]
        result[key] = {"mean": float(arr.mean()) if len(arr) else None,
                       "ci": _ci(arr, level) if len(arr) else None,
                       "n_bootstrap": int(len(arr))}
    return result


def _stage_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_seeds": len(rows), "seeds": sorted(r["seed"] for r in rows)}
    keys = sorted({k for r in rows for k, v in r["point"].items() if _finite_float(v) is not None})
    for key in keys:
        vals = np.array([r["point"][key] for r in rows if key in r["point"]], dtype=float)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "values": vals.tolist()}
    return out


def _write_tables(out_dir: Path, stage: str, stage_data: Dict[str, Any], endpoints: Dict[str, Any], verdict: Dict[str, Any]):
    csv_path = out_dir / "semantic_main_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["method", "n_seeds", *SEMANTIC_KEYS, *GEN_KEYS])
        for method, d in sorted(stage_data.items()):
            w.writerow([method, d["n_seeds"], *[d.get(k, {}).get("mean", "") for k in SEMANTIC_KEYS + GEN_KEYS]])
    md = ["# Decisive semantic gate", "", f"**Verdict: {verdict['status']}**", "", "| Method | JS ↓ | Rare mass ↑ | Coarse consistency ↑ | Memorization | FID ↓ | KID ↓ | Recall ↑ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, d in sorted(stage_data.items()):
        def cell(k):
            x = d.get(k); return "—" if not x else f"{x['mean']:.5g} ± {x['std']:.2g}"
        md.append(f"| {method} | {cell('js')} | {cell('rare_mode_mass')} | {cell('coarse_consistency')} | {cell('memorization')} | {cell('FID')} | {cell('KID')} | {cell('Recall')} |")
    md += ["", "## Hierarchical bootstrap endpoints", ""]
    for k, v in endpoints.items(): md.append(f"- `{k}`: mean={v['mean']}, CI={v['ci']}, B={v['n_bootstrap']}")
    md += ["", "## Verdict reasons", ""] + [f"- {x}" for x in verdict["reasons"]]
    (out_dir / "semantic_main_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _wandb_summary(campaign: LoadedCampaign, summary: Dict[str, Any], out_dir: Path) -> None:
    if campaign.server["runtime"].get("wandb_mode", os.getenv("WANDB_MODE", "online")) == "disabled": return
    try:
        import wandb
        run = wandb.init(project=campaign.server["runtime"].get("wandb_project", "longtail"),
                         entity=campaign.server["runtime"].get("wandb_entity") or None,
                         id=stable_id(campaign.raw["campaign"]["name"], "aggregate", length=16), resume="allow",
                         name=f"{campaign.raw['campaign']['name']}-aggregate", job_type="aggregate",
                         group=campaign.raw["campaign"]["name"], dir=str(out_dir),
                         mode=campaign.server["runtime"].get("wandb_mode", "online"))
        flat = {f"endpoint/{k}": v["mean"] for k, v in summary.get("endpoints", {}).items() if v.get("mean") is not None}
        flat["verdict/pass"] = int(summary.get("verdict", {}).get("status") == "PASS")
        run.log(flat); run.summary.update(flat); run.summary["verdict"] = summary.get("verdict", {}).get("status")
        artifact = wandb.Artifact(f"{campaign.raw['campaign']['name']}-aggregate", type="evaluation")
        artifact.add_dir(str(out_dir)); run.log_artifact(artifact); run.finish()
    except Exception as exc:
        print(f"[ltx] aggregate W&B upload skipped: {exc}")


def aggregate(campaign: LoadedCampaign) -> Dict[str, Any]:
    schema = json.loads((campaign.root / "contracts" / "semantic_metrics.schema.json").read_text(encoding="utf-8"))
    rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    draws: Dict[str, Dict[int, Dict[str, np.ndarray]]] = defaultdict(dict)
    load_errors = []
    cfg = campaign.raw.get("aggregation", {}); primary_stage = cfg.get("semantic_primary_stage", "decisive_semantic_gate")
    for task in campaign.tasks:
        run_dir = Path(task.run_dir)
        if task.stage == primary_stage:
            try:
                point, bdraws = _load_semantic(run_dir, schema)
                rows[task.stage][task.method].append({"seed": task.seed, "point": point})
                draws[task.method][task.seed] = bdraws
            except Exception as exc:
                load_errors.append(f"{task.method}/seed{task.seed}: {exc}")
        else:
            path = run_dir / "metrics.collected.json"
            if path.exists():
                p = json.loads(path.read_text(encoding="utf-8"))
                numeric = {k: float(v) for k, v in p.items() if _finite_float(v) is not None}
                rows[task.stage][task.method].append({"seed": task.seed, "point": numeric})

    summary: Dict[str, Any] = {"campaign": campaign.raw["campaign"]["name"], "stages": {}, "load_errors": load_errors}
    for stage, methods in rows.items():
        summary["stages"][stage] = {m: _stage_summary(rs) for m, rs in methods.items()}

    arms = [cfg.get(x) for x in ("baseline_arm", "oracle_arm", "inferred_arm", "permutation_arm", "pointfit_arm")]
    verdict = {"status": "INCOMPLETE", "reasons": []}; endpoints = {}
    try:
        if load_errors: raise ValueError("missing/invalid semantic outputs: " + "; ".join(load_errors))
        if any(a not in draws for a in arms): raise ValueError(f"missing decisive arms; required={arms}, found={list(draws)}")
        endpoints = _hierarchical_bootstrap(draws, arms, int(cfg.get("bootstrap_repetitions", 10000)), float(cfg.get("confidence_level", 0.95)))
        ni = cfg.get("noninferiority", {}); reasons = []
        def ci(name): return endpoints[name]["ci"]
        if ci("R_gen") is None or ci("R_gen")[0] <= 0: reasons.append("R_gen 95% CI does not exclude zero")
        if ci("perm_minus_predictive_js") is None or ci("perm_minus_predictive_js")[0] <= 0: reasons.append("predictive does not beat matched permutation in JS")
        if ci("predictive_minus_lt_rare") is None or ci("predictive_minus_lt_rare")[0] <= 0: reasons.append("rare-mode mass gain is not certified")
        if ci("predictive_minus_lt_consistency")[0] < -float(ni.get("coarse_consistency_absolute_drop", 0.01)): reasons.append("coarse consistency failed non-inferiority")
        if ci("predictive_minus_lt_memorization")[1] > float(ni.get("memorization_absolute_increase", 0.01)): reasons.append("memorization increase exceeds margin")
        primary = summary["stages"][primary_stage]
        lt, pred = primary[arms[0]], primary[arms[2]]
        if "FID" not in lt or "FID" not in pred or "KID" not in lt or "KID" not in pred or "Recall" not in lt or "Recall" not in pred:
            reasons.append("external FID/KID/Recall safety metrics missing")
        else:
            if pred["FID"]["mean"] > lt["FID"]["mean"] * (1 + float(ni.get("fid_relative_increase", 0.05))): reasons.append("FID safety margin failed")
            base_kid = max(abs(lt["KID"]["mean"]), 1e-12)
            if pred["KID"]["mean"] - lt["KID"]["mean"] > base_kid * float(ni.get("kid_relative_increase", 0.10)): reasons.append("KID safety margin failed")
            if pred["Recall"]["mean"] < lt["Recall"]["mean"] - float(ni.get("recall_absolute_drop", 0.01)): reasons.append("Recall safety margin failed")
        verdict = {"status": "PASS" if not reasons else "KILL", "reasons": reasons}
        # Point-fit is a scope gate, not a kill gate.
        if endpoints["predictive_minus_pointfit_js"]["ci"] and endpoints["predictive_minus_pointfit_js"]["ci"][0] <= 0:
            verdict["scope_note"] = "Predictive averaging is not certified better than point-fit; do not claim that component."
    except Exception as exc:
        verdict = {"status": "INCOMPLETE", "reasons": [str(exc)]}
    summary["endpoints"] = endpoints; summary["verdict"] = verdict

    out_dir = Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True); atomic_write_json(out_dir / "summary.json", summary)
    if primary_stage in summary["stages"]:
        _write_tables(out_dir, primary_stage, summary["stages"][primary_stage], endpoints, verdict)
    _wandb_summary(campaign, summary, out_dir)
    return summary
