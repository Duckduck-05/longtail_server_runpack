from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import Dict

_PATTERNS={
 "FID":re.compile(r"FID(?:/CIFAR100)?[:=]\s*([0-9.eE+-]+)"),
 "IS":re.compile(r"IS[:=]\s*([0-9.eE+-]+)"),
 "Recall":re.compile(r"(?:RECALL|Recall)[:=]\s*([0-9.eE+-]+)"),
 "Precision":re.compile(r"(?:PRECISION|Precision)[:=]\s*([0-9.eE+-]+)"),
 "KID":re.compile(r"KID[:=]\s*([0-9.eE+-]+)"),
}

_CORAL_IMPROVED_PRD = re.compile(
    r"Improved PRD:\s*([0-9.eE+-]+),\s*RECALL:\s*([0-9.eE+-]+)", re.I)
_CORAL_PRD = re.compile(
    r"PRD PRECISION:\s*([0-9.eE+-]+),\s*RECALL:\s*([0-9.eE+-]+)", re.I)

def _put(out,key,value):
    try:
        v=float(value)
        if math.isfinite(v): out[key]=v
    except Exception: pass

def parse_text_metrics(path: Path)->Dict[str,float]:
    if not path.exists(): return {}
    text=path.read_text(encoding="utf-8",errors="replace"); out={}
    for k,p in _PATTERNS.items():
        m=p.findall(text)
        if m: _put(out,k,m[-1])
    # CORAL labels the two PRD values "precision" and "recall" in its log,
    # but its paper reports them as F_8 and F_1/8.  Parse the paired lines
    # before the generic Recall regex can silently conflate two different
    # metrics.  Improved PRD recall is the table's Recall column.
    improved = _CORAL_IMPROVED_PRD.findall(text)
    if improved:
        _put(out, "ImprovedPrecision", improved[-1][0])
        _put(out, "Recall", improved[-1][1])
    prd = _CORAL_PRD.findall(text)
    if prd:
        _put(out, "F_8", prd[-1][0])
        _put(out, "F_1_8", prd[-1][1])
    return out

def _flatten(prefix,obj,out):
    if isinstance(obj,dict):
        if "mean" in obj and isinstance(obj["mean"],(int,float)):
            _put(out,prefix,obj["mean"]); return
        for k,v in obj.items(): _flatten(f"{prefix}/{k}" if prefix else str(k),v,out)
    elif isinstance(obj,(int,float)): _put(out,prefix,obj)

# Sample counts, seeds and protocol constants are provenance, not results.
# They belong in the run's config and the report's JSON, not plotted next to
# FID as if they were something a reader should compare across methods.
_BOOKKEEPING = ("num_generated", "num_reference", "generated", "reference",
                "seed", "samples", "label_histogram")


def _is_unified_v2_run(run_dir: Path) -> bool:
    """Identify the migrated host without importing the vendored trainer."""
    try:
        payload = json.loads((run_dir / "unified_host.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("host_revision") == "t2h-unified-common-v2"
        and payload.get("checkpoint_schema") == 2
    )

def _is_bookkeeping(key: str) -> bool:
    return key.rsplit("/", 1)[-1] in _BOOKKEEPING

def collect_metrics(run_dir: Path)->Dict[str,float]:
    out={}
    unified_v2 = _is_unified_v2_run(run_dir)
    semantic=run_dir/"semantic_metrics.json"
    if semantic.exists():
        p=json.loads(semantic.read_text())
        for k in ("js","rare_mode_mass","coarse_consistency","memorization","num_generated"):
            if k in p: _put(out,f"semantic/{k}",p[k])
        _flatten("generation",p.get("generation",{}),out); _flatten("safety",p.get("safety",{}),out)
    text_candidates = [] if unified_v2 else [run_dir/"stdout.log", *run_dir.glob("res_ema_*.txt"), *run_dir.glob("eval.txt")]
    for path in text_candidates:
        for k,v in parse_text_metrics(path).items(): out[f"generation/{k}"]=v
    metric_paths = run_dir.rglob("metrics*.json")
    if unified_v2:
        # A common-host run is allowed to publish only its namespaced metric
        # outputs.  In particular, do not let an old native metrics file or a
        # stale stdout line overwrite the new value in W&B.
        metric_paths = (
            path for path in metric_paths
            if path.name in {"metrics.unified.v2.json", "metrics.per_class.v2.json", "metrics.imagenet.json"}
        )
    for path in metric_paths:
        if path.name=="metrics.collected.json": continue
        try: payload = json.loads(path.read_text())
        except Exception: continue
        if not isinstance(payload, dict): continue
        if isinstance(payload.get("metrics"), dict):
            # The shared CCUA evaluator wraps FID/IS/KID/PRD in a "metrics" key
            # alongside "protocol"/"label_histogram" metadata. Flattening the
            # whole payload buried every metric two levels deep as
            # generation/metrics/FID and swept protocol constants in as if
            # they were metrics; unwrap to the clean generation/FID instead.
            _flatten("generation", payload["metrics"], out)
        elif isinstance(payload.get("groups"), dict):
            # metrics.per_class.json: keep the Many/Medium/Few tail summary,
            # skip the 100-class per_class breakdown and protocol metadata so
            # they don't flood the W&B summary with noise.
            _flatten("generation/tail", payload["groups"], out)
        else:
            _flatten("generation", payload, out)
    return {k: v for k, v in out.items() if not _is_bookkeeping(k)}

def latest_tensorboard_scalars(logdir: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception: return {}
    if not list(logdir.rglob("events.out.tfevents.*")): return {}
    try:
        acc=EventAccumulator(str(logdir),size_guidance={"scalars":1}); acc.Reload(); out={}
        for tag in acc.Tags().get("scalars",[]):
            ev=acc.Scalars(tag)
            if ev: out[tag]=(int(ev[-1].step),float(ev[-1].value))
        return out
    except Exception: return {}
