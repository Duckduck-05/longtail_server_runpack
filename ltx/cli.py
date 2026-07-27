from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from .config import load_campaign
from .eval import aggregate
from .preflight import run_preflight
from .scheduler import Scheduler
from .state import StateDB
from .utils import load_runtime_env


def campaign_state(campaign):
    return Path(campaign.server["runtime"]["runs_root"]) / campaign.raw["campaign"]["name"] / "state.sqlite"


def cmd_preflight(args) -> int:
    campaign = load_campaign(args.config)
    checks = run_preflight(campaign)
    order = {"ERROR": 0, "WARN": 1, "PASS": 2}
    for c in sorted(checks, key=lambda x: (order.get(x.level, 9), x.name)):
        print(f"[{c.level:5}] {c.name:24} {c.message}")
    errors = sum(c.level == "ERROR" for c in checks)
    warns = sum(c.level == "WARN" for c in checks)
    print(f"\nPreflight: {len(checks)-errors-warns} PASS, {warns} WARN, {errors} ERROR")
    return 1 if errors else 0


def cmd_plan(args) -> int:
    campaign = load_campaign(args.config)
    print(f"Campaign: {campaign.raw['campaign']['name']} ({len(campaign.tasks)} tasks)")
    for task in campaign.tasks:
        weight = task.method_config.get("weight_file") or task.method_config.get("generated_weight") or "-"
        print(f"p={task.priority:3d} {task.id} {task.stage:32} {task.method:16} seed={task.seed} adapter={task.adapter} weight={weight}")
    return 0


def cmd_run(args) -> int:
    campaign = load_campaign(args.config)
    if not args.skip_preflight:
        checks = run_preflight(campaign)
        errors = [c for c in checks if c.level == "ERROR"]
        if errors:
            for c in errors:
                print(f"[ERROR] {c.name}: {c.message}", file=sys.stderr)
            print("Refusing to launch. Fix preflight or use --skip-preflight only for an explicit smoke/debug run.", file=sys.stderr)
            return 2
    return Scheduler(campaign).run()


def cmd_status(args) -> int:
    while True:
        campaign = load_campaign(args.config)
        path = campaign_state(campaign)
        if not path.exists():
            print("No state database yet.")
            return 1
        db = StateDB(path); rows = db.rows(); counts = Counter(r["status"] for r in rows)
        if getattr(args, "watch", 0):
            print("\033[2J\033[H", end="")
        print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(f"{'STATUS':10} {'GPU':4} {'ATT':3} {'STAGE':31} {'METHOD':15} {'SEED':4} MESSAGE")
        for row in rows:
            task = json.loads(row["payload"])
            print(f"{row['status'][:10]:10} {str(row.get('gpu_id') if row.get('gpu_id') is not None else '-'):4} "
                  f"{row['attempt']:3d} {task['stage'][:31]:31} {task['method'][:15]:15} {task['seed']:4d} {(row.get('message') or '')[:80]}")
        db.close()
        if not getattr(args, "watch", 0): return 0
        if rows and all(r["status"] in {"completed","failed","skipped"} for r in rows): return 0
        time.sleep(args.watch)


def cmd_retry_failed(args) -> int:
    campaign = load_campaign(args.config); path = campaign_state(campaign)
    if not path.exists(): print("No state database yet."); return 1
    db = StateDB(path); n = db.retry_failed(getattr(args, "stage", None)); db.close()
    print(f"Requeued {n} failed task(s).")
    return 0


def cmd_stop(args) -> int:
    campaign = load_campaign(args.config)
    path = campaign_state(campaign)
    if not path.exists():
        print("No running campaign state.")
        return 1
    db = StateDB(path)
    killed = 0
    for row in db.rows("running"):
        pid = row.get("pid")
        if pid:
            try:
                os.killpg(int(pid), signal.SIGTERM)
                killed += 1
                print(f"Sent SIGTERM to task={row['id']} pid={pid}")
            except ProcessLookupError:
                pass
    db.close()
    print(f"Stopped {killed} workers. A later run resumes them from checkpoints when supported.")
    return 0


def cmd_aggregate(args) -> int:
    campaign = load_campaign(args.config)
    result = aggregate(campaign)
    print(json.dumps(result.get("verdict", {}), indent=2))
    return 0 if result.get("verdict", {}).get("status") == "PASS" else 1


def main() -> None:
    root = Path.cwd()
    load_runtime_env(root)
    parser = argparse.ArgumentParser(prog="ltx")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "plan", "stop", "aggregate"):
        p = sub.add_parser(name); p.add_argument("--config", default="configs/deadline_full.yaml")
    status = sub.add_parser("status"); status.add_argument("--config", default="configs/deadline_full.yaml")
    status.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    retry_failed = sub.add_parser("retry-failed"); retry_failed.add_argument("--config", default="configs/deadline_full.yaml")
    retry_failed.add_argument("--stage", default=None)
    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/deadline_full.yaml")
    run.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    handlers = {
        "preflight": cmd_preflight, "plan": cmd_plan, "run": cmd_run,
        "status": cmd_status, "stop": cmd_stop, "aggregate": cmd_aggregate, "retry-failed": cmd_retry_failed,
    }
    raise SystemExit(handlers[args.command](args))


if __name__ == "__main__":
    main()
