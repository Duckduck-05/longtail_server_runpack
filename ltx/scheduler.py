from __future__ import annotations

import os
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

from .config import LoadedCampaign
from .gpu import query_gpus
from .state import StateDB


TERMINAL = {"completed", "failed", "skipped"}


class Scheduler:
    def __init__(self, campaign: LoadedCampaign):
        self.campaign = campaign
        runs_root = Path(campaign.server["runtime"]["runs_root"])
        self.campaign_dir = runs_root / campaign.raw["campaign"]["name"]
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.campaign_dir / "state.sqlite"
        self.state = StateDB(self.state_path)
        canonical_tasks = [task.to_dict() for task in campaign.tasks]
        self.campaign_fingerprint = hashlib.sha256(
            json.dumps(canonical_tasks, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.state.initialize(campaign.tasks, campaign_fingerprint=self.campaign_fingerprint)
        (self.campaign_dir / "campaign_fingerprint.txt").write_text(self.campaign_fingerprint + "\n", encoding="utf-8")
        self.processes: Dict[str, subprocess.Popen] = {}
        self.gpu_for_task: Dict[str, int] = {}
        self.stop_requested = False
        self.launcher_logs: Dict[str, object] = {}

    def _signal(self, signum, frame):
        self.stop_requested = True
        print(f"[ltx] scheduler received signal {signum}; no new jobs will be started", flush=True)

    def _allowed_gpus(self) -> List[int]:
        configured = self.campaign.server.get("machine", {}).get("gpu_ids", "auto")
        detected = [g.index for g in query_gpus()]
        if configured == "auto":
            return detected
        return [int(x) for x in configured if int(x) in detected]

    def _free_gpus(self) -> List[int]:
        machine = self.campaign.server.get("machine", {})
        min_free = float(machine.get("min_free_gpu_memory_gb", 0)) * 1024
        busy = set(self.gpu_for_task.values())
        for row in self.state.rows("running"):
            if row.get("gpu_id") is not None:
                busy.add(int(row["gpu_id"]))
        free = []
        for gpu in query_gpus():
            if gpu.index not in self._allowed_gpus() or gpu.index in busy:
                continue
            if gpu.memory_free_mb >= min_free:
                free.append(gpu.index)
        return free

    def _disk_ok(self) -> bool:
        limit = float(self.campaign.server.get("machine", {}).get("disk_stop_free_gb", 0))
        free = shutil.disk_usage(self.campaign_dir).free / (1024 ** 3)
        if free < limit:
            print(f"[ltx] disk guard: only {free:.1f} GB free (< {limit:.1f} GB); pausing launch", flush=True)
            return False
        return True

    def _launch(self, task_id: str, gpu_id: int) -> None:
        command = [
            sys.executable, "-m", "ltx.worker",
            "--state", str(self.state_path), "--task", task_id,
            "--gpu", str(gpu_id), "--root", str(self.campaign.root),
        ]
        log = (self.campaign_dir / f"worker_{task_id}.launcher.log").open("a", encoding="utf-8")
        if not self.state.claim(task_id, gpu_id, 0):
            log.close()
            return
        proc = subprocess.Popen(command, cwd=str(self.campaign.root), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.state.mark_started(task_id, proc.pid)
        self.processes[task_id] = proc
        self.gpu_for_task[task_id] = gpu_id
        self.launcher_logs[task_id] = log
        print(f"[ltx] launched task={task_id} gpu={gpu_id} pid={proc.pid}", flush=True)

    def _reap(self) -> None:
        finished = []
        for task_id, proc in self.processes.items():
            code = proc.poll()
            if code is not None:
                finished.append(task_id)
                print(f"[ltx] worker exited task={task_id} code={code}", flush=True)
        for task_id in finished:
            self.processes.pop(task_id, None)
            self.gpu_for_task.pop(task_id, None)
            handle = self.launcher_logs.pop(task_id, None)
            if handle is not None:
                handle.close()

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        self.state.reset_stale_running()
        machine = self.campaign.server.get("machine", {})
        poll = int(machine.get("poll_seconds", 20))
        allowed = self._allowed_gpus()
        if not allowed:
            print("[ltx] no NVIDIA GPUs detected", flush=True)
            return 2
        max_concurrent = machine.get("max_concurrent", "auto")
        max_concurrent = len(allowed) if max_concurrent == "auto" else int(max_concurrent)
        print(f"[ltx] campaign={self.campaign.raw['campaign']['name']} GPUs={allowed} max_concurrent={max_concurrent}", flush=True)

        while True:
            self._reap()
            recovered = self.state.reset_stale_running()
            if recovered:
                print(f"[ltx] recovered {recovered} stale worker(s)", flush=True)
            rows = self.state.rows()
            if rows and all(r["status"] in TERMINAL for r in rows):
                break
            if not self.stop_requested and self._disk_ok():
                retry_delay = float(self.campaign.server.get("retry", {}).get("retry_delay_seconds", 0))
                now = time.time()
                pending = [r for r in rows if r["status"] == "pending" or
                           (r["status"] == "retry" and now - float(r.get("finished_at") or 0) >= retry_delay)]
                running_count = len(self.state.rows("running"))
                slots = max(0, max_concurrent - running_count)
                free_gpus = self._free_gpus()[:slots]
                for row, gpu_id in zip(pending, free_gpus):
                    self._launch(row["id"], gpu_id)
            if self.stop_requested and not self.processes:
                break
            time.sleep(poll)

        rows = self.state.rows()
        completed = sum(r["status"] == "completed" for r in rows)
        failed = sum(r["status"] == "failed" for r in rows)
        print(f"[ltx] campaign finished completed={completed} failed={failed} total={len(rows)}", flush=True)
        if self.campaign.raw.get("aggregation", {}).get("enabled", False):
            try:
                from .eval import aggregate
                result = aggregate(self.campaign)
                print(f"[ltx] scientific verdict={result.get('verdict', {}).get('status', 'INCOMPLETE')}", flush=True)
            except Exception as exc:
                print(f"[ltx] aggregation failed: {exc}", flush=True)
        self.state.close()
        return 1 if failed else 0
