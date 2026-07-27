from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import psutil

from .adapters import make_adapter
from .config import Task
from .gpu import gpu_metrics
from .metrics import collect_metrics, latest_tensorboard_scalars
from .state import StateDB
from .utils import atomic_write_json, run_capture, shell_join, stable_id


class Monitor(threading.Thread):
    def __init__(self, task_id: str, run_dir: Path, gpu_id: int, state: StateDB, wb_run, interval: int = 30):
        super().__init__(daemon=True)
        self.task_id = task_id
        self.run_dir = run_dir
        self.gpu_id = gpu_id
        self.state = state
        self.wb_run = wb_run
        self.interval = interval
        self.stop_event = threading.Event()
        self.last_images = set()
        self.last_tb_step: Dict[str, int] = {}

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.wait(self.interval):
            try:
                disk = shutil.disk_usage(self.run_dir)
                metrics = {
                    **gpu_metrics(self.gpu_id),
                    "system/cpu_pct": psutil.cpu_percent(interval=None),
                    "system/ram_used_gb": psutil.virtual_memory().used / (1024 ** 3),
                    "system/ram_pct": psutil.virtual_memory().percent,
                    "system/disk_free_gb": disk.free / (1024 ** 3),
                    "system/elapsed_hours": (time.time() - psutil.Process(os.getpid()).create_time()) / 3600,
                }
                tb = latest_tensorboard_scalars(self.run_dir)
                for tag, (step, value) in tb.items():
                    if step > self.last_tb_step.get(tag, -1):
                        metrics[f"train/{tag}"] = value
                        metrics["train/global_step"] = max(step, metrics.get("train/global_step", 0))
                        self.last_tb_step[tag] = step
                if self.wb_run is not None and metrics:
                    self.wb_run.log(metrics)
                self.state.heartbeat(self.task_id, f"monitor step={metrics.get('train/global_step', 'n/a')}")
                if self.wb_run is not None:
                    for image in sorted(self.run_dir.rglob("*.png")):
                        key = str(image)
                        if key not in self.last_images and image.stat().st_size > 0:
                            try:
                                import wandb
                                self.wb_run.log({"samples/latest": wandb.Image(str(image), caption=image.name)})
                                self.last_images.add(key)
                            except Exception:
                                pass
            except Exception:
                continue


def init_wandb(task: Task, run_dir: Path):
    mode = task.runtime.get("wandb_mode", os.environ.get("WANDB_MODE", "online"))
    try:
        import wandb
        run_id = stable_id(task.campaign, task.id, length=16)
        return wandb.init(
            project=task.runtime.get("wandb_project", os.environ.get("WANDB_PROJECT", "longtail")),
            entity=task.runtime.get("wandb_entity") or os.environ.get("WANDB_ENTITY") or None,
            name=f"{task.stage}-{task.method}-s{task.seed}",
            id=run_id,
            resume="allow",
            dir=str(run_dir),
            mode=mode,
            group=f"{task.campaign}/{task.stage}",
            job_type="train-eval",
            tags=list(dict.fromkeys(task.tags + [task.adapter, task.method, f"seed-{task.seed}"])),
            config=task.to_dict(),
            settings=wandb.Settings(start_method="thread"),
        )
    except Exception as exc:
        print(f"[ltx] W&B disabled for this task: {exc}", flush=True)
        return None


def run_phase(phase, env: Dict[str, str], log_path: Path, state: StateDB, task_id: str, wb_run) -> int:
    if phase.skip_if_exists and all(path.exists() for path in phase.skip_if_exists):
        print(f"[ltx] skip phase={phase.name}; outputs exist", flush=True)
        if wb_run is not None:
            wb_run.log({f"phase/{phase.name}_skipped": 1})
        return 0
    phase_env = os.environ.copy()
    phase_env.update(env)
    phase_env.update(phase.env)
    command_text = shell_join(phase.command)
    print(f"[ltx] phase={phase.name} cwd={phase.cwd}\n[ltx] command={command_text}", flush=True)
    state.heartbeat(task_id, f"phase={phase.name}")
    if wb_run is not None:
        wb_run.log({"phase/name": phase.name})
        wb_run.config.update({f"command_{phase.name}": command_text}, allow_val_change=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== PHASE {phase.name} =====\n{command_text}\n")
        log.flush()
        proc = subprocess.Popen(
            phase.command, cwd=str(phase.cwd), env=phase_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, universal_newlines=True, start_new_session=False,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                if "loss=" in line or "loss:" in line:
                    match = re.search(r"loss[=:]\s*([0-9.eE+-]+)", line)
                    if match and wb_run is not None:
                        try:
                            wb_run.log({"train/loss_stdout": float(match.group(1))})
                        except ValueError:
                            pass
                if int(time.time()) % 15 == 0:
                    state.heartbeat(task_id, f"phase={phase.name} pid={proc.pid}")
        except KeyboardInterrupt:
            proc.terminate()
            raise
        return proc.wait()


def write_provenance(task: Task, run_dir: Path, root: Path) -> None:
    directory = task.repository.get("directory", task.adapter)
    repo = Path(directory)
    if not repo.is_absolute():
        repo = Path(task.runtime["repos_root"]) / repo
    payload = {
        "task_id": task.id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version, "platform": sys.platform,
        "repo": str(repo),
        "repo_commit": task.repository.get("commit", "vendored-working-tree"),
        "repo_status": "vendored; see third_party/THIRD_PARTY_MANIFEST.json",
        "orchestrator_commit": run_capture(["git", "rev-parse", "HEAD"], root) if (root / ".git").exists() else "unversioned-runpack",
    }
    try:
        import torch
        payload.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available()})
    except Exception as exc:
        payload["torch_error"] = str(exc)
    atomic_write_json(run_dir / "provenance.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    state = StateDB(Path(args.state))
    row = state.get(args.task)
    task = Task(**row["task"])
    state.mark_started(task.id, os.getpid())
    run_dir = Path(task.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "task.resolved.json", task.to_dict())
    write_provenance(task, run_dir, Path(args.root))
    (run_dir / "gpu.txt").write_text(str(args.gpu), encoding="utf-8")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_PROJECT", task.runtime.get("wandb_project", "longtail"))
    os.environ.setdefault("WANDB_DIR", str(run_dir / "wandb"))

    wb_run = init_wandb(task, run_dir)
    if wb_run is not None:
        wb_run.summary["status"] = "running"
        wb_run.summary["gpu_id"] = args.gpu

    monitor = Monitor(task.id, run_dir, args.gpu, state, wb_run, interval=int(task.runtime.get("log_system_every_seconds", 30)))
    monitor.start()
    stdout_log = run_dir / "stdout.log"
    exit_code = 1
    message = ""
    try:
        adapter = make_adapter(task.adapter, Path(args.root))
        attempt = int(row.get("attempt", 1))
        oom_sizes = task.retry.get("oom_batch_sizes", [])
        batch_size = None
        if attempt > 1 and oom_sizes:
            idx = min(attempt - 1, len(oom_sizes) - 1)
            batch_size = int(oom_sizes[idx])
        phases = adapter.phases(task, batch_size=batch_size)
        env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "LTX_TASK_ID": task.id, "LTX_RUN_DIR": str(run_dir)}
        for phase in phases:
            code = run_phase(phase, env, stdout_log, state, task.id, wb_run)
            if code != 0:
                exit_code = code
                raise RuntimeError(f"phase {phase.name} exited with {code}")
        metrics = collect_metrics(run_dir)
        atomic_write_json(run_dir / "metrics.collected.json", metrics)
        if wb_run is not None and metrics:
            wb_run.log(metrics)
            for key, value in metrics.items():
                wb_run.summary[key] = value
        (run_dir / "SUCCESS").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")
        exit_code = 0
        message = "completed"
        state.finish(task.id, 0, "completed", message)
        if wb_run is not None:
            wb_run.summary["status"] = "completed"
            if os.environ.get("LTX_WANDB_ALERTS", "critical").lower() in {"true", "all"} or (os.environ.get("LTX_WANDB_ALERTS", "critical").lower() == "critical" and "critical" in task.tags):
                try:
                    import wandb
                    wandb.alert(title=f"LongTail finished: {task.method}", text=f"{task.stage}, seed {task.seed}")
                except Exception:
                    pass
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        tail = ""
        if stdout_log.exists():
            tail = stdout_log.read_text(encoding="utf-8", errors="replace")[-10000:]
        is_oom = "out of memory" in (message + tail).lower()
        max_attempts = int(task.retry.get("max_attempts", 2))
        status = state.retry_or_fail(task.id, exit_code, message + (" [OOM]" if is_oom else ""), max_attempts, task.retry.get("retry_exit_codes"))
        if wb_run is not None:
            wb_run.summary["status"] = status
            wb_run.summary["error"] = message
            wb_run.summary["oom_detected"] = is_oom
            if os.environ.get("LTX_WANDB_ALERTS", "critical").lower() not in {"false", "off", "0"}:
                try:
                    import wandb
                    wandb.alert(title=f"LongTail {status}: {task.method}", text=f"{task.stage}, seed {task.seed}: {message}")
                except Exception:
                    pass
    finally:
        monitor.stop()
        monitor.join(timeout=5)
        if wb_run is not None:
            try:
                if task.runtime.get("upload_stdout_artifact", True) and stdout_log.exists():
                    import wandb
                    artifact = wandb.Artifact(f"logs-{task.id}", type="run-logs")
                    artifact.add_file(str(stdout_log))
                    artifact.add_file(str(run_dir / "task.resolved.json"))
                    artifact.add_file(str(run_dir / "provenance.json"))
                    if (run_dir / "metrics.collected.json").exists():
                        artifact.add_file(str(run_dir / "metrics.collected.json"))
                    wb_run.log_artifact(artifact)
                wb_run.finish(exit_code=exit_code)
            except Exception:
                pass
        state.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
