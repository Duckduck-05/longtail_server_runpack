from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .base import Adapter, Phase
from ..config import Task


class OCAdapter(Adapter):
    name = "oc"

    @staticmethod
    def latest(run_dir: Path, target: int) -> int:
        values = []
        for p in run_dir.glob("ckpt_*.pt"):
            m = re.fullmatch(r"ckpt_(\d+)\.pt", p.name)
            if m and int(m.group(1)) < target: values.append(int(m.group(1)))
        return max(values, default=0)

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        repo = self.repo_dir(task); run_dir = Path(task.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
        py = task.runtime.get("python", "python")
        target = int(task.eval.get("checkpoint_step", task.train.get("total_steps", 200000)))
        total_steps = int(task.train.get("total_steps", target)) + 1
        batch = int(batch_size or task.train.get("batch_size", 128))
        train_cmd = [py, "main.py", "--train", "--transfer_x0", "--transfer_mode=t2h",
                     f"--data_type={task.dataset.get('data_type','cifar10lt')}",
                     f"--imb_factor={task.dataset.get('imbalance_factor',0.01)}",
                     f"--num_class={task.dataset.get('num_class',10)}", f"--logdir={run_dir}",
                     "--cfg", "--conditional", f"--batch_size={batch}", f"--seed={task.seed}",
                     f"--total_steps={total_steps}", f"--save_step={task.train.get('save_step',100000)}",
                     f"--sample_step={task.train.get('sample_step',10000)}",
                     f"--lr={task.train.get('lr', 2e-4)}", f"--T={task.train.get('T', 1000)}",
                     f"--dropout={task.train.get('dropout', 0.1)}", f"--warmup={task.train.get('warmup', 5000)}",
                     f"--num_workers={task.train.get('num_workers', 4)}"]
        latest = self.latest(run_dir, target)
        if latest:
            train_cmd += ["--resume", f"--resume_ckpt={run_dir}", f"--ckpt_step={latest}"]
        marker = run_dir / "OC_EVAL_DONE"
        samples = run_dir / "t2h_samples.npy"
        labels = Path(str(samples) + ".labels.npy")
        eval_core = [py, "ddpm_gen.py", "--eval", f"--ckpt_step={target}",
                     f"--w={task.method_config.get('guidance_scale', task.eval.get('guidance_scale',1.5))}", "--conditional", "--cfg",
                     f"--num_class={task.dataset.get('num_class',10)}", f"--logdir={run_dir}",
                     f"--num_images={task.eval.get('num_images',50000)}", f"--batch_size={batch}",
                     f"--sample_method={task.eval.get('sample_method', 'ddim')}", f"--ddim_skip_step={task.eval.get('ddim_skip_step',10)}",
                     "--sample_only", f"--sample_output={samples}"]
        if task.eval.get("uniform_labels", False):
            eval_core.append("--uniform_labels")
        # marker is written only after a successful full 50k evaluation.
        quoted = " ".join(__import__("shlex").quote(str(x)) for x in eval_core)
        phases = [
            Phase("train", train_cmd, repo, skip_if_exists=[run_dir / f"ckpt_{target}.pt"]),
            Phase("eval", ["bash", "-lc", f"{quoted} && touch {__import__('shlex').quote(str(marker))}"], repo,
                  skip_if_exists=[marker]),
        ]
        if task.eval.get("paper_metrics", False):
            metrics_file = str(task.eval.get("metrics_file", "metrics.paper.json"))
            metric_cmd = [py, str(self.root / "tools" / "evaluate_coral2025.py"),
                "--repo", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch"), "--data-type", str(task.dataset["data_type"]),
                "--samples", str(samples), "--labels", str(labels), "--metrics-root", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch" / "stats"),
                "--output", str(run_dir / metrics_file)]
            if task.eval.get("kid", False):
                metric_cmd += ["--kid", "--kid-subsets", str(task.eval.get("kid_subsets", 100)),
                               "--kid-subset-size", str(task.eval.get("kid_subset_size", 1000)),
                               "--kid-seed", str(task.eval.get("kid_seed", 2026))]
            per_class_file = str(task.eval.get("per_class_metrics_file", "")).strip()
            if per_class_file:
                metric_cmd += ["--per-class-output", str(run_dir / per_class_file),
                               "--longtail-groups", str(task.eval.get("longtail_groups", "none"))]
            outputs = [run_dir / metrics_file]
            if per_class_file:
                outputs.append(run_dir / per_class_file)
            phases.append(Phase("paper_metrics", metric_cmd, self.root, skip_if_exists=outputs))
        return phases
