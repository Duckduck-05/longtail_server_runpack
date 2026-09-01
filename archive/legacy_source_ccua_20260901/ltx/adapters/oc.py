from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .base import Adapter, Phase, resolve_inception_batch_size, resolve_num_workers
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
        inception_batch = resolve_inception_batch_size(task.eval)
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
                     f"--num_workers={resolve_num_workers(task.train, 4)}"]
        # Pin the backbone explicitly rather than trusting flag defaults to
        # keep agreeing with the other repos across vendored-source updates.
        train_cmd += [f"--ch={task.train['ch']}"] if "ch" in task.train else []
        train_cmd += [f"--ch_mult={v}" for v in task.train.get("ch_mult", [])]
        train_cmd += [f"--attn={v}" for v in task.train.get("attn", [])]
        for name in ("num_res_blocks", "ema_decay"):
            if name in task.train:
                train_cmd.append(f"--{name}={task.train[name]}")
        latest = self.latest(run_dir, target)
        if latest:
            train_cmd += ["--resume", f"--resume_ckpt={run_dir}", f"--ckpt_step={latest}"]
        phases = [Phase("train", train_cmd, repo, skip_if_exists=[run_dir / f"ckpt_{target}.pt"])]

        # One trained checkpoint can be sampled at several guidance strengths:
        # omega only affects sampling. A single scale keeps the historical file
        # names so existing runs still resume; a sweep suffixes each artefact.
        scales = task.method_config.get(
            "guidance_scales",
            [task.method_config.get("guidance_scale", task.eval.get("guidance_scale", 1.5))],
        )
        if not isinstance(scales, list):
            scales = [scales]
        single = len(scales) == 1
        shlex = __import__("shlex")
        for omega in scales:
            suffix = "" if single else f"_w{omega}"
            marker = run_dir / f"OC_EVAL_DONE{suffix}"
            samples = run_dir / f"t2h_samples{suffix}.npy"
            labels = Path(str(samples) + ".labels.npy")
            eval_core = [py, "ddpm_gen.py", "--eval", f"--ckpt_step={target}",
                         f"--w={omega}", "--conditional", "--cfg",
                         f"--num_class={task.dataset.get('num_class',10)}", f"--logdir={run_dir}",
                         f"--num_images={task.eval.get('num_images',50000)}", f"--batch_size={batch}",
                         f"--sample_method={task.eval.get('sample_method', 'ddim')}", f"--ddim_skip_step={task.eval.get('ddim_skip_step',10)}",
                         "--sample_only", f"--sample_output={samples}"]
            if task.eval.get("uniform_labels", False):
                eval_core.append("--uniform_labels")
            # marker is written only after a successful full evaluation.
            quoted = " ".join(shlex.quote(str(x)) for x in eval_core)
            phases.append(Phase(f"eval{suffix}", ["bash", "-lc", f"{quoted} && touch {shlex.quote(str(marker))}"], repo,
                                skip_if_exists=[marker]))
            if not task.eval.get("paper_metrics", False):
                continue
            metrics_file = str(task.eval.get("metrics_file", "metrics.paper.json"))
            if not single:
                metrics_file = metrics_file.replace(".json", f"{suffix}.json")
            metric_cmd = [py, str(self.root / "tools" / "evaluate_coral2025.py"),
                "--repo", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch"), "--data-type", str(task.dataset["data_type"]),
                "--samples", str(samples), "--labels", str(labels), "--metrics-root", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch" / "stats"),
                "--output", str(run_dir / metrics_file),
                "--inception-batch-size", str(inception_batch)]
            if task.eval.get("kid", False):
                metric_cmd += ["--kid", "--kid-subsets", str(task.eval.get("kid_subsets", 100)),
                               "--kid-subset-size", str(task.eval.get("kid_subset_size", 1000)),
                               "--kid-seed", str(task.eval.get("kid_seed", 2026))]
            per_class_file = str(task.eval.get("per_class_metrics_file", "")).strip()
            if per_class_file and single:
                metric_cmd += ["--per-class-output", str(run_dir / per_class_file),
                               "--longtail-groups", str(task.eval.get("longtail_groups", "none"))]
            outputs = [run_dir / metrics_file]
            if per_class_file and single:
                outputs.append(run_dir / per_class_file)
            phases.append(Phase(f"paper_metrics{suffix}", metric_cmd, self.root, skip_if_exists=outputs))
        return phases
